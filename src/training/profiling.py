"""Short, isolated training benchmark; never writes or resumes experiment runs."""

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

import torch

from src.progress import ConsoleProgress


class StageProfiler:
    """CUDA stream intervals, including host launch gaps; not kernel-only time."""

    STAGES = ('transfer', 'forward_loss', 'backward', 'optimizer', 'ema', 'bookkeeping')

    def __init__(self, device, warmup=8):
        self.device = torch.device(device)
        self.warmup = warmup
        self.records = []
        self.current = None
        self.started = None

    def begin(self, step):
        self.current = None
        if step < self.warmup:
            return
        if self.started is None:
            torch.cuda.synchronize(self.device)
            self.started = time.perf_counter()
        self.current = []
        self.mark()

    def mark(self):
        if self.current is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.current.append(event)

    def end(self, updated):
        if self.current is not None:
            self.records.append((self.current, updated))
            self.current = None

    def report(self):
        torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - self.started if self.started is not None else 0
        rows = []
        for events, updated in self.records:
            if len(events) != len(self.STAGES) + 1:
                raise ValueError('Incomplete profiling stage sequence')
            rows.append({**{name: events[i].elapsed_time(events[i + 1])
                            for i, name in enumerate(self.STAGES)}, 'updated': updated})
        count = len(rows)
        updates = sum(row['updated'] for row in rows)
        return {
            'batches': count, 'optimizer_updates': updates,
            'wall_ms_per_batch': wall * 1000 / max(count, 1),
            'cuda_ms_per_batch': {name: sum(row[name] for row in rows) / max(count, 1)
                                  for name in self.STAGES},
            'cuda_ms_per_update': {name: sum(row[name] for row in rows if row['updated']) / max(updates, 1)
                                   for name in ('optimizer', 'ema')},
        }


class BenchmarkBatches:
    def __init__(self, source, count, *, repeat=False, worker_profile=None, warmup=0):
        self.source, self.count, self.repeat = source, count, repeat
        self.worker_profile, self.warmup = worker_profile, warmup

    def __len__(self):
        return self.count

    def __iter__(self):
        if self.repeat:
            for _ in range(self.count):
                yield self.source
        else:
            iterator = iter(self.source)
            for index in range(self.count):
                batch = next(iterator)
                if self.worker_profile is not None:
                    self.worker_profile.consume(batch, include=index >= self.warmup)
                yield batch


def benchmark(config, *, batches=32, warmup=8, asynchronous_transfer=True, profile_data=False):
    from src.data.profiling import WorkerProfileSummary
    from src.training.base import set_random_seed
    from src.training.builders import (
        build_datasets,
        build_ema,
        build_loaders,
        build_model,
        build_optimizer,
        build_scheduler,
        configure_memory_format,
    )
    from src.training.engine import ExperimentRunner, _move_batch_to_device, train_one_epoch

    if batches <= 0 or warmup < 0:
        raise ValueError('batches must be positive and warmup nonnegative')
    runner = ExperimentRunner(config)
    if runner.device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('This benchmark requires CUDA')
    set_random_seed(config.seed)
    train_rows, val_rows = runner._split_data()
    datasets = build_datasets(config, runner.data_workspace, train_rows, val_rows)
    datasets[0].profile_data = profile_data
    worker_profile = WorkerProfileSummary() if profile_data else None
    loader, val_loader = build_loaders(config.train, *datasets)
    total = warmup + batches
    if total > len(loader):
        raise ValueError('Benchmark exceeds available training batches')
    # Keep one CPU batch for the isolated replay after workers have been released.
    frozen = next(iter(loader))
    if worker_profile is not None:
        worker_profile.consume(frozen, include=False)
    results = {}
    for mode in ('loader', 'gpu_replay'):
        if mode == 'gpu_replay':
            del loader, val_loader, datasets
            gc.collect()
            torch.cuda.empty_cache()
            frozen = _move_batch_to_device(frozen, runner.device)
            frozen['image'] = frozen['image'].contiguous(memory_format=torch.channels_last)
            source = BenchmarkBatches(frozen, total, repeat=True)
        else:
            source = BenchmarkBatches(loader, total, worker_profile=worker_profile, warmup=warmup)
        set_random_seed(config.seed)
        # Fresh disposable model/optimizer per phase, no pretrained downloads or checkpoints.
        model = configure_memory_format(build_model(config.model, aux_weight=config.loss.aux_weight, pretrained=False).to(runner.device))
        optimizer = build_optimizer(config.train, model)
        ema = build_ema(config.train, model)
        profiler = StageProfiler(runner.device, warmup)
        ConsoleProgress.info(f'Profile {mode}: warmup={warmup}, measured={batches}')
        train_one_epoch(model=model, loader=source, optimizer=optimizer,
                        scheduler=build_scheduler(config.train, optimizer, total,
                                                  full_steps_per_epoch=total),
                        scaler=runner.amp.scaler(), ema=ema, amp=runner.amp,
                        config=config, device=runner.device, profiler=profiler,
                        asynchronous_transfer=asynchronous_transfer and mode == 'loader')
        results[mode] = profiler.report()
        if mode == 'loader' and worker_profile is not None:
            results[mode]['worker_profile'] = worker_profile.report()
        ConsoleProgress.info(json.dumps(results[mode], ensure_ascii=False))
        del source, model, optimizer, ema
    return {'gpu': torch.cuda.get_device_name(runner.device), 'config': config.to_dict(),
            'warmup': warmup, 'asynchronous_transfer': asynchronous_transfer, 'results': results,
            'note': 'CUDA stream intervals include host launch gaps. With optimized transport, transfer measures exposed compute-stream wait, not total copy duration. Replay uses one fixed batch; no run is saved.'}


def benchmark_workers(config, workers, *, repeats=2, batches=32, warmup=8):
    if not workers or len(set(workers)) != len(workers) or any(w < 0 for w in workers) or repeats < 1:
        raise ValueError('Provide distinct nonnegative worker counts and positive repeats')
    runs = {w: [] for w in workers}
    for repeat in range(repeats):
        # Reverse alternate passes to reduce ordering/cache bias.
        for count in (workers if repeat % 2 == 0 else list(reversed(workers))):
            ConsoleProgress.info(f'Workers={count}, repeat={repeat + 1}/{repeats}')
            cfg = replace(config, train=replace(config.train, workers=count))
            runs[count].append(benchmark(cfg, batches=batches, warmup=warmup))
    ranking = []
    for count, reports in runs.items():
        times = [r['results']['loader']['wall_ms_per_batch'] for r in reports]
        ranking.append({'workers': count, 'loader_median_ms': median(times),
                        'loader_min_ms': min(times), 'loader_max_ms': max(times),
                        'gpu_replay_median_ms': median(r['results']['gpu_replay']['wall_ms_per_batch'] for r in reports)})
    ranking.sort(key=lambda row: row['loader_median_ms'])
    for row in ranking:
        ConsoleProgress.info(json.dumps(row))
    best = ranking[0]['workers']
    ConsoleProgress.info(f'Best measured loader time: AIIJC_WORKERS={best}; .env unchanged')
    return {'recommended_workers': best, 'ranking': ranking, 'runs': runs,
            'note': 'Worker-specific augmentation streams differ; compare repeated wall-time measurements.'}


def main():
    from src.config import load_experiment_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baseline.yaml')
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--workers', type=int, nargs='+', help='Compare worker counts, overriding .env for this benchmark only')
    parser.add_argument('--repeats', type=int, default=2, help='Passes for the worker comparison')
    parser.add_argument('--compare-transfer', action='store_true', help='Compare synchronous and asynchronous CUDA transfers')
    parser.add_argument('--profile-data', action='store_true', help='Measure worker preprocessing stages on consumed batches')
    parser.add_argument('--jpeg-triton-backward', action='store_true', help='Benchmark tiled categorical weight gradients')
    parser.add_argument('--foreach-grad-normalization', action='store_true', help='Benchmark grouped gradient normalization')
    parser.add_argument('--output', default='profiles/baseline.json')
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.jpeg_triton_backward:
        config = replace(config, model=replace(config.model, jpeg_triton_backward=True))
    if args.foreach_grad_normalization:
        config = replace(config, train=replace(config.train, foreach_grad_normalization=True))
    if args.profile_data and (args.compare_transfer or args.workers is not None):
        parser.error('--profile-data runs one configuration; set workers through AIIJC_WORKERS')
    if args.compare_transfer and args.workers is not None:
        parser.error('Use --compare-transfer or --workers separately')
    if args.compare_transfer:
        report = {mode: benchmark(config, batches=args.batches, warmup=args.warmup, asynchronous_transfer=mode == 'asynchronous')
                  for mode in ('synchronous', 'asynchronous')}
        ConsoleProgress.info('Transport comparison: ' + json.dumps({
            mode: result['results']['loader']['wall_ms_per_batch'] for mode, result in report.items()}))
    elif args.workers is not None:
        report = benchmark_workers(config, args.workers, repeats=args.repeats,
                                   batches=args.batches, warmup=args.warmup)
    else:
        report = benchmark(config, batches=args.batches, warmup=args.warmup, profile_data=args.profile_data)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    ConsoleProgress.info(f'Profile saved: {path}')


if __name__ == '__main__':
    main()
