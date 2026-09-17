import numpy as np  # noqa: F401 -- Initialize NumPy before torch on Windows.
import pytest
import torch

from src.training.profiling import BenchmarkBatches, StageProfiler


def test_benchmark_batches_are_bounded_and_replay_does_not_copy():
    assert list(BenchmarkBatches(range(10), 3)) == [0, 1, 2]
    batch = {'image': torch.zeros(1)}
    repeated = list(BenchmarkBatches(batch, 4, repeat=True))
    assert len(repeated) == 4 and all(item is batch for item in repeated)


def test_profiler_warmup_and_per_update_denominators(monkeypatch):
    class Event:
        counter = 0

        def __init__(self, **kwargs):
            self.index = Event.counter
            Event.counter += 1

        def record(self, stream):
            pass

        def elapsed_time(self, other):
            return (other.index - self.index) * 2.

    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    profiler = StageProfiler('cuda', warmup=1)
    for step in range(3):
        profiler.begin(step)
        for _ in profiler.STAGES:
            profiler.mark()
        profiler.end(step == 2)
    report = profiler.report()
    assert report['batches'] == 2
    assert report['optimizer_updates'] == 1
    assert report['cuda_ms_per_batch']['backward'] == 2.
    assert report['cuda_ms_per_update']['ema'] == 2.


def test_benchmark_rejects_cpu_before_loading_data():
    from dataclasses import replace

    from src.config import load_experiment_config
    from src.training.profiling import benchmark
    cfg = load_experiment_config('configs/baseline.yaml')
    cfg = replace(cfg, train=replace(cfg.train, device='cpu'))
    with pytest.raises(ValueError, match='CUDA'):
        benchmark(cfg)


def test_worker_sweep_preserves_hardware_settings_and_ranks_loader_time(monkeypatch):
    from src.config import load_experiment_config
    from src.training import profiling

    config = load_experiment_config('configs/baseline.yaml')
    calls = []

    def fake_benchmark(cfg, **kwargs):
        calls.append(cfg.train.workers)
        assert cfg.train.batch_size == config.train.batch_size
        assert cfg.train.grad_accum_steps == config.train.grad_accum_steps
        assert cfg.train.amp == config.train.amp
        return {'results': {'loader': {'wall_ms_per_batch': {2: 400, 4: 300, 10: 500}[cfg.train.workers]},
                            'gpu_replay': {'wall_ms_per_batch': 200}}}

    monkeypatch.setattr(profiling, 'benchmark', fake_benchmark)
    report = profiling.benchmark_workers(config, [2, 4, 10], repeats=2)
    assert calls == [2, 4, 10, 10, 4, 2]
    assert report['recommended_workers'] == 4
    assert report['ranking'][0]['loader_median_ms'] == 300


@pytest.mark.parametrize('workers,repeats', [([], 2), ([2, 2], 2), ([-1], 2), ([2], 0)])
def test_worker_sweep_rejects_invalid_arguments(workers, repeats):
    from src.training.profiling import benchmark_workers
    with pytest.raises(ValueError):
        benchmark_workers(None, workers, repeats=repeats)


def test_compare_transfer_cli_runs_all_variants_and_saves_report(tmp_path, monkeypatch):
    import json

    from src.training import profiling

    called = []
    def fake_benchmark(config, *, asynchronous_transfer, batches, warmup):
        called.append((asynchronous_transfer, batches, warmup))
        return {'results': {'loader': {'wall_ms_per_batch': 300}}}

    monkeypatch.setattr(profiling, 'benchmark', fake_benchmark)
    output = tmp_path / 'transfer.json'
    monkeypatch.setattr('sys.argv', ['profiling', '--compare-transfer', '--batches', '64', '--output', str(output)])
    profiling.main()
    assert called == [(mode, 64, 8) for mode in (False, True)]
    assert list(json.loads(output.read_text())) == ['synchronous', 'asynchronous']


def test_reference_cli_benchmarks_original_kernels_without_changing_recipe(tmp_path, monkeypatch):
    from src.config import load_experiment_config
    from src.training import profiling

    path = 'configs/experiments/disentangle_b2_li760_r8_all_data_hard_pixel_ft.yaml'
    original = load_experiment_config(path)
    called = []

    def benchmark(config, **kwargs):
        called.append(config)
        return {}

    monkeypatch.setattr(profiling, 'benchmark', benchmark)
    monkeypatch.setattr('sys.argv', ['profiling', '--config', path, '--reference-kernels',
                                    '--output', str(tmp_path / 'reference.json')])
    profiling.main()
    assert len(called) == 1
    reference = called[0]
    assert not reference.model.jpeg_triton_backward
    assert reference.model.jpeg_specialize_width
    assert not reference.train.foreach_grad_normalization
    assert reference.loss == original.loss
    assert reference.train.batch_size == original.train.batch_size
    assert reference.train.grad_accum_steps == original.train.grad_accum_steps
