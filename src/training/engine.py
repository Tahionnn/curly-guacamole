from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable, Sized
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from src.budget import count_gflops
from src.config import ExperimentConfig, SnapshotAdapter
from src.data.data_workspace import DataWorkspace
from src.eval.diagnostics import EvaluationReport
from src.eval.protocol import EvaluationProtocol
from src.losses import LossMeter, SegmentationLoss
from src.progress import ConsoleProgress
from src.training.base import set_random_seed
from src.training.builders import (
    AmpContext,
    build_amp,
    build_datasets,
    build_ema,
    build_loaders,
    build_model,
    build_optimizer,
    build_scheduler,
    configure_memory_format,
)
from src.training.distributed import TrainingRuntime
from src.training.metric import AICResult
from src.training.runs import Run
from src.training.sampling import DistributedBatchSampler
from src.training.transfer import BatchTransfer
from src.training.validation import validate


@dataclass(frozen=True)
class EpochTrainResult:
    loss: float
    skipped_steps: int
    seen: int
    negative_fraction: float = 0.0
    loss_components: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingState:
    start_epoch: int = 0
    seen_total: int = 0
    best_aic: float = -1.0
    best_result: AICResult | None = None
    pending_train_result: EpochTrainResult | None = None
    train_elapsed_s: float = 0.0


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, *, arm: str = "baseline", runtime=None) -> None:
        self.config = config
        self.arm = arm
        self.runtime = runtime
        self.device = runtime.device if runtime is not None else torch.device(config.train.device)
        self.amp = build_amp(config.train, self.device)
        self.data_workspace = DataWorkspace(config.paths.data_path)

    def run(self) -> Run:
        if self.runtime is None:
            if int(os.environ.get('WORLD_SIZE', '1')) > 1:
                raise ValueError('Use python -m src.training with train.devices; '
                                 'external torchrun launch is not supported')
            devices = TrainingRuntime.selected_devices(self.config.train)
            if len(devices) > 1:
                return TrainingRuntime.launch(self.config, self.arm, devices)
            self.device = torch.device(f'cuda:{devices[0]}') if devices else self.device
            if self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
            self.amp = build_amp(self.config.train, self.device)
            self.runtime = TrainingRuntime(self.device)
        return self._run()

    def _run(self) -> Run:
        cfg = self.config
        runtime = self.runtime
        runtime.validate_sync_batchnorm(True)
        self._check_resume_protocol()
        ConsoleProgress.info(f"Эксперимент {cfg.run_name}: device={self.device}, amp={cfg.train.amp}, seed={cfg.seed}")
        set_random_seed(cfg.seed)

        ConsoleProgress.info("Подготовка метаданных и train/val разбиения")
        train_df, val_df = runtime.main_call(self._split_data)
        ConsoleProgress.info(f"Разбиение готово: train={len(train_df)}, val={len(val_df)}; создание датасетов и аугментаций")
        train_ds, val_ds = build_datasets(cfg, self.data_workspace, train_df, val_df)
        if runtime.distributed:
            train_ds.seed = cfg.seed + runtime.rank
        model = self._build_training_model()
        ConsoleProgress.info("Модель готова; подсчёт GFLOPS")
        gflops = runtime.main_call(count_gflops, model, cfg.dataset.image_size,
                             native_size=(1024, 1024))
        ConsoleProgress.info("JPEG GFLOPs measured at native 1024x1024; larger frames cost more")
        ConsoleProgress.info(f"Подсчёт завершён: {gflops:.2f} GFLOPS; создание оптимизатора")
        optimizer = build_optimizer(cfg.train, model)
        ConsoleProgress.info(f"Создание DataLoader: batch_size={cfg.train.batch_size}, workers={cfg.train.workers}")
        train_loader, val_loader = build_loaders(cfg.train, train_ds, val_ds, runtime=runtime)

        ConsoleProgress.info(f"DataLoader готовы: train={len(train_loader)} батчей, val={len(val_loader)}; настройка scheduler, AMP scaler и EMA")
        steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum_steps)
        scheduler_kwargs = {}
        if cfg.train.full_pass_epochs:
            full_batches = (DistributedBatchSampler.batch_count(
                len(train_ds), cfg.train.batch_size, runtime.world_size, False)
                if runtime.distributed else math.ceil(len(train_ds) / cfg.train.batch_size))
            scheduler_kwargs['full_steps_per_epoch'] = math.ceil(full_batches / cfg.train.grad_accum_steps)
        scheduler = build_scheduler(cfg.train, optimizer, steps_per_epoch, **scheduler_kwargs)
        scaler = self.amp.scaler()
        ema = build_ema(cfg.train, model)

        ConsoleProgress.info("Создание папки эксперимента и сохранение конфигурации")
        plain_config = cfg.to_flat_dict()
        plain_config['world_size'] = runtime.world_size
        protocol = EvaluationProtocol.load(cfg.dataset.protocol_path)
        plain_config.update(protocol.provenance())
        run = None

        def prepare_run():
            nonlocal run
            run = Run.create(cfg.paths.runs_path, cfg.run_name, resume=cfg.train.resume)
            train_df.to_parquet(run.dir / 'training_rows.parquet', index=False)
            val_df.to_parquet(run.dir / 'development_rows.parquet', index=False)
            run.save_summary({'evaluation_role': 'development', 'protocol_digest': protocol.digest,
                              'holdout_evaluated': False, 'training_complete': False})
            run.save_snapshot(self._snapshot(plain_config, gflops))
            run.info(f"{cfg.run_name}, {gflops} GFLOPS; GPUs={runtime.world_size}, "
                     f"effective batch={cfg.train.batch_size * cfg.train.grad_accum_steps * runtime.world_size}")
            run.info(f'SyncBN: encoder/decoder/fusion, global forward batch up to '
                     f'{cfg.train.batch_size * runtime.world_size}; JPEG branch keeps per-frame normalization')
            return str(run.dir)

        run_path = runtime.main_call(prepare_run)
        if run is None:
            run = Run.open(run_path)

        if runtime.is_main:
            run.info(f"Результаты: {run.dir.resolve()}; проверка возобновления обучения")
        if runtime.distributed:
            set_random_seed(cfg.seed + runtime.rank)
        state = self._resume_if_needed(
            run=run,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        training_model = runtime.wrap(model)
        runtime.synchronize_buffers(ema.module)
        if runtime.is_main and runtime.cpu_collectives:
            run.info('Gloo: GPU вычисления, обмен градиентами через CPU')

        try:
            for epoch in range(state.start_epoch, cfg.train.epochs):
                train_ds.set_epoch(epoch)
                train_ds.augmentations.set_epoch(epoch)
                if runtime.distributed:
                    train_loader.batch_sampler.set_epoch(epoch, seed=cfg.seed)
                    steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum_steps)
                elif cfg.train.full_pass_epochs:
                    train_loader.sampler.set_epoch(epoch)
                    steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum_steps)
                if not runtime.distributed:
                    train_loader.sampler.generator = torch.Generator().manual_seed(cfg.seed + epoch)
                started = time.time()
                if state.pending_train_result is not None:
                    train_result = state.pending_train_result
                    started -= state.train_elapsed_s
                    if runtime.is_main:
                        run.info(f"Эпоха {epoch + 1}: обучение уже сохранено; повторяю только валидацию")
                else:
                    if runtime.is_main:
                        run.info(f"Полные кадры: p={train_ds.augmentations.full_frame_probability:.2f}; "
                             "валидация: original")
                        run.info(f"Эпоха {epoch + 1}/{cfg.train.epochs}: обучение; "
                                 f"optimizer steps={steps_per_epoch}")
                    train_result = train_one_epoch(
                        model=training_model,
                        loader=train_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        ema=ema,
                        amp=self.amp,
                        config=cfg,
                        device=self.device,
                        runtime=runtime,
                    )
                    state.seen_total += train_result.seen
                    state.pending_train_result = train_result
                    state.train_elapsed_s = time.time() - started
                    # Commit training before any post-training output or validation.
                    self._checkpoint(
                        run, model, ema, optimizer, scheduler, scaler, epoch, state, plain_config,
                        validation_complete=False,
                    )
                if runtime.is_main:
                    run.info(f"Обучение завершено: loss={train_result.loss:.5f}, примеров={train_result.seen}; валидация EMA")
                validation = validate(ema.module, val_loader, self.amp, cfg, self.device, runtime=runtime)
                tuned = validation.tuned

                if tuned.aic > state.best_aic:
                    state.best_aic = tuned.aic
                    state.best_result = tuned
                    runtime.main_call(self._save_best,
                                      run, model, ema, epoch, state, tuned, plain_config, validation, val_df)

                self._checkpoint(
                    run, model, ema, optimizer, scheduler, scaler, epoch, state, plain_config,
                    validation_complete=True,
                )
                state.pending_train_result = None
                runtime.main_call(self._log_epoch,
                    run=run, epoch=epoch, model=model, train_result=train_result,
                    tuned=tuned, seen_total=state.seen_total, started=started,
                    steps_per_epoch=steps_per_epoch, optimizer=optimizer, validation=validation,
                )

            runtime.main_call(self._save_final_summary, run, state, gflops)
            if runtime.is_main:
                run.info(f"Эксперимент завершён: лучший AIC={state.best_aic:.4f}; результаты в {run.dir.resolve()}")
            return run
        finally:
            if runtime.is_main:
                run.close()
            del training_model, model, ema, optimizer, val_loader, train_loader, scaler
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    @staticmethod
    def _save_best(run, model, ema, epoch, state, tuned, plain_config, validation, val_df):
        run.save_state({
            "model": model.state_dict(),
            "ema": ema.module.state_dict(),
            "ema_n_averaged": int(ema.n_averaged),
            "epoch": epoch,
            "samples": state.seen_total,
            "best_aic": state.best_aic,
            "operating_point": tuned.as_dict(),
            "cfg": plain_config,
        }, "best.pt")
        run.info("Сохранение OOF-предсказаний, строк валидации и метрик")
        run.save_eval(validation, val_df)
        report = EvaluationReport(validation.accumulator, val_df, tuned)
        report.save(run.dir / 'development')
        run.save_summary({'development': report.summary()})
        run.info(f"  новый лучший AIC {state.best_aic:.4f} -> ckpt/best.pt")

    def _checkpoint(self, run, model, ema, optimizer, scheduler, scaler, epoch, state,
                    plain_config, *, validation_complete):
        runtime = self.runtime
        runtime.synchronize_buffers(model)
        runtime.synchronize_buffers(ema.module)
        runtime.main_call(self._save_training_checkpoint,
            run, model, ema, optimizer, scheduler, scaler, epoch, state, plain_config,
            validation_complete=validation_complete)

    @staticmethod
    def _save_training_checkpoint(run, model, ema, optimizer, scheduler, scaler,
                                  epoch, state, plain_config, *, validation_complete):
        """Atomic training commit; validation artifacts are a separate phase."""
        run.save_state({
            'model': model.state_dict(),
            'ema': ema.module.state_dict(),
            'ema_n_averaged': int(ema.n_averaged),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'samples': state.seen_total,
            'best_aic': state.best_aic,
            'best_result': state.best_result.as_dict() if state.best_result is not None else None,
            'validation_complete': validation_complete,
            'train_result': asdict(state.pending_train_result),
            'train_elapsed_s': state.train_elapsed_s,
            'cfg': plain_config,
        }, 'last.pt')

    def _build_training_model(self):
        cfg = self.config
        checkpoint = cfg.paths.runs_path / cfg.run_name / 'ckpt' / 'last.pt'
        resume_available = cfg.train.resume and checkpoint.is_file()
        finetune = cfg.train.finetune_from is not None and not resume_available
        initialization = (f'из checkpoint {checkpoint}' if resume_available else
                          f'дотюн из {cfg.train.finetune_from}' if finetune else 'из pretrained-весов')
        ConsoleProgress.info(f'Создание модели {cfg.model.encoder}: инициализация {initialization}, '
                             f'перенос на {self.device}')
        model = build_model(cfg.model, aux_weight=cfg.loss.aux_weight, pretrained=not (resume_available or finetune))
        if finetune:
            self._load_finetune_weights(model)
        return configure_memory_format(model.to(self.device))

    def _load_finetune_weights(self, model) -> None:
        """Start a new optimizer/schedule/EMA from the selected source weights."""
        cfg = self.config
        checkpoint = cfg.paths.runs_path / cfg.train.finetune_from
        if (checkpoint.parent.parent / 'holdout_claim.json').exists():
            raise ValueError('Cannot finetune a run after holdout evaluation was claimed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        EvaluationProtocol.load(cfg.dataset.protocol_path).verify_run(saved['cfg'])
        source = ExperimentConfig.from_dict(SnapshotAdapter.normalize(saved['cfg']))
        if (source.model.encoder != cfg.model.encoder
                or source.model.jpeg_channels != cfg.model.jpeg_channels
                or (source.loss.aux_weight > 0) != (cfg.loss.aux_weight > 0)):
            raise ValueError('Finetune requires matching encoder, JPEG channels and auxiliary head')
        model.load_state_dict(saved[cfg.train.finetune_weights])

    def _check_resume_protocol(self) -> None:
        cfg = self.config
        run_dir = cfg.paths.runs_path / cfg.run_name
        if not cfg.train.resume or not (run_dir / "ckpt" / "last.pt").exists():
            return
        if (run_dir / 'holdout_claim.json').exists():
            raise ValueError('Cannot resume a run after holdout evaluation was claimed; choose a new run_name')
        snapshot = Run.open(run_dir).snapshot
        world_size = self.runtime.world_size if self.runtime is not None else 1
        if snapshot.get('world_size', 1) != world_size:
            raise ValueError('Cannot resume with a different world_size; use finetune_from in a new run')
        previous = ExperimentConfig.from_dict(SnapshotAdapter.normalize(snapshot))
        current = cfg.to_dict()
        saved = previous.to_dict()
        for section in ('model', 'dataset', 'loss', 'augmentation'):
            for key, value in current[section].items():
                if saved[section][key] != value:
                    raise ValueError(f'Cannot resume with a different {section}.{key}; choose a new run_name')
        # Optimizer/scheduler state is meaningful only for the same training recipe.
        runtime_fields = {'device', 'devices', 'distributed_backend', 'workers', 'resume',
                          'finetune_from', 'finetune_weights'}
        for key, value in current['train'].items():
            if key not in runtime_fields and saved['train'][key] != value:
                raise ValueError(f'Cannot resume with a different train.{key}; choose a new run_name')
        if previous.eval != cfg.eval:
            raise ValueError('Cannot resume with different evaluation settings; choose a new run_name')
        if previous.seed != cfg.seed:
            raise ValueError('Cannot resume with a different seed; choose a new run_name')
        EvaluationProtocol.load(cfg.dataset.protocol_path).verify_run(snapshot)

    def _split_data(self):
        cfg = self.config
        protocol = EvaluationProtocol.load(cfg.dataset.protocol_path)
        development = EvaluationReport.add_jpeg_metadata(protocol.rows('development'),
                                                         self.data_workspace.train_root, cfg.train.workers)
        return protocol.rows('train'), development

    def _snapshot(self, plain_config: dict, gflops: float) -> dict:
        result = {**plain_config, 'arm': self.arm, 'gflops': round(gflops, 3)}
        result['gflops_native_size'] = [1024, 1024]
        result['gflops_variable_native_size'] = True
        return result

    def _resume_if_needed(
        self,
        *,
        run: Run,
        model,
        ema,
        optimizer,
        scheduler,
        scaler,
    ) -> TrainingState:
        cfg = self.config
        state = TrainingState()
        if not (cfg.train.resume and (run.dir / "ckpt" / "last.pt").exists()):
            return state

        saved = run.load_state("last.pt", map_location=self.device)
        model.load_state_dict(saved["model"])
        ema.module.load_state_dict(saved["ema"])
        # Older checkpoints saved only the averaged module. Constant-decay EMA
        # needs any positive count to continue averaging instead of overwriting it.
        ema.n_averaged.fill_(saved.get("ema_n_averaged", 1))
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])

        saved_epoch = int(saved["epoch"])
        state.seen_total = int(
            saved.get(
                "samples",
                (saved_epoch + 1) * cfg.train.samples_per_epoch,
            )
        )
        validation_complete = saved.get('validation_complete', True)
        state.start_epoch = saved_epoch + 1 if validation_complete else saved_epoch
        state.best_aic = float(saved.get('best_aic', -1.0))
        best_result = saved.get('best_result')
        if validation_complete:
            state.best_aic = max(state.best_aic, float(run.summary.get('best_aic', -1.0)))
            best_result = best_result or run.summary.get('best')
        else:
            # Summary/OOF may have been partially written by the failed evaluation.
            # Replay against the pre-validation best, including artifact writes.
            state.pending_train_result = EpochTrainResult(**saved['train_result'])
            state.train_elapsed_s = float(saved.get('train_elapsed_s', 0.0))
        if best_result is not None:
            state.best_result = AICResult(**best_result)
        if self.runtime is None or self.runtime.is_main:
            run.info(
                f"ПРОДОЛЖАЮ: эпоха {state.start_epoch} из {cfg.train.epochs}, "
                f"показов {state.seen_total}, лучший AIC {state.best_aic:.4f}"
            )
        return state

    def _log_epoch(
        self,
        *,
        run: Run,
        epoch: int,
        model,
        train_result: EpochTrainResult,
        tuned: AICResult,
        seen_total: int,
        started: float,
        steps_per_epoch: int,
        optimizer,
        validation=None,
    ) -> None:
        extra = {f"train/loss_{key}": value for key, value in train_result.loss_components.items() if key != "total"}
        if validation is not None:
            extra.update({f"val/loss_{key}": value for key, value in validation.loss_components.items()})
            if validation.loss_components:
                extra["val/loss_main"] = sum(
                    validation.loss_components.get(key, 0.0)
                    for key in ("bce", "focal", "dice", "boundary")
                )
            if validation.fixed is not None:
                extra.update({"val/aic_fixed": validation.fixed.aic,
                              "val/dice_fixed": validation.fixed.dice_pos,
                              "val/fpr_fixed": validation.fixed.fpr_neg})
        run.log(
            epoch,
            {
                **extra,
                "samples": seen_total,
                "train/loss": train_result.loss,
                "train/negative_fraction": train_result.negative_fraction,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/skipped_steps": train_result.skipped_steps,
                "train/fmap_gamma": model.forensic_gate_stats()["max_abs"],
                "train/disentangle_gamma": model.disentangle_gate_stats()["max_abs"],
                "val/aic_tuned": tuned.aic,
                "val/dice_tuned": tuned.dice_pos,
                "val/fpr_tuned": tuned.fpr_neg,
                "val/best_thr": tuned.mask_threshold,
                "val/best_cls_thr": tuned.cls_threshold,
                "epoch_time_s": round(time.time() - started, 1),
                "gpu_gb": (
                    round(torch.cuda.max_memory_allocated() / 1e9, 2)
                    if self.device.type == "cuda"
                    else 0.0
                ),
            },
        )
        run.info(f"эпоха {epoch}: {tuned} | пропущено шагов {train_result.skipped_steps}")
        if train_result.skipped_steps > steps_per_epoch * 0.05:
            run.info("  ВНИМАНИЕ: пропущено >5% шагов - переполнения fp16, ставь amp='bf16'")

    def _save_final_summary(self, run: Run, state: TrainingState, gflops: float) -> None:
        run.save_summary(
            {
                "best_aic": state.best_aic,
                "best": (
                    state.best_result.as_dict()
                    if state.best_result is not None
                    else run.summary.get("best")
                ),
                "samples": state.seen_total,
                "gflops": round(gflops, 1),
                "within_limit": None,
                "gflops_native_size": [1024, 1024],
                "reference_within_limit": gflops <= 100,
                "validation_resolution": "original",
                "training_complete": True,
            }
        )


def train_one_epoch(
    *,
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    ema,
    amp: AmpContext,
    config: ExperimentConfig,
    device: torch.device,
    profiler=None,
    asynchronous_transfer=True,
    runtime=None,
) -> EpochTrainResult:
    runtime = runtime or TrainingRuntime(device)
    model.train()
    skipped_steps = 0
    meter = LossMeter()
    criterion = SegmentationLoss(**config.loss.to_dict())
    seen = 0
    accumulation_samples = 0
    negatives = torch.zeros((), device=device)
    total_batches = len(loader) if isinstance(loader, Sized) else None
    optimizer.zero_grad(set_to_none=True)
    transfer = BatchTransfer(device, asynchronous=asynchronous_transfer)

    progress = ConsoleProgress.iterate(
        loader, "Обучение, батчи", image_count=lambda batch: len(batch["image"]), measure_wait=True
    ) if runtime.is_main else loader
    for step, batch in enumerate(progress):
        if profiler is not None:
            profiler.begin(step)
        batch = transfer(batch)
        images = batch["image"].to(memory_format=torch.channels_last)
        if profiler is not None:
            profiler.mark()

        is_accum_boundary = (step + 1) % config.train.grad_accum_steps == 0
        is_last_batch = total_batches is not None and (step + 1) == total_batches
        runtime.before_forward(model)
        sync = (model.no_sync() if runtime.distributed and not runtime.cpu_collectives
                and not (is_accum_boundary or is_last_batch)
                else nullcontext())
        with sync:
            with amp.autocast():
                model_kwargs = {'jpeg': batch['jpeg']} if 'jpeg' in batch else {}
                loss_result = criterion(model(images, **model_kwargs), batch)
            if profiler is not None:
                profiler.mark()
            loss = loss_result.total
            if config.train.full_pass_epochs or runtime.distributed:
                accumulation_samples += len(images)
                scaler.scale(loss * len(images)).backward()
            else:
                scaler.scale(loss / config.train.grad_accum_steps).backward()
        if profiler is not None:
            profiler.mark()
        if is_accum_boundary or is_last_batch:
            runtime.synchronize_gradients()
            if config.train.grad_clip or config.train.full_pass_epochs or runtime.distributed:
                scaler.unscale_(optimizer)
            if config.train.full_pass_epochs or runtime.distributed:
                divisor = runtime.sum(accumulation_samples) / runtime.world_size
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(divisor)
                accumulation_samples = 0
            if config.train.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += int(scaler.get_scale() < before)
            optimizer.zero_grad(set_to_none=True)
        if profiler is not None:
            profiler.mark()
        if is_accum_boundary or is_last_batch:
            ema.update_parameters(runtime.unwrap(model))
            scheduler.step()
        if profiler is not None:
            profiler.mark()

        batch_size = images.size(0)
        negatives += (batch["label"] <= 0.5).sum()
        meter.update(loss_result, batch_size)
        seen += batch_size
        if profiler is not None:
            profiler.mark()
            profiler.end(is_accum_boundary or is_last_batch)

    runtime.reduce_meter(meter)
    seen, negative_count = runtime.sum([seen, float(negatives)]).tolist()
    components = meter.compute()
    return EpochTrainResult(
        loss=components.get("total", 0.0),
        loss_components=components,
        skipped_steps=skipped_steps,
        seen=int(seen),
        negative_fraction=negative_count / max(1, seen),
    )


def run_experiment(config: ExperimentConfig, *, arm: str = "baseline") -> Run:
    return ExperimentRunner(config, arm=arm).run()


def _move_batch_to_device(
    batch: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    return BatchTransfer(device, asynchronous=False)(batch)



__all__ = [
    "EpochTrainResult",
    "ExperimentRunner",
    "TrainingState",
    "run_experiment",
    "train_one_epoch",
    "validate",
]

