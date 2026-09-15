from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, WeightedRandomSampler

import global_config
from src.config import ExperimentConfig, ModelConfig, TrainConfig
from src.data.augmentation.pipeline import AugmentationPipeline
from src.data.collation import ValidationCollator
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset
from src.training.sampling import (
    DistributedBatchSampler,
    DistributedValidationSampler,
    FinalFullTrainSampler,
)

if TYPE_CHECKING:
    from src.modules.segmenter import Segmenter


class DataLoaderThreadLimits:
    """Keep CPU libraries single-threaded in the parent and loader workers."""

    @staticmethod
    def apply(worker_id: int | None = None) -> None:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = "1"
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        # Also limit libraries already imported by a notebook or forked worker.
        threadpool_limits(limits=1)


@dataclass(frozen=True)
class AmpContext:
    device: torch.device
    dtype: torch.dtype
    enabled: bool
    scaler_enabled: bool

    def autocast(self) -> Any:
        return torch.amp.autocast(
            self.device.type,
            dtype=self.dtype,
            enabled=self.enabled,
        )

    def scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(
            self.device.type,
            enabled=self.scaler_enabled,
        )


def build_model(config: ModelConfig, *, aux_weight: float = .4, pretrained: bool = True) -> Segmenter:
    from src.modules.segmenter import Segmenter
    from src.modules.sync_batchnorm import SynchronizedBatchNorm

    model = Segmenter(encoder=config.encoder, jpeg_channels=config.jpeg_channels,
                      aux_weight=aux_weight, pretrained=pretrained,
                      jpeg_similarity=config.jpeg_similarity,
                      disentangle_levels=config.disentangle_levels,
                      disentangle_mode=config.disentangle_mode,
                      disentangle_reduction=config.disentangle_reduction,
                      disentangle_cross_strides=config.disentangle_cross_strides,
                      disentangle_attention_width=config.disentangle_attention_width,
                      disentangle_attention_heads=config.disentangle_attention_heads,
                      disentangle_return_to_stride4=config.disentangle_return_to_stride4,
                      bifpn_width=config.bifpn_width, bifpn_repeats=config.bifpn_repeats)
    model.forensic_fusion.branch.artifact.dc_layer0_dil[0].specialize_width = config.jpeg_specialize_width
    model.forensic_fusion.branch.artifact.dc_layer1_tail[0].use_matmul = config.jpeg_pointwise_matmul
    if pretrained and config.jpeg_pretrained is not None:
        path = Path(config.jpeg_pretrained)
        if not path.is_absolute():
            path = global_config.PROJECT_ROOT / path
        model.forensic_fusion.branch.artifact.load_pretrained(path)
    return SynchronizedBatchNorm.apply(model)


def configure_memory_format(model):
    """Use NHWC for fixed RGB shapes, NCHW for the variable-size JPEG stream.

    On RTX 3070, new native shapes in the NHWC JPEG stream incurred seconds
    of cold convolution overhead. NCHW avoids this without resampling inputs.
    Apply before constructing the optimizer or EMA.
    """
    model.to(memory_format=torch.channels_last)
    model.forensic_fusion.branch.to(memory_format=torch.contiguous_format)
    return model


def build_datasets(
    config: ExperimentConfig,
    data_workspace: DataWorkspace,
    train_df,
    val_df,
) -> tuple[AIIJCDataset, AIIJCDataset]:
    augmentations = AugmentationPipeline(config.augmentation, total_epochs=config.train.epochs)
    train_ds = AIIJCDataset(data_workspace, train_df, True, config.dataset.image_size,
                           config.seed, augmentations=augmentations, mode='train')
    val_ds = AIIJCDataset(data_workspace, val_df, False, config.dataset.image_size,
                         config.seed, mode='val', original_targets=True)

    for dataset in (train_ds, val_ds):
        dataset.preprocessor.rgb_uint8_transport = config.dataset.rgb_uint8_transport
    return train_ds, val_ds


def build_sampler(
    config: TrainConfig,
    train_ds: AIIJCDataset,
) -> WeightedRandomSampler:
    neg = train_ds.is_negative
    if neg is None:
        raise ValueError("train dataset must expose is_negative for weighted sampling")

    n_neg = int(neg.sum())
    n_pos = int((~neg).sum())
    if n_neg == 0 or n_pos == 0:
        raise ValueError(f"weighted sampling requires both classes, got pos={n_pos}, neg={n_neg}")

    neg_fr = config.negative_fraction
    weights = np.where(neg, neg_fr / n_neg, (1 - neg_fr) / n_pos)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=config.samples_per_epoch,
        replacement=True,
    )


def build_loaders(
    config: TrainConfig,
    train_ds: AIIJCDataset,
    val_ds: AIIJCDataset,
    *,
    runtime=None,
) -> tuple[DataLoader, DataLoader]:
    DataLoaderThreadLimits.apply()
    pin_memory = torch.device(config.device).type == "cuda"
    # Native JPEG sizes vary; keep worker prefetch memory bounded.
    prefetch = 1
    sampler = build_sampler(config, train_ds)
    if config.full_pass_epochs:
        sampler = FinalFullTrainSampler(sampler, len(train_ds), config.epochs - config.full_pass_epochs)
    batching = dict(batch_size=config.batch_size, sampler=sampler,
                    drop_last=not bool(config.full_pass_epochs))
    if runtime is not None and runtime.distributed:
        batching = dict(batch_sampler=DistributedBatchSampler(
            sampler, config.batch_size, runtime.rank, runtime.world_size,
            drop_last=not bool(config.full_pass_epochs)))
    train_loader = DataLoader(
        train_ds,
        **batching,
        collate_fn=ValidationCollator(),
        num_workers=config.workers,
        pin_memory=pin_memory,
        persistent_workers=config.workers > 0,
        worker_init_fn=DataLoaderThreadLimits.apply,
        prefetch_factor=prefetch if config.workers else None,
    )
    val_batch_size = config.batch_size
    val_sampler = (DistributedValidationSampler(val_ds, val_batch_size, runtime.rank, runtime.world_size)
                   if runtime is not None and runtime.distributed else None)
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=pin_memory,
        persistent_workers=config.workers > 0,
        worker_init_fn=DataLoaderThreadLimits.apply,
        collate_fn=ValidationCollator(),
        prefetch_factor=prefetch if config.workers else None,
    )
    return train_loader, val_loader


def build_optimizer(
    config: TrainConfig,
    model,
) -> torch.optim.AdamW:
    """AdamW with separate LR/weight decay for encoder, forensic branch and decoder."""
    groups = {"enc": [], "enc_nd": [], "jpeg": [], "jpeg_nd": [], "dec": [], "dec_nd": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim <= 1 or name.endswith(("channel_gate", "gamma"))
        if name.startswith(("forensic_fusion.", "branch.", "fuse.")):
            groups["jpeg_nd" if no_decay else "jpeg"].append(parameter)
        elif name.startswith("encoder.") and not name.startswith('encoder.fusions.'):
            groups["enc_nd" if no_decay else "enc"].append(parameter)
        else:
            groups["dec_nd" if no_decay else "dec"].append(parameter)

    param_groups = [
        {"params": groups["enc"], "lr": config.encoder_lr, "weight_decay": config.weight_decay},
        {"params": groups["enc_nd"], "lr": config.encoder_lr, "weight_decay": 0.0},
        {"params": groups["jpeg"], "lr": config.jpeg_lr, "weight_decay": config.weight_decay},
        {"params": groups["jpeg_nd"], "lr": config.jpeg_lr, "weight_decay": 0.0},
        {"params": groups["dec"], "lr": config.head_lr, "weight_decay": config.weight_decay},
        {"params": groups["dec_nd"], "lr": config.head_lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([group for group in param_groups if group["params"]])


def build_scheduler(
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    *,
    full_steps_per_epoch: int | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    steps_per_epoch = max(1, int(steps_per_epoch))
    if config.full_pass_epochs:
        if full_steps_per_epoch is None or full_steps_per_epoch <= 0:
            raise ValueError('full_steps_per_epoch is required for full train finetuning')
        main_steps = steps_per_epoch * (config.epochs - config.full_pass_epochs)
        final_steps = full_steps_per_epoch * config.full_pass_epochs
        warmup = int(config.warmup_fraction * main_steps)

        def phase_lr(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, max(0.0, (step - main_steps) / final_steps))
            return config.min_lr_factor + (1 - config.min_lr_factor) * .5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, phase_lr)
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup = int(config.warmup_fraction * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return config.min_lr_factor + (1 - config.min_lr_factor) * 0.5 * (
            1 + math.cos(math.pi * min(progress, 1.0))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_amp(
    config: TrainConfig,
    device: torch.device | None = None,
) -> AmpContext:
    device = torch.device(config.device) if device is None else device
    return AmpContext(
        device=device,
        dtype=torch.bfloat16 if config.amp == "bf16" else torch.float16,
        enabled=device.type == "cuda" and config.amp != "off",
        scaler_enabled=device.type == "cuda" and config.amp == "fp16",
    )


def build_ema(
    config: TrainConfig,
    model,
) -> torch.optim.swa_utils.AveragedModel:
    return torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema_decay),
    )


__all__ = [
    "AmpContext",
    "build_amp",
    "build_datasets",
    "build_ema",
    "build_loaders",
    "build_model",
    "build_optimizer",
    "build_sampler",
    "build_scheduler",
]
