from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.data.geometry import restore_probability
from src.losses import LossMeter, build_loss
from src.progress import ConsoleProgress
from src.training.distributed import TrainingRuntime
from src.training.metric import AICAccumulator, AICResult, threshold_bin
from src.training.sampling import DistributedValidationSampler
from src.training.transfer import BatchTransfer

if TYPE_CHECKING:
    from src.config import ExperimentConfig
    from src.training.builders import AmpContext


@dataclass(frozen=True)
class ValidationResult:
    """Global metrics; in distributed runs only rank zero retains OOF histograms."""

    accumulator: AICAccumulator
    tuned: AICResult
    resolution: str = "original"
    loss_components: dict[str, float] = field(default_factory=dict)
    fixed: AICResult | None = None

    @property
    def operating_point(self) -> tuple[float, float, float, float]:
        return (self.tuned.mask_threshold, self.tuned.cls_threshold, self.tuned.min_area, self.tuned.area_cap)


@torch.no_grad()
def validate(
    model,
    loader: Iterable[dict[str, torch.Tensor]],
    amp: AmpContext,
    config: ExperimentConfig,
    device: torch.device,
    *,
    thresholds=None,
    runtime=None,
) -> ValidationResult:
    runtime = runtime or TrainingRuntime(device)
    if runtime.distributed:
        if not isinstance(getattr(loader, 'sampler', None), DistributedValidationSampler):
            raise ValueError('Distributed validation requires DistributedValidationSampler')
        error = None
        try:
            acc, meter = _collect_validation(model, loader, amp, config, device, progress=runtime.is_main)
        except Exception as exc:
            error = f'rank {runtime.rank}: {type(exc).__name__}: {exc}'
        # All ranks finish their independent forwards before any statistics
        # collective, including ranks with no samples or a failed loader/model.
        errors = runtime.gather_objects(error)
        if any(errors):
            raise RuntimeError('Validation failed: ' + '; '.join(error for error in errors if error))
        runtime.reduce_meter(meter)
        shards = runtime.gather_to_main(ValidationHistograms.pack(acc))
        acc = (ValidationHistograms.merge(shards, config.eval.n_bins, config.eval.selection_small_mask_weight)
               if runtime.is_main else AICAccumulator(n_bins=config.eval.n_bins,
                                                      small_mask_weight=config.eval.selection_small_mask_weight))
    else:
        acc, meter = _collect_validation(model, loader, amp, config, device)
    tuned, fixed = runtime.main_call(_score_validation, acc, config, thresholds)
    return ValidationResult(accumulator=acc, tuned=tuned, resolution='original',
                            loss_components=meter.compute(), fixed=fixed)


class ValidationHistograms:
    """Compact CPU arrays for transfer; rank order equals original dataset order."""

    @staticmethod
    def pack(acc):
        return (np.asarray(acc.hist_all, dtype=np.int64).reshape(len(acc), acc.n_bins),
                np.asarray(acc.hist_gt, dtype=np.int64).reshape(len(acc), acc.n_bins),
                np.asarray(acc.gt_sum, dtype=np.int64), np.asarray(acc.n_pixels, dtype=np.int64),
                np.asarray(acc.cls_prob, dtype=np.float32))

    @staticmethod
    def merge(shards, n_bins, small_mask_weight):
        acc = AICAccumulator(n_bins=n_bins, small_mask_weight=small_mask_weight)
        for shard in shards:
            acc.update_hist(*shard)
        return acc


def _score_validation(acc, config, thresholds):
    if thresholds is None:
        ConsoleProgress.info('Подбор порогов маски, классификации и минимальной площади по AIC')
        tuned = acc.best(config.eval.mask_thresholds, config.eval.cls_thresholds,
                         config.eval.min_areas, config.eval.area_caps)
    else:
        boundary = threshold_bin(thresholds.mask_threshold, acc.n_bins) / acc.n_bins
        if thresholds.mask_threshold >= 1 or not np.isclose(thresholds.mask_threshold, boundary, rtol=0., atol=1e-12):
            raise ValueError('Frozen mask threshold must match an exact histogram boundary')
        tuned = acc.evaluate(thresholds.mask_threshold, thresholds.cls_threshold,
                             thresholds.min_area, thresholds.area_cap)
    ConsoleProgress.info(f'Оценка завершена: {tuned}')
    return tuned, acc.evaluate(.5, .0, .0)


def _collect_validation(model, loader, amp, config, device, *, progress=True):
    model.eval()
    n_bins = config.eval.n_bins
    acc = AICAccumulator(n_bins=n_bins, small_mask_weight=config.eval.selection_small_mask_weight)
    histograms = DeviceHistogramAccumulator(acc, device)
    meter = LossMeter()
    criterion = build_loss(config.loss).eval()

    batches = ConsoleProgress.iterate(loader, 'Валидация, батчи') if progress else loader
    for batch in batches:
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)

        with amp.autocast():
            kwargs = {'jpeg': BatchTransfer.move_jpeg(batch['jpeg'], device)} if 'jpeg' in batch else {}
            out = model(images, **kwargs)

        # Comparable main-head loss on the network grid; AIC can use original GT.
        loss_batch = {key: batch[key].to(device) for key in
                      ("mask", "label") if key in batch}
        meter.update(criterion(out, loss_batch), len(images))
        probs = torch.sigmoid(out["logits"].float())
        cls = torch.sigmoid(out["cls_logits"].float()).reshape(-1)

        if "original_mask" not in batch:
            raise ValueError("original validation requires original_mask from the dataset")
        for index, mask in enumerate(batch["original_mask"]):
            restored = restore_probability(probs[index:index + 1], mask.shape[-2:])
            histograms.update(restored, mask.to(device, non_blocking=True).reshape(1, 1, *mask.shape[-2:]),
                              cls[index:index + 1])

    histograms.flush()
    return acc, meter


class DeviceHistogramAccumulator:
    """Buffer compact per-image statistics on device; copy only at chunk boundaries.

    At 256 bins, 1024 rows use about 4 MiB regardless of original image sizes.
    Integer counts and float32 classification probabilities keep their original
    precision. The CPU accumulator still owns threshold selection and OOF output.
    """

    def __init__(self, accumulator: AICAccumulator, device, capacity: int = 1024):
        if capacity < 1:
            raise ValueError('Histogram capacity must be positive')
        self.accumulator = accumulator
        self.n_bins = accumulator.n_bins
        self.capacity = capacity
        self.count = 0
        self.counts = torch.empty((capacity, 2 * self.n_bins + 2), dtype=torch.int64, device=device)
        self.confidence = torch.empty(capacity, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, probs, masks, cls) -> None:
        batch_size = probs.shape[0]
        n_bins = self.n_bins
        flat = probs.clamp(0, 1).reshape(batch_size, -1)
        idx = (flat * n_bins).long().clamp_(max=n_bins - 1)
        gt = masks.reshape(batch_size, -1) > 0.5
        # Fixed-size integer reductions avoid dynamic boolean selection and
        # bincount output sizing, which can synchronize CUDA with the host.
        hist_all = torch.zeros((batch_size, n_bins), dtype=torch.int64, device=idx.device)
        hist_gt = torch.zeros_like(hist_all)
        hist_all.scatter_add_(1, idx, torch.ones_like(idx))
        hist_gt.scatter_add_(1, idx, gt.to(torch.int64))
        gt_sum = gt.sum(1)
        cls = cls.reshape(-1)
        start = 0
        while start < batch_size:
            take = min(batch_size - start, self.capacity - self.count)
            rows = self.counts[self.count:self.count + take]
            rows[:, :n_bins].copy_(hist_all[start:start + take])
            rows[:, n_bins:2 * n_bins].copy_(hist_gt[start:start + take])
            rows[:, -2].copy_(gt_sum[start:start + take])
            rows[:, -1].fill_(flat.shape[1])
            self.confidence[self.count:self.count + take].copy_(cls[start:start + take])
            self.count += take
            start += take
            if self.count == self.capacity:
                self.flush()

    def flush(self) -> None:
        if not self.count:
            return
        counts = self.counts[:self.count].cpu().numpy()
        confidence = self.confidence[:self.count].cpu().numpy()
        n_bins = self.n_bins
        self.accumulator.update_hist(counts[:, :n_bins], counts[:, n_bins:2 * n_bins],
                                     counts[:, -2], counts[:, -1], confidence)
        self.count = 0



