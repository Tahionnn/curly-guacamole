"""Checkpoint inference and binary masks at the original image resolution."""

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isclose

import numpy as np
import torch

from src.data.geometry import restore_probability
from src.training.builders import AmpContext
from src.training.metric import operating_bins, threshold_bin
from src.training.transfer import BatchTransfer


@dataclass(frozen=True)
class ThresholdConfig:
    mask_threshold: float = 0.5
    cls_threshold: float = 0.0
    min_area: float = 0.0
    # 0 keeps the historical gate, which zeroes a doubted frame outright. Above 0
    # the gate instead raises that frame's threshold until its area drops below
    # the cap, so a wrongly gated positive keeps a partial mask.
    area_cap: float = 0.0
    # Thresholds are resolved on the validation histogram grid; inference must
    # use the same one or a tuned operating point does not reproduce here.
    n_bins: int = 256

    def __post_init__(self) -> None:
        for name in ("mask_threshold", "cls_threshold", "min_area", "area_cap"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if type(self.n_bins) is not int or self.n_bins <= 0:
            raise ValueError("n_bins must be a positive integer")
        if not isclose(self.mask_threshold * self.n_bins, round(self.mask_threshold * self.n_bins),
                       rel_tol=0., abs_tol=1e-10):
            raise ValueError(
                f"mask_threshold {self.mask_threshold} is not a boundary of the {self.n_bins}-bin "
                "histogram grid; validation could not have selected it")

    @property
    def bin_index(self) -> int:
        return threshold_bin(self.mask_threshold, self.n_bins)

    @classmethod
    def from_summary(cls, summary: dict, snapshot: dict) -> 'ThresholdConfig':
        best = summary.get('best') or {}
        keys = ('mask_threshold', 'cls_threshold', 'min_area')
        if not all(key in best for key in keys):
            raise ValueError('run summary has no operating point; pass thresholds explicitly')
        return cls(**{key: float(best[key]) for key in keys},
                   area_cap=float(best.get('area_cap', 0.0)),
                   n_bins=int(snapshot.get('eval', snapshot).get('n_bins', 256)))


@dataclass(frozen=True)
class Prediction:
    image_path: str
    mask: np.ndarray


class Predictor:
    def __init__(self, model, thresholds: ThresholdConfig, amp: AmpContext) -> None:
        self.model = model.to(amp.device)
        self.thresholds = thresholds
        self.amp = amp

    def predict(self, loader: Iterable, pool=None, queue_depth: int = 64) -> Iterator[Prediction]:
        """Resize probabilities before thresholding; gate on the restored mask area.

        With a `pool`, the per-frame numpy work (histogram, threshold, packing to
        uint8) is handed to worker threads while the main thread goes straight
        back to the next forward pass, so the GPU stops waiting on it. Results
        are drained in submission order, so the writer still sees every frame
        exactly once and in a deterministic sequence.

        The GPU half stays on this thread on purpose: `probabilities` is an
        inference tensor, and touching one outside the InferenceMode region that
        made it raises. Only the numpy array that comes back is shared.
        """
        if type(queue_depth) is not int or queue_depth <= 0:
            raise ValueError('queue_depth must be a positive integer')
        self.model.eval()
        pending: deque = deque()
        for batch in loader:
            # Native JPEG shapes vary: autotuning would search again for new sizes.
            # Restore caller flags and leave inference/autocast before yielding.
            with torch.inference_mode(), torch.backends.cudnn.flags(benchmark=False), self.amp.autocast():
                kwargs = {'jpeg': BatchTransfer.move_jpeg(batch['jpeg'], self.amp.device)} if 'jpeg' in batch else {}
                # Same transfer as src.training.validation: the thresholds were
                # tuned on the kernels channels_last selects.
                images = batch['image'].to(self.amp.device, non_blocking=True,
                                           memory_format=torch.channels_last)
                output = self.model(images, **kwargs)
                probabilities = output["logits"].float().sigmoid()
                cls_probs = output["cls_logits"].float().sigmoid().flatten()
                restored = [
                    (image_path, float(cls_probs[index]),
                     self.restore(probabilities[index:index + 1],
                                  tuple(int(value) for value in batch["original_size"][index])))
                    for index, image_path in enumerate(batch["image_path"])
                ]
            for image_path, cls_probability, frame in restored:
                if pool is None:
                    yield Prediction(image_path, self.mask_from_probabilities(frame, cls_probability))
                else:
                    pending.append((image_path,
                                    pool.submit(self.mask_from_probabilities, frame, cls_probability)))
                    # Drain inside the batch so even a large batch obeys the bound.
                    if len(pending) >= queue_depth:
                        pending_path, future = pending.popleft()
                        yield Prediction(pending_path, future.result())
        while pending:
            image_path, future = pending.popleft()
            yield Prediction(image_path, future.result())

    def restore(self, probability: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
        """Upsample on device, then hand back a host array the worker threads may use."""
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("original size must contain positive height and width")
        return restore_probability(probability, size)[0, 0].cpu().numpy()

    def mask_from_probabilities(self, probabilities: np.ndarray, cls_probability: float) -> np.ndarray:
        bins, blank = self.operating_point(probabilities, cls_probability)
        if blank:
            mask = np.zeros(probabilities.shape, dtype=bool)
        else:
            mask = self._mask_at_bin(probabilities, bins)
        return mask.astype(np.uint8) * 255

    def _mask_at_bin(self, probabilities: np.ndarray, bins: int) -> np.ndarray:
        # Match histogram quantization, including non-power-of-two rounding.
        n_bins = self.thresholds.n_bins
        return (probabilities * n_bins >= bins if n_bins & (n_bins - 1)
                else probabilities >= bins / n_bins)

    def binary_mask(
        self, probability: torch.Tensor, cls_probability: float, size: tuple[int, int],
    ) -> np.ndarray:
        return self.mask_from_probabilities(self.restore(probability, size), cls_probability)

    def operating_point(self, probabilities: np.ndarray, cls_probability: float) -> tuple[int, bool]:
        """Per-frame threshold bin and blanking flag, from the validation rule itself.

        The sweep that chose these thresholds calls the same `operating_bins`, on
        histograms of the same restored probabilities. Reimplementing the rule here
        would let inference and selection drift apart silently, and the reported
        score would stop describing the submission.
        """
        n_bins, start = self.thresholds.n_bins, self.thresholds.bin_index
        size = probabilities.size
        gated = cls_probability < self.thresholds.cls_threshold
        if gated and self.thresholds.area_cap == 0:
            return start, True
        if not gated:
            if self.thresholds.min_area == 0:
                return start, False
            area = np.count_nonzero(self._mask_at_bin(probabilities, start)) / max(size, 1)
            return start, area < self.thresholds.min_area
        # Only bins at or above bin_index can change the answer: operating_bins
        # takes max(bin_index, cap_index), so a cap satisfied lower down still
        # resolves to bin_index. Histogramming the suprathreshold pixels alone
        # turns a full-frame pass into one over the predicted region, which on a
        # negative frame is a rounding error. Bins below start are filled with
        # the frame size, an area of 1.0, so cap_bins never selects them.
        pred_counts = np.full((1, n_bins), float(size))
        above = probabilities.reshape(-1)
        if n_bins & (n_bins - 1):
            scaled = above * n_bins
            above = scaled[scaled >= start]
        else:
            above = above[above >= start / n_bins] * n_bins
        if above.size:
            index = np.clip(above.astype(np.int32), start, n_bins - 1)
            histogram = np.bincount(index - start, minlength=n_bins - start)
            pred_counts[0, start:] = np.cumsum(histogram[::-1])[::-1]
        else:
            pred_counts[0, start:] = 0.0
        bins, blank = operating_bins(
            pred_counts,
            np.array([probabilities.size], dtype=np.float64),
            np.array([cls_probability], dtype=np.float64),
            self.thresholds.bin_index,
            self.thresholds.cls_threshold,
            self.thresholds.min_area,
            self.thresholds.area_cap,
        )
        return int(bins[0]), bool(blank[0])
