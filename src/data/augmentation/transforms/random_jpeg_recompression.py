from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from src.data.augmentation.base import AIIJCAugmentation, AugmentationStage, require_rng
from src.data.data_sample import DataSample
from src.forensic.jpeg import luma_qtable
from src.forensic.jpeg_input import JPEGInput


class RandomJPEGRecompression(AIIJCAugmentation):
    """Re-encode with an optional grid shift; both probabilities are per sample.

    The quality range follows NumPy's exclusive upper-bound convention.
    Shifted recompression is a subset of the total recompression probability.
    """

    def __init__(self, quality_range: tuple[int, int], probability: float,
                 *, grid_shift_probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= grid_shift_probability <= probability <= 1.0:
            raise ValueError("JPEG probabilities must satisfy "
                             "0 <= grid_shift_probability <= probability <= 1")
        self.quality_range = quality_range
        self.probability = probability
        self.grid_shift_probability = grid_shift_probability

    @staticmethod
    def _shift_grid(sample: DataSample, rng: np.random.Generator) -> DataSample:
        """Crop to a nonzero 8x8 phase, keeping at least one block per axis."""
        height, width = sample.image.shape[:2]
        max_top, max_left = min(7, max(0, height - 8)), min(7, max(0, width - 8))
        phases = (max_top + 1) * (max_left + 1)
        if phases == 1:
            return sample
        top, left = divmod(int(rng.integers(1, phases)), max_left + 1)
        image = np.ascontiguousarray(sample.image[top:, left:])
        mask = (None if sample.mask is None
                else np.ascontiguousarray(sample.mask[top:, left:]))
        return replace(sample, image=image, mask=mask)

    def jpeg_recompression(
            self,
            image: np.ndarray,
            rng: np.random.Generator | None,
            *, native=False, include_coefficients=False,
    ):
        """JPEG re-encode image and extract the resulting luminance qtable."""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        quality = int(rng.integers(*self.quality_range))
        ok, buffer = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            if native:
                raise ValueError('JPEG recompression failed')
            return image, None

        jpeg_bytes = buffer.tobytes()
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        if native:
            jpeg = JPEGInput.read(jpeg_bytes, include_coefficients=include_coefficients)
            return decoded, jpeg.qtable, jpeg
        return decoded, luma_qtable(jpeg_bytes)

    def apply(self, sample: DataSample, rng: np.random.Generator | None = None) -> DataSample:
        if self.probability <= 0:
            return sample

        rng = require_rng(rng, AugmentationStage.BEFORE_FORENSICS)
        draw = rng.random()
        if draw >= self.probability:
            return sample

        if draw < self.grid_shift_probability:
            sample = self._shift_grid(sample, rng)

        if sample.jpeg is not None:
            image, qtable, jpeg = self.jpeg_recompression(
                sample.image, rng, native=True, include_coefficients=sample.jpeg.coefficients is not None)
            return replace(sample, image=image, qtable=qtable, jpeg=jpeg)
        image, qtable = self.jpeg_recompression(sample.image, rng)
        return replace(sample, image=image, qtable=qtable)
