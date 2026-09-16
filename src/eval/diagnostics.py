"""Per-image and cohort metrics at a frozen operating point; never tune here."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.training.metric import harmonic_aic, operating_bins, threshold_bin


class EvaluationReport:
    @staticmethod
    def add_jpeg_metadata(rows, root, workers=4):
        rows = rows.copy()

        def probe(path):
            with Image.open(Path(root) / path) as image:
                q = getattr(image, 'quantization', {}).get(0)
            return ('unit' if min(q) == max(q) == 1 else 'nonunit') if q else 'unknown'

        missing = rows.q_kind.isna() if 'q_kind' in rows else pd.Series(True, index=rows.index)
        if 'q_kind' not in rows:
            rows['q_kind'] = 'unknown'
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            rows.loc[missing, 'q_kind'] = list(pool.map(probe, rows.loc[missing, 'chng_img_path']))
        return rows

    def __init__(self, accumulator, rows, thresholds):
        if len(accumulator) != len(rows):
            raise ValueError('Metric rows do not align with accumulator')
        boundary = threshold_bin(thresholds.mask_threshold, accumulator.n_bins) / accumulator.n_bins
        if thresholds.mask_threshold >= 1 or not np.isclose(thresholds.mask_threshold, boundary, rtol=0., atol=1e-12):
            raise ValueError('Frozen threshold must match histogram boundary')
        self.accumulator, self.rows, self.thresholds = accumulator, rows.reset_index(drop=True), thresholds

    def per_image(self):
        p, i, g, n, c = self.accumulator.tables()
        k = threshold_bin(self.thresholds.mask_threshold, self.accumulator.n_bins)
        bins, blank = operating_bins(p, n, c, k, self.thresholds.cls_threshold,
                                     self.thresholds.min_area, self.thresholds.area_cap)
        keep_area = p[:, k] / n >= self.thresholds.min_area
        pred_no_gate, inter_no_gate = p[:, k] * keep_area, i[:, k] * keep_area
        indices = np.arange(len(g))
        pred = np.where(blank, 0.0, p[indices, bins])
        inter = np.where(blank, 0.0, i[indices, bins])
        area = pred / n
        frame = self.rows.copy()
        if 'target_kind' not in frame:
            frame['target_kind'] = 'provided'
        if 'q_kind' not in frame:
            frame['q_kind'] = 'unknown'
        frame['is_positive'] = g > 0
        frame['gt_fraction'] = g / n
        frame['cls_probability'] = c
        frame['pred_fraction'] = area
        frame['dice'] = np.where(g > 0, 2 * inter / (pred + g + 1e-6), np.nan)
        frame['false_positive'] = (g == 0) & (area >= .01)
        frame['dice_no_gate'] = np.where(g > 0, 2 * inter_no_gate / (pred_no_gate + g + 1e-6), np.nan)
        frame['false_positive_no_gate'] = (g == 0) & (pred_no_gate / n >= .01)
        frame['area_bin'] = pd.cut(frame.gt_fraction, [-1, 0, .01, .05, .15, 1.],
                                    labels=['negative', '(0,1%]', '(1,5%]', '(5,15%]', '(15,100%]'])
        return frame

    @staticmethod
    def aggregate(frame):
        pos = frame.is_positive
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        dice = float(frame.loc[pos, 'dice'].mean()) if n_pos else None
        fp = int(frame.loc[~pos, 'false_positive'].sum())
        fpr = fp / n_neg if n_neg else None
        return dict(n_pos=n_pos, n_neg=n_neg, dice_pos=dice, false_positives=fp, fpr_neg=fpr,
                    aic=harmonic_aic(dice, fpr) if n_pos and n_neg else None,
                    dice_no_gate=float(frame.loc[pos, 'dice_no_gate'].mean()) if n_pos else None,
                    fpr_no_gate=float(frame.loc[~pos, 'false_positive_no_gate'].mean()) if n_neg else None)

    def summary(self):
        frame = self.per_image()
        summary = {name: self.aggregate(subset) for name, subset in (
            ('combined', frame), ('provided', frame.loc[frame.target_kind != 'original_zero']),
            ('originals', frame.loc[frame.target_kind == 'original_zero']))}
        if self.accumulator.small_mask_weight != 1.0 and frame.is_positive.any() and (~frame.is_positive).any():
            result = self.accumulator.evaluate(self.thresholds.mask_threshold, self.thresholds.cls_threshold,
                                               self.thresholds.min_area, self.thresholds.area_cap)
            summary['selection'] = result.as_dict()
        return summary

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        frame = self.per_image()
        frame.to_parquet(directory / 'per_image.parquet', index=False)
        (directory / 'metrics.json').write_text(json.dumps(self.summary(), indent=2, allow_nan=False), encoding='utf-8')
        slices = []
        for column in ('domain', 'q_kind', 'area_bin'):
            for (kind, value), subset in frame.groupby(['target_kind', column], observed=True, dropna=False):
                slices.append(dict(target_kind=kind, slice=column, value=str(value), **self.aggregate(subset)))
        pd.DataFrame(slices).to_csv(directory / 'slices.csv', index=False)
