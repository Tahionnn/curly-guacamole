"""Retune development histograms without a model or GPU; read-only unless --write.

Example: python -m src.tools.retune --run runs/disentangle_fuse_cross32 --small-mask-weight 1
"""

import argparse
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import yaml

from src.training.metric import DEFAULT_CAP_GRID, DEFAULT_MASK_GRID, AICAccumulator, AICResult
from src.training.selection import RetunedSelection

CLS_GRID = (0., .5, .6, .65, .7, .74, .78, .8, .82, .84, .86, .9, .95)


class RunRetuner:
    """Select and freeze postprocessing using the best checkpoint's saved validation."""

    def __init__(self, run_dir: str | Path, *, small_mask_weight: float | None = None):
        self.run_dir = Path(run_dir)
        self.histogram_path = self.run_dir / 'oof/val.npz'
        self.checkpoint_path = self.run_dir / 'ckpt/best.pt'
        self.summary_path = self.run_dir / 'summary.json'
        self.snapshot_path = self.run_dir / 'config.yaml'
        self.summary = json.loads(self.summary_path.read_text(encoding='utf-8')) if self.summary_path.exists() else {}
        self.snapshot = yaml.safe_load(self.snapshot_path.read_text(encoding='utf-8')) if self.snapshot_path.exists() else {}
        self.histogram_digest = RetunedSelection.digest(self.histogram_path)
        self.checkpoint_digest = RetunedSelection.digest(self.checkpoint_path) if self.checkpoint_path.exists() else None
        self.acc = AICAccumulator.load(self.histogram_path)
        if small_mask_weight is not None:
            self.acc = replace(self.acc, small_mask_weight=small_mask_weight)
        self.grid = None
        self.ranked = []

    def sweep(self, masks=DEFAULT_MASK_GRID, classifiers=CLS_GRID, min_areas=(0.,), caps=DEFAULT_CAP_GRID):
        self.grid = dict(mask=list(masks), cls=list(classifiers), min_area=list(min_areas), area_cap=list(caps))
        for name, values in self.grid.items():
            if not values or any(not 0. <= value <= 1. for value in values):
                raise ValueError(f'{name} grid must contain finite probabilities in [0, 1]')
        self.ranked = self.acc.sweep(self.grid['mask'], self.grid['cls'], self.grid['min_area'], self.grid['area_cap'])
        return self.ranked

    def current(self, *, stored_weight: bool = False) -> AICResult | None:
        point = self.summary.get('best')
        if not point:
            return None
        acc = self.acc
        if stored_weight:
            acc = replace(acc, small_mask_weight=point.get('small_mask_weight', 1.0))
        return acc.evaluate(point['mask_threshold'], point['cls_threshold'], point['min_area'], point.get('area_cap', 0.))

    def save(self, best: AICResult) -> None:
        """One atomic summary write; checkpoint weights and historical reports stay intact."""
        if self.grid is None or best not in self.ranked:
            raise ValueError('Select an operating point with sweep before saving')
        current_summary = json.loads(self.summary_path.read_text(encoding='utf-8'))
        if current_summary != self.summary:
            raise ValueError('Run summary changed during retuning; reload the run')
        if not current_summary.get('training_complete'):
            raise ValueError('Complete development training before writing a retuned selection')
        if (current_summary.get('holdout_evaluated') or (self.run_dir / 'holdout_claim.json').exists()
                or (self.run_dir / 'holdout').exists()):
            raise ValueError('Cannot retune a run after holdout evaluation has started')
        if not self.snapshot_path.exists() or not self.checkpoint_path.is_file():
            raise ValueError('Writing a retuned selection requires config.yaml and ckpt/best.pt')
        snapshot = yaml.safe_load(self.snapshot_path.read_text(encoding='utf-8'))
        if snapshot != self.snapshot:
            raise ValueError('Run snapshot changed during retuning; reload the run')
        if int(snapshot.get('eval', snapshot).get('n_bins', 256)) != self.acc.n_bins:
            raise ValueError('Retuned histogram grid differs from the run snapshot')
        if (RetunedSelection.digest(self.histogram_path) != self.histogram_digest
                or RetunedSelection.digest(self.checkpoint_path) != self.checkpoint_digest):
            raise ValueError('Retuning inputs changed during selection; reload the run')
        RetunedSelection.verify(current_summary, snapshot, self.checkpoint_path, check_histograms=True)
        current = self.current(stored_weight=True)
        previous = current_summary.get('best')
        if current is not None and 'aic' in previous and abs(current.aic - previous['aic']) > 1e-8:
            raise ValueError('Saved summary score does not match development histograms; verify the run first')
        point = best.as_dict()
        summary = dict(current_summary, best=point, val_best=point, best_aic=best.aic,
                       retuned=dict(version=1, split='val', histograms='oof/val.npz',
                                    histograms_sha256=self.histogram_digest, checkpoint_sha256=self.checkpoint_digest,
                                    n_bins=self.acc.n_bins, best=point, previous=previous, grid=self.grid))
        # Replace one file, so interruption cannot leave thresholds and provenance out of sync.
        with tempfile.NamedTemporaryFile(mode='w', dir=self.run_dir, prefix='summary.', suffix='.tmp',
                                         encoding='utf-8', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, self.summary_path)
        finally:
            temporary.unlink(missing_ok=True)
        self.summary = summary


def _grid(value):
    return tuple(float(item) for item in value.replace(',', ' ').split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--split', choices=('val',), default='val', help='only development may select thresholds')
    parser.add_argument('--mask-thresholds', type=_grid, default=DEFAULT_MASK_GRID)
    parser.add_argument('--cls-thresholds', type=_grid, default=CLS_GRID)
    parser.add_argument('--min-areas', type=_grid, default=(0.,))
    parser.add_argument('--area-caps', type=_grid, default=DEFAULT_CAP_GRID)
    parser.add_argument('--small-mask-weight', type=float, help='1 selects on the official competition metric')
    parser.add_argument('--top', type=int, default=8)
    parser.add_argument('--write', action='store_true', help='freeze selection in summary; keep weights unchanged')
    args = parser.parse_args()
    try:
        retuner = RunRetuner(args.run, small_mask_weight=args.small_mask_weight)
        ranked = retuner.sweep(args.mask_thresholds, args.cls_thresholds, args.min_areas, args.area_caps)
        acc, best = retuner.acc, ranked[0]
        uncapped = acc.best(args.mask_thresholds, args.cls_thresholds, args.min_areas, [0.])
        current = retuner.current()
        print(f'histograms: {retuner.histogram_path}')
        print(f'frames={len(acc)} bins={acc.n_bins} small_mask_weight={acc.small_mask_weight:g}')
        for result in ranked[:max(1, args.top)]:
            print(f'  {result}')
        if current is not None:
            print(f'\nsummary point at this weight: {current}')
            print(f'total over summary: {best.aic - current.aic:+.6f}')
        print(f'no cap:   AIC={uncapped.aic:.6f} cls={uncapped.cls_threshold:g}')
        print(f'with cap: AIC={best.aic:.6f} cls={best.cls_threshold:g} cap={best.area_cap:g}')
        print(f'cap contributes: {best.aic - uncapped.aic:+.6f}')
        if args.write:
            retuner.save(best)
            print(f'Wrote {retuner.summary_path}; inference and holdout use this frozen selection.')
        else:
            print('\nNothing written. Use explicit thresholds, or rerun with --write before holdout:')
            print(f'python -m src.inference "{args.run}" <output_dir> '
                  f'--mask-threshold {best.mask_threshold} --cls-threshold {best.cls_threshold} '
                  f'--min-area {best.min_area} --area-cap {best.area_cap} --n-bins {acc.n_bins}')
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
