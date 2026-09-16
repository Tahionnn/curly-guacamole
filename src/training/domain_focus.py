"""Training-only domain selection; preserve negatives for false-positive control."""

from src.eval.diagnostics import EvaluationReport


class PlainNonunitFocus:
    @staticmethod
    def select(rows, root, workers=4):
        if not rows.role.eq('train').all():
            raise ValueError('Domain focus accepts train rows only')
        candidates = rows.loc[rows.is_negative | rows.domain.eq('plain')].copy()
        candidates = EvaluationReport.add_jpeg_metadata(candidates, root, workers)
        selected = candidates.loc[
            candidates.is_negative | candidates.q_kind.eq('nonunit')].copy()
        if selected.is_negative.all() or not selected.is_negative.any():
            raise ValueError('Domain focus requires positive plain/nonunit and negative rows')
        return selected.reset_index(drop=True)
