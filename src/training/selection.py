"""A frozen retuned operating point, bound to unchanged weights and development data."""

import hashlib
from pathlib import Path


class RetunedSelection:
    """Verify the selection recorded atomically with summary.best by RunRetuner."""

    @staticmethod
    def digest(path: Path) -> str:
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    @classmethod
    def verify(cls, summary: dict, snapshot: dict, checkpoint: Path, *, check_histograms: bool = False) -> bool:
        record = summary.get('retuned')
        if record is None:
            return False
        if not isinstance(record, dict) or record.get('version') != 1:
            raise ValueError('Retuned selection has no verified provenance; rerun retune on development')
        if record.get('best') != summary.get('best') or record.get('best') != summary.get('val_best'):
            raise ValueError('Retuned operating point differs from the frozen selection')
        n_bins = int(snapshot.get('eval', snapshot).get('n_bins', 256))
        if record.get('n_bins') != n_bins:
            raise ValueError('Retuned histogram grid differs from the run snapshot')
        if not checkpoint.is_file() or cls.digest(checkpoint) != record.get('checkpoint_sha256'):
            raise ValueError('Retuned checkpoint differs from the weights used for selection')
        if record.get('split') != 'val' or record.get('histograms') != 'oof/val.npz':
            raise ValueError('Retuned selection must use development histograms in oof/val.npz')
        if check_histograms:
            histograms = checkpoint.parent.parent / 'oof/val.npz'
            if not histograms.is_file() or cls.digest(histograms) != record.get('histograms_sha256'):
                raise ValueError('Retuned development histograms changed after selection')
        return True
