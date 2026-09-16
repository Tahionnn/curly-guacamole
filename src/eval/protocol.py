"""Immutable group manifests for development selection and independent holdout."""

import hashlib
import json
from pathlib import Path

import pandas as pd

import global_config
from src.eval.splits import make_stratified_val_folds


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows_digest(rows: pd.DataFrame) -> str:
    columns = ['chng_img_path', 'gt_path', 'group_id', 'target_kind']
    return hashlib.sha256(rows[columns].fillna('').to_json(orient='records').encode()).hexdigest()


class GroupConnections:
    """Union existing groups linked by any shared original path or decoded hash."""

    def __init__(self, metadata: pd.DataFrame, pairs: pd.DataFrame):
        self.parents = {str(g): str(g) for g in metadata.group_id.unique()}
        for table, key in ((metadata, 'orgl_img_path'), (pairs, 'pixel_hash'), (pairs, 'orgl_img_path')):
            if key not in table:
                continue
            for _, rows in table.dropna(subset=[key]).groupby(key, sort=False):
                groups = [str(g) for g in rows.group_id.unique()]
                for group in groups[1:]:
                    a, b = self.find(groups[0]), self.find(group)
                    self.parents[max(a, b)] = min(a, b)

    def find(self, group: str) -> str:
        group = str(group)
        self.parents.setdefault(group, group)
        root = group
        while root != self.parents[root]:
            root = self.parents[root]
        while group != root:
            parent = self.parents[group]
            self.parents[group] = root
            group = parent
        return root


class EvaluationProtocol:
    ROLES = ('train', 'development', 'holdout')

    def __init__(self, path: Path, manifest: dict, samples: pd.DataFrame, originals: pd.DataFrame):
        self.path, self.manifest = path, manifest
        self.samples, self.originals = samples, originals
        self.digest = file_digest(path / 'protocol.json')

    @classmethod
    def create(cls, path, metadata, originals, pairs, *, seed=42):
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f'Protocol already exists: {path}')
        groups = GroupConnections(metadata, pairs)
        samples = metadata.loc[~metadata.broken].copy()
        samples['source_group_id'] = samples.group_id
        samples['group_id'] = samples.group_id.map(groups.find)
        samples['target_kind'] = 'provided'
        samples = make_stratified_val_folds(samples, n_folds=5, seed=seed)
        samples['role'] = samples.fold.map({0: 'development', 1: 'holdout', 2: 'train', 3: 'train', 4: 'train'})
        extra = originals.copy()
        if not extra.empty:
            extra['source_group_id'] = extra.group_id
            extra['group_id'] = extra.group_id.map(groups.find)
            roles = samples.groupby('group_id').role.first()
            extra['role'] = extra.group_id.map(roles)
            extra = extra.loc[extra.role.notna()].reset_index(drop=True)
            if extra.chng_img_path.duplicated().any() or extra.pixel_hash.duplicated().any():
                raise ValueError('duplicate originals')
        cls._check(samples, extra)
        path.mkdir(parents=True)
        samples.to_parquet(path / 'samples.parquet', index=False)
        extra.to_parquet(path / 'originals.parquet', index=False)
        manifest = dict(version=1, seed=seed, role_folds={'train': [2, 3, 4], 'development': [0], 'holdout': [1]},
                        selection='development AIC including verified unique originals; holdout never tunes',
                        hashes={name: file_digest(path / name) for name in ('samples.parquet', 'originals.parquet')},
                        limitations=['Exact original pixel duplicates and supplied groups linked; perceptual duplicates not exhaustively checked.',
                                     'Historical models are not independent on this newly assigned holdout.'],
                        counts=samples.groupby(['role', 'is_negative']).size().reset_index(name='n').to_dict('records'))
        (path / 'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return cls.load(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        path = (path if path.is_absolute() else global_config.PROJECT_ROOT / path).resolve()
        manifest = json.loads((path / 'protocol.json').read_text(encoding='utf-8'))
        if manifest.get('version') != 1:
            raise ValueError('Unsupported protocol version')
        for name in ('samples.parquet', 'originals.parquet'):
            if manifest['hashes'].get(name) != file_digest(path / name):
                raise ValueError(f'Protocol hash mismatch: {name}')
        samples, originals = (pd.read_parquet(path / name) for name in ('samples.parquet', 'originals.parquet'))
        cls._check(samples, originals)
        return cls(path, manifest, samples, originals)

    @classmethod
    def _check(cls, samples, originals):
        all_rows = pd.concat([samples, originals], ignore_index=True)
        if all_rows.groupby('group_id').role.nunique().gt(1).any():
            raise ValueError('group leakage across protocol roles')
        if samples.chng_img_path.duplicated().any():
            raise ValueError('duplicate sample paths')
        if set(samples.role) != set(cls.ROLES):
            raise ValueError('protocol requires all three roles')
        for role in cls.ROLES:
            if samples.loc[samples.role == role, 'is_negative'].nunique() != 2:
                raise ValueError(f'{role} requires both positive and negative provided samples')

    def rows(self, role: str, *, include_originals: bool | None = None) -> pd.DataFrame:
        if role not in self.ROLES:
            raise ValueError(f'Unknown role: {role}')
        include_originals = True if include_originals is None else include_originals
        frames = [self.samples.loc[self.samples.role == role]]
        if include_originals and not self.originals.empty:
            frames.append(self.originals.loc[self.originals.role == role])
        return pd.concat(frames, ignore_index=True)

    def all_training_rows(self) -> pd.DataFrame:
        """All supplied labeled data, including development and holdout originals."""
        return pd.concat([self.rows(role) for role in self.ROLES], ignore_index=True)

    def provenance(self, *, train_all_data=False) -> dict:
        return dict(protocol_digest=self.digest, training_originals=True,
                    training_rows_digest=rows_digest(self.all_training_rows() if train_all_data else self.rows('train')),
                    development_rows_digest=rows_digest(self.rows('development')))

    def verify_run(self, snapshot: dict, *, train_all_data=False) -> None:
        expected = self.provenance(train_all_data=train_all_data)
        for key, value in expected.items():
            if snapshot.get(key) != value:
                raise ValueError(f'Run {key} does not match protocol; historical training cannot claim independent holdout')
