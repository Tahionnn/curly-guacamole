"""Exercise the actual epoch loop, logs, OOF output and checkpoint serialization."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_experiment_config
from src.training.engine import ExperimentRunner


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.augmentations = SimpleNamespace(full_frame_probability=1., set_epoch=lambda epoch: None)
        self.is_negative = np.array([False, True])

    def __len__(self):
        return 2

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        mask = torch.zeros(1, 8, 8)
        if index == 0:
            mask[:, 2:6, 2:6] = 1
        return {'image': mask.expand(3, -1, -1).clone(), 'mask': mask,
                'label': torch.tensor([float(index == 0)]), 'original_mask': mask[0].bool()}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 1, 1)

    def forward(self, image, fmap=None, **kwargs):
        logits = self.head(image)
        return {'logits': logits, 'cls_logits': logits.mean((2, 3))}

    def forensic_gate_stats(self):
        return {'max_abs': 0.}

    def disentangle_gate_stats(self):
        return {'max_abs': 0.}


def configure_tiny_run(tmp_path, monkeypatch):
    import src.training.engine as engine

    cfg = load_experiment_config('configs/baseline_long.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path),
                  augmentation=replace(cfg.augmentation, final_full_frame_epochs=1),
                  train=replace(cfg.train, device='cpu', amp='off', workers=0, batch_size=2,
                                grad_accum_steps=1, epochs=1, full_pass_epochs=1, resume=False, finetune_from=None))
    rows = pd.DataFrame({'chng_img_path': ['positive.jpg', 'negative.jpg'],
                         'stem': ['positive', 'negative'],
                         'domain': ['test', 'test'], 'target_kind': ['provided', 'provided']})
    monkeypatch.setattr(ExperimentRunner, '_split_data', lambda self: (rows, rows))
    monkeypatch.setattr(ExperimentRunner, '_build_training_model', lambda self: TinyModel())
    monkeypatch.setattr(engine, 'build_datasets', lambda *args: (TinyDataset(), TinyDataset()))
    monkeypatch.setattr(engine.TrainingRuntime, 'validate_sync_batchnorm', lambda self, enabled: None)
    monkeypatch.setattr(engine, 'count_gflops', lambda *args, **kwargs: 0.)
    protocol = SimpleNamespace(digest='synthetic-test', provenance=lambda: {'protocol_digest': 'synthetic-test'},
                               verify_run=lambda snapshot: None)
    monkeypatch.setattr(engine.EvaluationProtocol, 'load', lambda path: protocol)
    return cfg


def test_baseline_epoch_saves_best_last_and_logs(tmp_path, monkeypatch):
    cfg = configure_tiny_run(tmp_path, monkeypatch)

    run = ExperimentRunner(cfg).run()

    for filename in ('best.pt', 'last.pt'):
        saved = torch.load(run.dir / 'ckpt' / filename, weights_only=True)
        assert saved['epoch'] == 0 and saved['samples'] == 2
        assert saved['cfg']['pipeline_version'] == 'jpeg640_v1'
    summary = json.loads((run.dir / 'summary.json').read_text())
    assert summary['training_complete'] is True
    assert (run.dir / 'development/per_image.parquet').exists()
    row = json.loads(run.jsonl_path.read_text().splitlines()[0])
    assert 'val/loss_bce' in row and 'val/loss_focal' not in row
    np.testing.assert_allclose(row['val/loss_main'],
                               row['val/loss_bce'] + row['val/loss_dice'])


@pytest.mark.parametrize('failure', ['validation', 'logging', 'report'])
@pytest.mark.parametrize('epochs', [1, 2])
def test_completed_training_survives_failure_and_resume(tmp_path, monkeypatch, failure, epochs):
    import src.training.engine as engine

    cfg = configure_tiny_run(tmp_path, monkeypatch)
    cfg = replace(cfg, train=replace(cfg.train, resume=True, epochs=epochs, full_pass_epochs=epochs),
                  augmentation=replace(cfg.augmentation, final_full_frame_epochs=epochs))
    train_original = engine.train_one_epoch
    validate_original = engine.validate
    log_original = ExperimentRunner._log_epoch
    report_original = engine.EvaluationReport.save
    calls = {'train': 0, 'validation': 0}
    events = []

    def train(**kwargs):
        events.append('train')
        calls['train'] += 1
        return train_original(**kwargs)

    def validation(*args, **kwargs):
        events.append('validation')
        calls['validation'] += 1
        if failure == 'validation':
            raise RuntimeError('injected validation failure')
        return validate_original(*args, **kwargs)

    def fail(*args, **kwargs):
        raise RuntimeError('injected output failure')

    monkeypatch.setattr(engine, 'train_one_epoch', train)
    monkeypatch.setattr(engine, 'validate', validation)
    if failure == 'logging':
        monkeypatch.setattr(ExperimentRunner, '_log_epoch', fail)
    if failure == 'report':
        monkeypatch.setattr(engine.EvaluationReport, 'save', fail)
    with pytest.raises(RuntimeError, match='injected'):
        ExperimentRunner(cfg).run()

    directory = cfg.paths.runs_path / cfg.run_name
    checkpoint = torch.load(directory / 'ckpt/last.pt', weights_only=True)
    assert checkpoint['epoch'] == 0 and checkpoint['samples'] == 2
    assert checkpoint['validation_complete'] is (failure == 'logging')
    if failure == 'logging':
        assert (directory / 'ckpt/best.pt').is_file()

    def recovered_validation(*args, **kwargs):
        events.append('validation')
        return validate_original(*args, **kwargs)
    monkeypatch.setattr(engine, 'validate', recovered_validation)
    monkeypatch.setattr(ExperimentRunner, '_log_epoch', log_original)
    monkeypatch.setattr(engine.EvaluationReport, 'save', report_original)
    resumed = ExperimentRunner(cfg).run()
    assert calls['train'] == epochs  # Never repeat the saved training epoch.
    expected = ['train', 'validation']
    if failure != 'logging':
        expected.append('validation')
    if epochs == 2:
        expected.extend(['train', 'validation'])
    assert events == expected
    final = torch.load(directory / 'ckpt/last.pt', weights_only=True)
    assert final['validation_complete'] is True
    assert final['samples'] == 2 * epochs
    assert final['epoch'] == epochs - 1
    if epochs == 1:
        for key, value in checkpoint['model'].items():
            torch.testing.assert_close(value, final['model'][key])
        assert final['scheduler'] == checkpoint['scheduler']
    assert (directory / 'development/per_image.parquet').is_file()
    assert resumed.summary['training_complete'] is True
    assert resumed.summary['best_aic'] >= 0
