from dataclasses import replace
import json

import pytest
import numpy as np
import torch

from src.config import TrainConfig
from src.training.engine import ExperimentRunner
from tests.test_epoch_completion import configure_tiny_run


def test_blind_epochs_never_validate_and_resume(tmp_path, monkeypatch):
    import src.training.engine as engine

    cfg = configure_tiny_run(tmp_path, monkeypatch)
    cfg = replace(cfg, augmentation=replace(cfg.augmentation, final_full_frame_epochs=3),
                  train=replace(cfg.train, train_all_data=True, epochs=3,
                                    full_pass_epochs=3, resume=True))
    from types import SimpleNamespace
    monkeypatch.setattr(engine.EvaluationProtocol, 'load', lambda path: SimpleNamespace(
        digest='synthetic-test', provenance=lambda **kw: {'protocol_digest': 'synthetic-test'},
        verify_run=lambda snapshot, **kw: None))
    monkeypatch.setattr(engine, 'validate', lambda *a, **kw: pytest.fail('validation forbidden'))
    run = ExperimentRunner(cfg).run()
    checkpoint = torch.load(run.dir / 'ckpt/last.pt', weights_only=True)
    assert checkpoint['epoch'] == 2 and checkpoint['samples'] == 6
    assert checkpoint['validation_complete'] is True
    assert not (run.dir / 'ckpt/best.pt').exists()
    assert run.summary['evaluation_role'] == 'none'
    assert run.summary['training_complete'] is True
    for line in run.jsonl_path.read_text().splitlines():
        assert not any(key.startswith('val/') for key in json.loads(line))
    monkeypatch.setattr(engine, 'train_one_epoch', lambda **kw: pytest.fail('completed run repeated'))
    ExperimentRunner(cfg).run()


def test_all_data_split_includes_every_protocol_role(monkeypatch):
    from src.config import load_experiment_config
    from src.eval.protocol import EvaluationProtocol, rows_digest
    cfg = load_experiment_config('configs/baseline_long.yaml')
    cfg = replace(cfg, augmentation=replace(cfg.augmentation, final_full_frame_epochs=cfg.train.epochs),
                  train=replace(cfg.train, train_all_data=True, full_pass_epochs=cfg.train.epochs))
    protocol = EvaluationProtocol.load(cfg.dataset.protocol_path)
    rows, validation = ExperimentRunner(cfg)._split_data()
    assert len(validation) == 0
    assert len(rows) == sum(len(protocol.rows(role)) for role in protocol.ROLES)
    assert set(rows.role) == set(protocol.ROLES)
    assert not rows.chng_img_path.duplicated().any()
    provenance = protocol.provenance(train_all_data=True)
    assert provenance['training_rows_digest'] == rows_digest(rows)
    with pytest.raises(ValueError):
        protocol.verify_run(provenance)


def test_all_data_requires_full_passes():
    with pytest.raises(ValueError, match='train_all_data'):
        TrainConfig(train_all_data=True)
