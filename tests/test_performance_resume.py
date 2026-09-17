from dataclasses import replace

import numpy as np
import pytest

from src.config import load_experiment_config
from src.eval.protocol import EvaluationProtocol
from src.training.engine import ExperimentRunner
from src.training.runs import Run


@pytest.mark.parametrize('change', ['kernels', 'batch', 'loss'])
def test_resume_allows_execution_changes_but_preserves_recipe(tmp_path, change):
    cfg = load_experiment_config('configs/experiments/disentangle_b2_li760_r8_all_data_hard_pixel_ft.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path),
                  model=replace(cfg.model, jpeg_triton_backward=False, jpeg_specialize_width=True),
                  train=replace(cfg.train, foreach_grad_normalization=False))
    run = Run.create(tmp_path, cfg.run_name, tensorboard=False)
    protocol = EvaluationProtocol.load(cfg.dataset.protocol_path)
    run.save_snapshot({**cfg.to_flat_dict(), **protocol.provenance(train_all_data=True)})
    (run.dir / 'ckpt' / 'last.pt').touch()
    changed = replace(cfg, model=replace(cfg.model, jpeg_triton_backward=True, jpeg_specialize_width=False),
                      train=replace(cfg.train, foreach_grad_normalization=True))
    if change == 'batch':
        changed = replace(changed, train=replace(changed.train, batch_size=cfg.train.batch_size * 2))
    elif change == 'loss':
        changed = replace(changed, loss=replace(changed.loss, hard_pixel_weight=.2))
    if change == 'kernels':
        ExperimentRunner(changed)._check_resume_protocol()
    else:
        with pytest.raises(ValueError, match='train.batch_size' if change == 'batch' else 'loss.hard_pixel_weight'):
            ExperimentRunner(changed)._check_resume_protocol()
