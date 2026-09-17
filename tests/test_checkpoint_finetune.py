from dataclasses import replace

import pytest
import torch

from src.config import TrainConfig, load_experiment_config
from src.training.builders import build_loaders, build_scheduler
from src.training.engine import ExperimentRunner
from tests.test_full_train_finetune import Rows


def test_all_epochs_cover_full_train_and_decay_lr():
    cfg = TrainConfig(device='cpu', workers=0, epochs=3, full_pass_epochs=3,
                      batch_size=3, grad_accum_steps=1, warmup_fraction=0)
    loader, _ = build_loaders(cfg, Rows(), Rows())
    for epoch in range(3):
        loader.sampler.set_epoch(epoch)
        assert sorted(torch.cat([batch["index"] for batch in loader]).tolist()) == list(range(11))
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=1.)
    scheduler = build_scheduler(cfg, opt, 4, full_steps_per_epoch=4)
    curve = scheduler.lr_lambdas[0]
    assert curve(0) == 1.
    assert curve(6) == pytest.approx(.51)
    assert curve(12) == pytest.approx(.02)


@pytest.mark.parametrize('resume_exists', [False, True])
@pytest.mark.parametrize('weights', ['model', 'ema'])
def test_finetune_loads_model_weights_unless_resuming(tmp_path, monkeypatch, resume_exists, weights):
    import src.training.engine as engine

    cfg = load_experiment_config('configs/baseline.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path),
                  train=replace(cfg.train, device='cpu', finetune_from='source/ckpt/last.pt',
                                finetune_weights=weights))
    source = tmp_path / cfg.train.finetune_from
    source.parent.mkdir(parents=True)
    torch.save({'model': {'weight': torch.tensor([[7.]])},
                'ema': {'weight': torch.tensor([[9.]])}, 'cfg': cfg.to_flat_dict()}, source)
    if resume_exists:
        target = tmp_path / cfg.run_name / 'ckpt/last.pt'
        target.parent.mkdir(parents=True)
        target.touch()
        source.unlink()  # Resume must not depend on the original checkpoint.
    calls = []
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    monkeypatch.setattr(engine, 'build_model', lambda config, pretrained, aux_weight: calls.append(pretrained) or model)
    monkeypatch.setattr(engine, 'configure_memory_format', lambda model: model)

    class Protocol:
        def verify_run(self, snapshot):
            assert snapshot == cfg.to_flat_dict()

    monkeypatch.setattr(engine.EvaluationProtocol, 'load', lambda path: Protocol())
    actual = ExperimentRunner(cfg)._build_training_model()
    assert calls == [False]
    assert actual.weight.item() == (0. if resume_exists else 9. if weights == 'ema' else 7.)


@pytest.mark.parametrize('destination_all_data', [False, True])
def test_all_data_source_requires_all_data_destination(tmp_path, destination_all_data):
    from src.eval.protocol import EvaluationProtocol
    cfg = load_experiment_config('configs/baseline.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path),
                  augmentation=replace(cfg.augmentation, final_full_frame_epochs=3),
                  train=replace(cfg.train, epochs=3, full_pass_epochs=3, train_all_data=True,
                                finetune_from='source/ckpt/last.pt', finetune_weights='ema'))
    snapshot = {**cfg.to_flat_dict(),
                **EvaluationProtocol.load(cfg.dataset.protocol_path).provenance(train_all_data=True)}
    path = tmp_path / cfg.train.finetune_from
    path.parent.mkdir(parents=True)
    torch.save({'ema': {'weight': torch.tensor([[9.]])}, 'cfg': snapshot}, path)
    destination = replace(cfg, train=replace(cfg.train, train_all_data=destination_all_data))
    model = torch.nn.Linear(1, 1, bias=False)
    if destination_all_data:
        ExperimentRunner(destination)._load_finetune_weights(model)
        assert model.weight.item() == 9.
    else:
        with pytest.raises(ValueError, match='all-data'):
            ExperimentRunner(destination)._load_finetune_weights(model)
