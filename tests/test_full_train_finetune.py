
import numpy as np
import pytest
import torch

from src.config import TrainConfig, load_experiment_config
from src.training.builders import build_loaders, build_scheduler


class Rows(torch.utils.data.Dataset):
    is_negative = np.array([False] * 8 + [True] * 3)

    def __len__(self):
        return 11

    def __getitem__(self, index):
        return {"index": index}


def test_final_pass_covers_every_row_including_partial_batch():
    cfg = TrainConfig(device='cpu', workers=0, epochs=5, samples_per_epoch=8,
                      batch_size=3, grad_accum_steps=2, full_pass_epochs=1)
    loader, _ = build_loaders(cfg, Rows(), Rows())
    assert len(loader) == 3
    loader.sampler.set_epoch(4)
    assert len(loader) == 4
    seen = torch.cat([batch["index"] for batch in loader]).tolist()
    assert sorted(seen) == list(range(11))
    loader.sampler.generator = torch.Generator().manual_seed(42)
    first = list(loader.sampler)
    loader.sampler.generator = torch.Generator().manual_seed(42)
    assert list(loader.sampler) == first


def test_full_train_schedule_and_resume():
    cfg = TrainConfig(epochs=5, full_pass_epochs=1, warmup_fraction=.05,
                      min_lr_factor=.1)
    parameter = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.SGD([parameter], lr=1.)
    scheduler = build_scheduler(cfg, opt, 10, full_steps_per_epoch=30)
    curve = scheduler.lr_lambdas[0]
    assert curve(0) < 1
    assert curve(2) == 1
    assert curve(39) == 1
    assert curve(40) == 1
    assert curve(55) == pytest.approx(.55)
    assert curve(70) == pytest.approx(.1)
    for _ in range(45):
        opt.step()
        scheduler.step()
    saved_opt, saved_sched = opt.state_dict(), scheduler.state_dict()
    other_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=1.)
    other = build_scheduler(cfg, other_opt, 10, full_steps_per_epoch=30)
    other_opt.load_state_dict(saved_opt)
    other.load_state_dict(saved_sched)
    opt.step()
    scheduler.step()
    other_opt.step()
    other.step()
    assert other.get_last_lr() == scheduler.get_last_lr()


def test_recipe():
    cfg = load_experiment_config('configs/baseline_long.yaml')
    assert cfg.train.epochs == 18
    assert cfg.train.full_pass_epochs == 3
    assert cfg.augmentation.final_full_frame_epochs == 3
    assert cfg.loss.aux_weight == .4


@pytest.mark.parametrize('value', [-1, 1.5, True, 6])
def test_invalid_full_train_epochs(value):
    with pytest.raises(ValueError, match='full_pass_epochs'):
        TrainConfig(epochs=5, full_pass_epochs=value)


@pytest.mark.parametrize('foreach', [False, True])
def test_partial_accumulation_keeps_update_scale(foreach):
    from src.training.builders import build_amp
    from src.training.engine import train_one_epoch
    from tests.test_engine import CountingEma, CountingScheduler, _cpu_config

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, image, fmap=None):
            return {'logits': self.weight.expand(len(image), 1, 2, 2),
                    'cls_logits': self.weight.expand(len(image), 1)}

    results = []
    for accum in (1, 4):
        cfg = _cpu_config(grad_accum_steps=accum, full_pass_epochs=1,
                          foreach_grad_normalization=foreach)
        model = Model()
        amp = build_amp(cfg.train)
        batch = {'image': torch.zeros(1, 3, 2, 2), 'mask': torch.ones(1, 1, 2, 2),
                 'label': torch.ones(1, 1)}
        train_one_epoch(model=model, loader=[batch], optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                        scheduler=CountingScheduler(), ema=CountingEma(), scaler=amp.scaler(),
                        amp=amp, config=cfg, device=torch.device('cpu'))
        results.append(model.weight.item())
    assert results[0] == pytest.approx(results[1])
