import math

import numpy as np
import pytest
import torch

from src.config import LossConfig, TrainConfig
from src.losses import SegmentationLoss
from src.training.builders import build_scheduler


def hard_terms(logits, target, **kwargs):
    criterion = SegmentationLoss(aux_weight=0, hard_pixel_weight=1.,
                                 hard_pixel_fraction=.5, hard_pixel_radius=0, **kwargs)
    return criterion({'logits': logits, 'cls_logits': logits.mean((2, 3))},
                     {'mask': target}).components


def test_mining_balances_classes_and_only_updates_hard_pixels():
    logits = torch.tensor([[[[-2., 2., -2., -1., 1., 2.]]]], requires_grad=True)
    target = torch.tensor([[[[1., 1., 0., 0., 0., 0.]]]])
    terms = hard_terms(logits, target)
    assert terms['hard_pixel_positive'].item() == pytest.approx(math.log1p(math.exp(2)))
    assert terms['hard_pixel_negative'].item() == pytest.approx(
        (math.log1p(math.exp(1)) + math.log1p(math.exp(2))) / 2)
    (terms['hard_pixel_positive'] + terms['hard_pixel_negative']).backward()
    assert logits.grad[0, 0, 0, 0] < 0
    assert logits.grad[0, 0, 0, 4] > 0
    assert logits.grad[0, 0, 0, 5] > 0
    assert torch.equal(logits.grad[0, 0, 0, 1:4], torch.zeros(3))


@pytest.mark.parametrize('foreground', [0., 1.])
def test_uniform_masks_have_finite_loss_and_gradient(foreground):
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = torch.full_like(logits, foreground)
    terms = hard_terms(logits, target)
    loss = terms['hard_pixel_positive'] + terms['hard_pixel_negative']
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_boundary_exclusion_skips_vanished_positive_without_nan():
    logits = torch.zeros(1, 1, 7, 7, requires_grad=True)
    target = torch.zeros_like(logits)
    target.data[:, :, 3, 3] = 1
    criterion = SegmentationLoss(aux_weight=0, hard_pixel_weight=1., hard_pixel_radius=1)
    terms = criterion({'logits': logits, 'cls_logits': logits.mean((2, 3))},
                      {'mask': target}).components
    assert terms['hard_pixel_positive'].item() == 0
    loss = terms['hard_pixel_positive'] + terms['hard_pixel_negative']
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, :, 2:5, 2:5]) == 0
    assert torch.count_nonzero(logits.grad) > 0


def test_no_scheduler_keeps_optimizer_rates_from_first_step():
    cfg = TrainConfig(scheduler='none', epochs=3, full_pass_epochs=3)
    opt = torch.optim.SGD([{'params': [torch.nn.Parameter(torch.zeros(()))], 'lr': 1e-5},
                           {'params': [torch.nn.Parameter(torch.zeros(()))], 'lr': 3e-5}])
    assert build_scheduler(cfg, opt, 2) is None
    for _ in range(6):
        opt.step()
        assert [g['lr'] for g in opt.param_groups] == [1e-5, 3e-5]


@pytest.mark.parametrize('kwargs', [{'hard_pixel_weight': -1}, {'hard_pixel_fraction': 0},
                                   {'hard_pixel_fraction': 1.1}, {'hard_pixel_radius': -1}])
def test_invalid_mining_config(kwargs):
    with pytest.raises(ValueError):
        LossConfig(**kwargs)
