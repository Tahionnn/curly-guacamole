import math

import pytest
import torch

from src.config import ExperimentConfig, load_experiment_config
from src.losses import SegmentationLoss


def evaluate(target, logits, **options):
    out = {'logits': logits, 'cls_logits': torch.zeros(len(target), 1)}
    return SegmentationLoss(aux_weight=0, **options)(out, {'mask': target})


def test_boundary_gradients_only_in_two_sided_band():
    target = torch.zeros(1, 1, 24, 24)
    target[..., :, 12:] = 1
    logits = torch.zeros_like(target, requires_grad=True)
    loss = evaluate(target, logits, boundary_weight=.2, boundary_radius=4).components['boundary_bce']
    loss.backward()
    assert logits.grad[..., :8].eq(0).all()
    assert logits.grad[..., 16:].eq(0).all()
    assert logits.grad[..., 8:12].gt(0).all()  # Remove outer FP.
    assert logits.grad[..., 12:16].lt(0).all()  # Fill inner FN.
    assert loss.item() == pytest.approx(.2 * math.log(2))


def test_equal_image_weight_not_proportional_to_boundary_length():
    target = torch.zeros(3, 1, 32, 32)
    target[0, :, 14:18, 14:18] = 1
    target[1, :, 5:27, 5:27] = 1
    logits = torch.zeros_like(target)
    logits[1] = torch.where(target[1] > .5, -2., 2.)
    result = evaluate(target, logits, boundary_weight=1, boundary_radius=2)
    # Negative image has no contour; its extra loss is zero in the batch mean.
    expected = (math.log(2) + math.log1p(math.exp(2))) / 3
    assert result.components['boundary_bce'].item() == pytest.approx(expected)


@pytest.mark.parametrize('fill', [0., 1.])
def test_uniform_targets_have_no_artificial_frame_boundary(fill):
    target = torch.full((2, 1, 16, 16), fill)
    logits = torch.zeros_like(target, requires_grad=True)
    result = evaluate(target, logits, boundary_weight=.2, boundary_radius=4)
    assert result.components['boundary_bce'].item() == 0
    result.total.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_disabled_boundary_loss_is_exactly_legacy_and_soft_targets_are_preserved():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 8:] = 1
    target[..., 8] = .75
    logits = torch.randn_like(target, requires_grad=True)
    old = evaluate(target, logits)
    disabled = evaluate(target, logits, boundary_weight=0, boundary_radius=4)
    assert 'boundary_bce' not in disabled.components
    assert torch.equal(old.total, disabled.total)
    active = evaluate(target, logits, boundary_weight=.2, boundary_radius=1)
    expected = .2 * torch.nn.functional.binary_cross_entropy_with_logits(logits[..., 7:9], target[..., 7:9])
    torch.testing.assert_close(active.components['boundary_bce'], expected)


@pytest.mark.parametrize('options', [
    {'boundary_weight': -1}, {'boundary_weight': float('nan')},
    {'boundary_radius': 0}, {'boundary_radius': 1.5}, {'boundary_radius': True},
])
def test_invalid_boundary_settings_rejected_by_config_and_loss(options):
    raw = load_experiment_config('configs/baseline.yaml').to_dict()
    raw['loss'].update(options)
    with pytest.raises(ValueError, match='boundary|weights'):
        ExperimentConfig.from_dict(raw)
    with pytest.raises(ValueError, match='boundary|weights'):
        SegmentationLoss(**options)


def test_matched_finetune_configs_and_snapshot_roundtrip():
    candidate = load_experiment_config('configs/experiments/disentangle_b2_li760_r8_boundary_ft.yaml')
    control = load_experiment_config('configs/experiments/disentangle_b2_li760_r8_boundary_control_ft.yaml')
    assert candidate.loss.boundary_weight == .2 and candidate.loss.boundary_radius == 4
    assert control.loss.boundary_weight == 0
    a, b = candidate.to_dict(), control.to_dict()
    a.pop('run_name'); b.pop('run_name')
    a['loss']['boundary_weight'] = 0
    assert a == b
    assert ExperimentConfig.from_dict(candidate.to_dict()) == candidate
