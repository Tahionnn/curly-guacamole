import torch
import torch.nn.functional as F
import pytest


def test_dfdg_patch_is_pointwise_and_edge_has_spatial_support():
    from src.modules.dgforce import DFDGLevel

    level = DFDGLevel(8, reduction=2).eval()
    baseline = torch.zeros(1, 8, 5, 7)
    changed = baseline.clone()
    changed[:, :, 2, 3] = 1

    with torch.no_grad():
        patch0, edge0, *_ = level.disentangle(baseline)
        patch1, edge1, *_ = level.disentangle(changed)

    patch_delta = (patch1 - patch0).abs().sum(1)[0]
    edge_delta = (edge1 - edge0).abs().sum(1)[0]
    assert torch.count_nonzero(patch_delta) == 1
    assert torch.count_nonzero(edge_delta) > 1


def test_dfdg_adaptive_weights_sum_to_one_per_position():
    from src.modules.dgforce import DFDGLevel

    level = DFDGLevel(8, reduction=2)
    enriched, patch, edge, patch_logits, edge_logits, weights = level(
        torch.randn(2, 8, 5, 7))

    assert enriched.shape == patch.shape == edge.shape == (2, 8, 5, 7)
    assert patch_logits.shape == edge_logits.shape == (2, 1, 5, 7)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 5, 7))


def test_intra_scale_transfer_preserves_rectangular_shape_and_starts_as_identity():
    from src.modules.dgforce import IntraScaleTransfer

    transfer = IntraScaleTransfer(8)
    current = torch.randn(2, 8, 5, 7)
    shallow = torch.randn(2, 8, 5, 7)

    torch.testing.assert_close(transfer(current, shallow), current)


def test_cross_scale_transfer_preserves_target_shape_and_starts_as_identity():
    from src.modules.dgforce import CrossScaleTransfer

    transfer = CrossScaleTransfer(target_channels=8, source_channels=4, width=8, heads=2)
    target = torch.randn(2, 8, 5, 7)
    source = torch.randn(2, 4, 10, 14)

    torch.testing.assert_close(transfer(target, source), target)


def test_pvt_dgforce_encoder_matches_paper_layer_placement():
    from src.modules.pvt_dgforce import PVTDGForceEncoder

    encoder = PVTDGForceEncoder(pretrained=False, reduction=16,
                                attention_width=32, attention_heads=4)

    assert encoder.strides == [4, 8, 16, 32]
    assert encoder.depths == [3, 4, 6, 3]
    assert len(encoder.dfdg) == 13
    assert encoder.intra_pairs == (((0, 2),), ((0, 3),), ((0, 4), (1, 5)))
    assert set(encoder.cross_transfers) == {'s2_patch', 's2_edge', 's3_patch', 's3_edge'}


def test_pvt_dgforce_encoder_returns_layer_supervision_only_while_training():
    from src.modules.pvt_dgforce import PVTDGForceEncoder

    encoder = PVTDGForceEncoder(pretrained=False, reduction=16,
                                attention_width=32, attention_heads=4)
    image = torch.randn(1, 3, 64, 96)

    encoder.train()
    features, patch, edge = encoder(image, supervise=True)
    assert [feature.shape[-2:] for feature in features] == [(16, 24), (8, 12), (4, 6), (2, 3)]
    assert len(patch) == len(edge) == 13
    assert set(patch) == set(edge) == {
        *(f's1_b{i}' for i in range(3)),
        *(f's2_b{i}' for i in range(4)),
        *(f's3_b{i}' for i in range(6)),
    }

    encoder.eval()
    with torch.no_grad():
        evaluated, patch, edge = encoder(image, supervise=False)
    assert [feature.shape for feature in evaluated] == [feature.shape for feature in features]
    assert patch == edge == {}


def test_dgforce_loss_is_weighted_mean_of_paper_objectives():
    from src.losses import DGForceLoss, coarse_target, edge_band_target

    target = torch.tensor([[[[0., 0., 0., 0.], [0., 1., 1., 0.],
                             [0., 1., 1., 0.], [0., 0., 0., 0.]]]])
    out = {
        'logits': torch.full_like(target, 0.25),
        'patch_logits': {'a': torch.full((1, 1, 2, 2), -0.5),
                         'b': torch.full((1, 1, 1, 1), 0.75)},
        'edge_logits': {'a': torch.full((1, 1, 2, 2), 0.1),
                        'b': torch.full((1, 1, 1, 1), -0.2)},
    }
    criterion = DGForceLoss(mask_weight=2.0, patch_weight=1.0, edge_weight=1.0)

    result = criterion(out, {'mask': target})

    patch = torch.stack([
        F.binary_cross_entropy_with_logits(value, coarse_target(target, value.shape[-2:]))
        for value in out['patch_logits'].values()
    ]).mean()
    edge = torch.stack([
        F.binary_cross_entropy_with_logits(
            value, edge_band_target(coarse_target(target, value.shape[-2:]), 3))
        for value in out['edge_logits'].values()
    ]).mean()
    expected = 2 * F.binary_cross_entropy_with_logits(out['logits'], target) + patch + edge
    torch.testing.assert_close(result.total, expected)
    assert set(result.components) == {'mask_bce', 'patch_bce', 'edge_bce'}


def test_dgforce_loss_requires_all_training_outputs():
    from src.losses import DGForceLoss

    with pytest.raises(ValueError, match='patch_logits.*edge_logits'):
        DGForceLoss()({'logits': torch.zeros(1, 1, 4, 4)},
                      {'mask': torch.zeros(1, 1, 4, 4)})


def test_pvt_dgforce_experiment_config_is_isolated_and_uses_paper_loss():
    from src.config import load_experiment_config

    baseline = load_experiment_config('configs/baseline.yaml')
    config = load_experiment_config('configs/experiments/pvt_dgforce.yaml')

    assert config.model.architecture == 'pvt_dgforce'
    assert config.model.encoder == 'pvt_v2_b2'
    assert config.model.disentangle_levels == ()
    assert config.loss.mode == 'dgforce'
    assert (config.loss.mask_weight, config.loss.patch_weight,
            config.loss.edge_weight) == (2.0, 1.0, 1.0)
    assert config.loss.aux_weight == config.loss.dice_weight == 0.0
    assert config.dataset == baseline.dataset
    assert config.augmentation == baseline.augmentation
    assert config.train == baseline.train
