import pytest
import torch

from src.config import ModelConfig


def test_bifpn_config_is_opt_in_and_validated():
    assert ModelConfig().bifpn_repeats == 0
    with pytest.raises(ValueError):
        ModelConfig(bifpn_width=0)
    with pytest.raises(ValueError):
        ModelConfig(bifpn_repeats=-1)


def test_fusion_uses_positive_normalized_weights():
    from src.modules.bifpn import WeightedFusion
    fusion = WeightedFusion(2)
    with torch.no_grad():
        fusion.weights.copy_(torch.tensor([1., 3.]))
    actual = fusion([torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 5.)])
    torch.testing.assert_close(actual, torch.full_like(actual, 16 / 4.0001))
    with torch.no_grad():
        fusion.weights.fill_(-1)
    assert torch.isfinite(fusion([actual, actual])).all()


@pytest.mark.parametrize('repeats', [1, 2])
def test_bifpn_preserves_odd_rectangular_shapes_and_trains_all_nodes(repeats):
    from src.modules.bifpn import BiFPN
    block = BiFPN([64, 128, 320, 512], [4, 8, 16, 32], width=16, repeats=repeats)
    features = [torch.randn(2, c, h, w, requires_grad=True)
                for c, h, w in [(64, 31, 23), (128, 16, 12), (320, 8, 6), (512, 4, 3)]]
    updated = block(features)
    assert [x.shape for x in updated] == [x.shape for x in features]
    sum(x.square().mean() for x in updated).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in features)


def test_bifpn_segmenter_loss_and_checkpoint_roundtrip():
    from src.training.builders import build_model
    from src.losses import SegmentationLoss
    config = ModelConfig(encoder='pvt_v2_b2_li', bifpn_width=16, bifpn_repeats=1,
                         disentangle_levels=(4, 8, 16, 32))
    model = build_model(config, pretrained=False)
    batch = {'image': torch.randn(2, 3, 64, 96), 'mask': torch.zeros(2, 1, 64, 96),
             'label': torch.tensor([[1.], [0.]])}
    batch['mask'][0, :, 10:40, 20:60] = 1
    jpeg = [{'available': False}] * 2
    output = model(batch['image'], jpeg=jpeg)
    loss = SegmentationLoss(patch_weight=.5, edge_weight=.5)(output, batch)
    loss.total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.bifpn.parameters())
    assert {'aux_logits', 'patch_logits', 'edge_logits'} <= output.keys()
    assert output['logits'].shape == batch['mask'].shape
    clone = build_model(ModelConfig.from_dict(config.to_dict()), pretrained=False)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    clone.eval()
    with torch.no_grad():
        expected = model(batch['image'], jpeg=jpeg)
        actual = clone(batch['image'], jpeg=jpeg)
    assert set(actual) == {'logits', 'cls_logits'}
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
