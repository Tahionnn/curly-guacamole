import pytest
import torch

from src.budget import count_gflops
from src.config import LossConfig, ModelConfig, load_experiment_config
from src.losses import SegmentationLoss, coarse_target, edge_band_target
from src.modules.forensic_disentangle import ForensicDisentangle
from src.training.builders import build_model

STRIDES = [4, 8, 16, 32]
CHANNELS = [64, 128, 320, 512]
ARMS = ['disentangle_supervision', 'disentangle_fuse', 'disentangle_fuse_cross32']


def pyramid(size=64, batch=2):
    return [torch.randn(batch, c, size // s, size // s) for s, c in zip(STRIDES, CHANNELS)]


def module(**kwargs):
    return ForensicDisentangle(STRIDES, CHANNELS, kwargs.pop('levels', STRIDES), **kwargs)


def arm(name):
    return load_experiment_config(f'configs/experiments/{name}.yaml')


def model_for(config, **kwargs):
    return build_model(config.model, aux_weight=config.loss.aux_weight, pretrained=False, **kwargs)


def jpeg_batch(native=128, batch=2):
    return [{'bins': torch.zeros(native, native, dtype=torch.uint8), 'qtable': torch.ones(8, 8),
             'geometry': (0, 0, native, native, 0, 0, 0)} for _ in range(batch)]


def test_untrained_module_leaves_features_untouched():
    """The gate starts at zero, so an arm begins bit-identical to its baseline."""
    features = pyramid()
    updated, _, _ = module(cross_strides=[32]).eval()([f.clone() for f in features], supervise=False)
    assert all(torch.equal(before, after) for before, after in zip(features, updated))


def test_supervision_mode_is_free_outside_training():
    block = module(mode='supervision')
    features = pyramid()
    counter = torch.utils.flop_counter.FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        updated, patch, edge = block([f.clone() for f in features], supervise=False)
    assert counter.get_total_flops() == 0
    assert not patch and not edge
    assert all(torch.equal(before, after) for before, after in zip(features, updated))


def test_supervision_mode_never_touches_features_in_training():
    block = module(mode='supervision').train()
    features = pyramid()
    updated, patch, edge = block([f.clone() for f in features], supervise=True)
    assert all(torch.equal(before, after) for before, after in zip(features, updated))
    assert sorted(patch) == STRIDES and sorted(edge) == STRIDES


def test_rectangular_input_keeps_its_shape():
    """The reference recovers the grid with int(sqrt(N)) and is wrong here."""
    features = [torch.randn(2, c, 128 // s, 96 // s) for s, c in zip(STRIDES, CHANNELS)]
    updated, patch, _ = module(cross_strides=[16, 32]).train()(features, supervise=True)
    assert [tuple(f.shape) for f in updated] == [tuple(f.shape) for f in features]
    assert patch[16].shape[-2:] == features[2].shape[-2:]


def test_every_parameter_receives_gradient():
    block = module(cross_strides=[32]).train()
    updated, patch, edge = block(pyramid(), supervise=True)
    total = sum(f.square().mean() for f in updated)
    for logits in list(patch.values()) + list(edge.values()):
        total = total + logits.square().mean()
    total.backward()
    assert [name for name, p in block.named_parameters() if p.grad is None] == []


@pytest.mark.parametrize('kwargs, match', [
    (dict(levels=[8, 16], mode='supervision', cross_strides=[16]), 'mode=fuse'),
    (dict(levels=[16], cross_strides=[32]), 'must also be'),
    (dict(levels=[8, 8]), 'unique'),
    (dict(levels=[64]), 'missing disentangle strides'),
    (dict(levels=[4], cross_strides=[4]), 'requires encoder stride'),
    (dict(levels=[8], attention_width=100, attention_heads=8), 'divisible by the head count'),
])
def test_module_rejects_impossible_configurations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        module(**kwargs)


@pytest.mark.parametrize('kwargs, match', [
    (dict(disentangle_levels=[8, 16], disentangle_mode='supervision', disentangle_cross_strides=[16]), 'mode=fuse'),
    (dict(disentangle_levels=[16], disentangle_cross_strides=[32]), 'must also be'),
    (dict(disentangle_cross_strides=[32]), 'requires disentangle_levels'),
    (dict(disentangle_levels=[8], disentangle_mode='both'), 'disentangle_mode'),
])
def test_model_config_rejects_impossible_configurations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ModelConfig(**kwargs)


@pytest.mark.parametrize('kwargs, match', [
    (dict(edge_band=4), 'odd'),
    (dict(patch_weight=-1.), 'patch_weight'),
    (dict(edge_max_pos_weight=.5), 'edge_max_pos_weight'),
])
def test_loss_config_rejects_impossible_supervision(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LossConfig(**kwargs)


def test_supervision_weights_require_the_module():
    from dataclasses import replace
    config = arm('disentangle_fuse')
    with pytest.raises(ValueError, match='requires model.disentangle_levels'):
        replace(config, model=ModelConfig())


def test_edge_band_is_a_thin_ring_around_the_edit():
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1
    band = edge_band_target(coarse_target(mask, (16, 16)), 3)
    assert band.max() == 1 and band.min() == 0
    # A ring, never the interior: the centre of a large edit stays negative.
    assert band[0, 0, 8, 8] == 0
    assert band[0, 0, 4, 4] == 1
    assert band.mean() < .5


def test_edge_band_of_an_untouched_frame_is_empty():
    assert edge_band_target(coarse_target(torch.zeros(2, 1, 64, 64), (16, 16)), 3).sum() == 0


@pytest.mark.parametrize('name', ARMS)
def test_arm_configs_stay_inside_the_budget(name):
    """Every arm is measured against its own baseline, not against a memory."""
    baseline = model_for(load_experiment_config('configs/baseline.yaml')).to('meta')
    candidate = model_for(arm(name)).to('meta')
    reference = count_gflops(baseline, 640, native_size=(1024, 1024))
    measured = count_gflops(candidate, 640, native_size=(1024, 1024))
    assert measured <= 100
    assert measured - reference < (1e-6 if name.endswith('supervision') else 1.0)


def test_training_step_produces_supervision_losses_and_eval_does_not():
    config = arm('disentangle_fuse_cross32')
    model = model_for(config).train()
    batch = {'image': torch.randn(2, 3, 128, 128), 'mask': torch.zeros(2, 1, 128, 128),
             'label': torch.tensor([[1.], [0.]])}
    batch['mask'][0, :, 20:60, 30:70] = 1
    out = model(batch['image'], jpeg=jpeg_batch())
    assert sorted(out['patch_logits']) == STRIDES and sorted(out['edge_logits']) == STRIDES

    components = SegmentationLoss(**config.loss.to_dict())(out, batch).components
    assert {'patch_bce', 'patch_dice', 'edge_bce'} <= set(components)
    assert all(torch.isfinite(value) for value in components.values())

    model.eval()
    with torch.no_grad():
        evaluated = model(batch['image'], jpeg=jpeg_batch())
    assert 'patch_logits' not in evaluated and 'edge_logits' not in evaluated


def test_baseline_model_has_no_disentangle_block_or_supervision():
    config = load_experiment_config('configs/baseline.yaml')
    assert config.model.disentangle_levels == ()
    assert (config.loss.patch_weight, config.loss.edge_weight) == (0., 0.)
    assert model_for(config).disentangle is None


@pytest.mark.parametrize('similarity', [False, True])
def test_disentangle_preserves_seeded_model_outputs(similarity):
    """Adding a zero-gated arm must not reinitialize the existing decoder/heads."""
    from dataclasses import replace

    config = load_experiment_config('configs/baseline.yaml')
    baseline_config = replace(config.model, jpeg_similarity=similarity)
    candidate_config = replace(baseline_config, disentangle_levels=tuple(STRIDES),
                               disentangle_cross_strides=(32,))
    torch.manual_seed(42)
    baseline = build_model(baseline_config, pretrained=False).eval()
    torch.manual_seed(42)
    candidate = build_model(candidate_config, pretrained=False).eval()
    candidate_state = candidate.state_dict()
    assert all(torch.equal(value, candidate_state[name])
               for name, value in baseline.state_dict().items())
    image = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        expected = baseline(image, jpeg=jpeg_batch())
        actual = candidate(image, jpeg=jpeg_batch())
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
