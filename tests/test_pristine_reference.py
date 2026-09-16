# Initialize MKL before torch on Windows, including when running this file alone.
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import pytest
import torch

from src.modules.pristine_reference import PristineReferenceHead


def test_reference_weights_are_per_image_and_correction_starts_at_zero():
    head = PristineReferenceHead(12, width=8, references=4).eval()
    x = torch.randn(2, 12, 9, 11)
    out = head(x)
    assert out['correction'].shape == (2, 1, 9, 11)
    assert torch.count_nonzero(out['correction']) == 0
    torch.testing.assert_close(out['reference_weights'].flatten(2).sum(-1), torch.ones(2, 4))
    single = head(x[:1])
    torch.testing.assert_close(single['similarities'], out['similarities'][:1])
    assert not torch.allclose(out['reference_weights'][:, 0], out['reference_weights'][:, 1])


@pytest.mark.parametrize('target_value', [0., 1.])
def test_reference_loss_is_finite_for_empty_and_full_masks(target_value):
    head = PristineReferenceHead(12, width=8, references=4)
    out = head(torch.randn(2, 12, 9, 11))
    target = torch.full((2, 1, 72, 88), target_value)
    terms = head.supervision(out, target, torch.tensor([True, False]))
    loss = sum(terms.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert head.authenticity.weight.grad is not None
    assert torch.isfinite(head.authenticity.weight.grad).all()


def test_no_jpeg_has_zero_supervision_and_no_mask_leaks_into_forward():
    head = PristineReferenceHead(12, width=8, references=4)
    out = head(torch.randn(2, 12, 9, 11))
    terms = head.supervision(out, torch.rand(2, 1, 72, 88), torch.zeros(2, dtype=torch.bool))
    assert sum(terms.values()).item() == 0
    with pytest.raises(TypeError):
        head(torch.randn(2, 12, 9, 11), target=torch.zeros(2, 1, 9, 11))


def test_supervision_prefers_clean_references_even_when_most_of_image_is_changed():
    target = torch.ones(1, 1, 8, 8)
    target[:, :, :2, :2] = 0
    clean = torch.zeros(1, 4, 8, 8)
    clean[:, :, :2, :2] = .25
    contaminated = torch.zeros_like(clean)
    contaminated[:, :, 4:6, 4:6] = .25
    logits = torch.zeros(1, 1, 8, 8)
    good = PristineReferenceHead.supervision(
        {'authenticity_logits': logits, 'reference_weights': clean}, target, torch.tensor([True]))
    bad = PristineReferenceHead.supervision(
        {'authenticity_logits': logits, 'reference_weights': contaminated}, target, torch.tensor([True]))
    assert good['reference_contamination'] == 0
    assert bad['reference_contamination'] > 10


def test_eval_criterion_does_not_require_training_only_reference_outputs():
    from src.losses import SegmentationLoss
    out = {'logits': torch.zeros(1, 1, 8, 8), 'cls_logits': torch.zeros(1, 1)}
    batch = {'mask': torch.zeros(1, 1, 8, 8)}
    assert torch.isfinite(SegmentationLoss(reference_weight=.1).eval()(out, batch).total)


def test_config_rejects_supervision_without_reference_head():
    from src.config import ExperimentConfig
    with pytest.raises(ValueError, match='reference'):
        ExperimentConfig.from_dict({'run_name': 'invalid', 'loss': {'reference_weight': .1}})


def test_additive_checkpoint_loading_allows_only_complete_new_head():
    from src.training.finetune import FinetuneWeights
    model = torch.nn.Module()
    model.core = torch.nn.Linear(3, 2)
    model.reference_head = PristineReferenceHead(12)
    source = {'core.weight': torch.full_like(model.core.weight, 3.),
              'core.bias': torch.full_like(model.core.bias, 2.)}
    fresh = {k: v.clone() for k, v in model.reference_head.state_dict().items()}
    FinetuneWeights.load(model, source, initialize_reference=True)
    torch.testing.assert_close(model.core.weight, source['core.weight'])
    for key, value in fresh.items():
        torch.testing.assert_close(model.reference_head.state_dict()[key], value)
    with pytest.raises(ValueError, match='checkpoint'):
        FinetuneWeights.load(model, {'core.weight': source['core.weight']}, initialize_reference=True)
    with pytest.raises(ValueError, match='checkpoint'):
        FinetuneWeights.load(model, {**source, 'unexpected': torch.zeros(1)}, initialize_reference=True)


def test_integration_preserves_source_at_initialization_and_png_after_head_changes():
    from dataclasses import replace
    from src.config import ModelConfig
    from src.training.builders import build_model
    from src.training.finetune import FinetuneWeights
    cfg = ModelConfig()
    base = build_model(cfg, pretrained=False).eval()
    model = build_model(replace(cfg, pristine_reference=True), pretrained=False).eval()
    FinetuneWeights.load(model, base.state_dict(), initialize_reference=True)
    inputs = torch.randn(2, 3, 64, 64)
    jpeg = [{'bins': torch.zeros(64, 64, dtype=torch.uint8), 'qtable': torch.ones(8, 8),
             'geometry': (0, 0, 64, 64, 0, 0, 0)}, {'available': False}]
    with torch.no_grad():
        expected = base(inputs, jpeg=jpeg)
        actual = model(inputs, jpeg=jpeg)
        torch.testing.assert_close(actual['logits'], expected['logits'], rtol=0, atol=0)
        model.reference_head.correction[-1].bias.fill_(1.)
        changed = model(inputs, jpeg=jpeg)
    torch.testing.assert_close(changed['logits'][1], expected['logits'][1], rtol=0, atol=0)
    assert not torch.equal(changed['logits'][0], expected['logits'][0])
    assert 'reference' not in actual


def test_new_head_has_separate_lr_and_all_parameters_are_optimized_once():
    from src.config import TrainConfig
    from src.training.builders import build_optimizer
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(3, 2)
    model.reference_head = PristineReferenceHead(12)
    cfg = TrainConfig(encoder_lr=1e-5, head_lr=3e-5, reference_lr=3e-4)
    optimizer = build_optimizer(cfg, model)
    rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    assert len(rates) == sum(len(g['params']) for g in optimizer.param_groups) == len(list(model.parameters()))
    assert all(rates[id(p)] == cfg.reference_lr for p in model.reference_head.parameters())
    assert all(rates[id(p)] == cfg.encoder_lr for p in model.encoder.parameters())


def test_reference_head_stays_below_full_hd_budget():
    from dataclasses import replace
    from src.budget import count_gflops
    from src.config import load_experiment_config
    from src.training.builders import build_model
    cfg = load_experiment_config('configs/experiments/disentangle_b2_li760_r8_long.yaml')
    with torch.device('meta'):
        base = build_model(cfg.model, pretrained=False)
        model = build_model(replace(cfg.model, pristine_reference=True), pretrained=False)
        base_flops = count_gflops(base, 760, native_size=(1080, 1920))
        flops = count_gflops(model, 760, native_size=(1080, 1920))
    assert base_flops < flops < 100


def test_optional_head_preserves_common_initialization_and_sampler_rng():
    from src.config import ModelConfig
    from src.training.builders import build_model
    torch.manual_seed(43)
    base = build_model(ModelConfig(), pretrained=False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(43)
    model = build_model(ModelConfig(pristine_reference=True), pretrained=False)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
