import pytest
import torch

from src.config import ModelConfig


def test_similarity_option_is_explicit_and_boolean():
    assert ModelConfig.from_dict({'jpeg_similarity': True}).jpeg_similarity
    assert not ModelConfig().jpeg_similarity
    with pytest.raises(ValueError, match='jpeg_similarity'):
        ModelConfig.from_dict({'jpeg_similarity': 1})


def test_similarity_starts_as_identity_and_learns():
    from src.modules.jpeg_similarity import JPEGSimilarity

    block = JPEGSimilarity(8)
    features = torch.randn(2, 8, 7, 9, requires_grad=True)
    torch.testing.assert_close(block(features), features, rtol=0, atol=0)
    with torch.no_grad():
        block.channel_gate.fill_(0.1)
    block(features).square().mean().backward()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_similarity_matches_pairwise_reference_and_handles_single_patch():
    from src.modules.jpeg_similarity import JPEGSimilarity

    tokens = torch.randn(2, 64, 15)
    matrix = tokens.transpose(1, 2) @ tokens / 8
    matrix.diagonal(dim1=-2, dim2=-1).zero_()
    expected = matrix.sum(dim=1).softmax(dim=-1) * 15 - 1
    torch.testing.assert_close(JPEGSimilarity.patch_scores(tokens), expected)
    torch.testing.assert_close(JPEGSimilarity.patch_scores(tokens[:, :, :1]),
                               torch.zeros(2, 1))


def test_experiment_preserves_baseline_weights_and_roundtrips():
    from src.config import ExperimentConfig, SnapshotAdapter
    from src.training.builders import build_model, build_optimizer

    torch.manual_seed(42)
    baseline = build_model(ModelConfig(), pretrained=False)
    config = ExperimentConfig.from_dict({'run_name': 'iis', 'model': {'jpeg_similarity': True}})
    restored = ExperimentConfig.from_dict(SnapshotAdapter.normalize(config.to_dict()))
    torch.manual_seed(42)
    experiment = build_model(restored.model, pretrained=False)
    for name, weight in baseline.state_dict().items():
        torch.testing.assert_close(experiment.state_dict()[name], weight, rtol=0, atol=0)
    optimizer = build_optimizer(config.train, experiment)
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in optimized for p in experiment.forensic_fusion.similarity.parameters())
    experiment.load_state_dict(experiment.state_dict(), strict=True)


def test_similarity_integration_preserves_missing_jpeg_bypass():
    from src.modules.forensic_fusion import ForensicFusion
    from src.modules.jpeg_similarity import JPEGSimilarity

    class Branch(torch.nn.Module):
        def forward(self, jpeg, sizes):
            return {s: torch.randn(len(jpeg), 8, *sizes[s]) for s in sizes}

    fusion = ForensicFusion([8, 16, 32], [8, 8, 8], [8, 8, 8])
    fusion.branch = Branch()
    fusion.similarity = JPEGSimilarity(8)
    features = [torch.randn(2, 8, 4, 6) for _ in range(3)]
    calls = []
    hook = fusion.similarity.register_forward_hook(lambda module, args, result: calls.append(result.shape[0]))
    output = fusion(features, jpeg=[{'available': False}, {'available': True}])
    hook.remove()
    assert calls == [1]
    for original, updated in zip(features, output):
        torch.testing.assert_close(updated[0], original[0], rtol=0, atol=0)
