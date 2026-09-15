import torch

from src.modules.forensic_disentangle import ForensicDisentangle


def test_parallel_attention_reads_original_context_and_backpropagates():
    block = ForensicDisentangle(
        [4, 8, 16, 32], [16, 16, 16, 16], [16, 32],
        cross_strides=(32,), attention_width=8, attention_heads=2,
        parallel_16_32=True).eval()
    # Nonzero gates ensure a sequential implementation cannot pass by identity.
    with torch.no_grad():
        block.cross['32'].channel_gate.fill_(1)
        block.parallel_return16.channel_gate.fill_(1)
    features = [torch.randn(2, 16, h, w, requires_grad=True)
                for h, w in [(24, 32), (12, 16), (6, 8), (3, 4)]]
    f16 = block.blocks['16'](features[2], supervise=False)[0]
    f32 = block.blocks['32'](features[3], supervise=False)[0]
    expected16 = block.parallel_return16(f16, f32)
    expected32 = block.cross['32'](f32, f16)
    actual, _, _ = block(features, supervise=False)
    torch.testing.assert_close(actual[2], expected16)
    torch.testing.assert_close(actual[3], expected32)
    assert not torch.allclose(actual[2], block.parallel_return16(f16, expected32))
    (actual[2].square().mean() + actual[3].square().mean()).backward()
    for module in (block.cross['32'], block.parallel_return16):
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in module.parameters())
