import numpy as np
import pytest
import torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape,constant', [((2, 17, 31), None), ((1, 8, 8), 0),
                                           ((1, 40, 24), 20), ((1, 192, 192), None)])
def test_categorical_backward_matches_dense(dtype, shape, constant):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from src.modules.jpeg_lookup_cuda import categorical_weight_gradient
    torch.manual_seed(71)
    bins = torch.randint(0, 21, shape, device='cuda', dtype=torch.uint8)
    if constant is not None:
        bins.fill_(constant)
    grad = torch.randn(shape[0], 64, *shape[1:], device='cuda', dtype=dtype)
    actual = categorical_weight_gradient(bins, grad)
    volume = torch.nn.functional.one_hot(bins.long(), 21).permute(0, 3, 1, 2).float()
    with torch.backends.cudnn.flags(allow_tf32=False):
        expected = torch.nn.grad.conv2d_weight(volume, (64, 21, 3, 3), grad.float(), padding=8, dilation=8)
    torch.testing.assert_close(actual.float(), expected.to(dtype).float(), rtol=0.01, atol=0.02)
    assert (actual.float() - expected).norm() / expected.norm() < 0.005


def test_foreach_normalization_matches_scalar_with_missing_gradients():
    from src.training.gradients import GradientNormalizer
    params = [torch.nn.Parameter(torch.randn(5, dtype=dtype)) for dtype in (torch.float32, torch.float64)]
    params.append(torch.nn.Parameter(torch.zeros(3)))
    for p in params[:2]:
        p.grad = torch.randn_like(p)
    expected = [p.grad.clone() / 7 for p in params[:2]]
    GradientNormalizer.divide(params, torch.tensor(7., dtype=torch.float64), foreach=True)
    for p, reference in zip(params, expected):
        torch.testing.assert_close(p.grad, reference)
    assert params[-1].grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('weight_grad,bias_grad', [(True, True), (True, False), (False, True)])
def test_triton_backward_autograd_routing(monkeypatch, weight_grad, bias_grad):
    from src.modules.jpeg_branch import JPEGCategoryConv2d
    layer = JPEGCategoryConv2d().cuda()
    layer.triton_backward = True
    layer.weight.requires_grad_(weight_grad)
    layer.bias.requires_grad_(bias_grad)
    def forbidden(*args, **kwargs):
        pytest.fail('Dense convolution backward was called')
    monkeypatch.setattr(torch.nn.grad, 'conv2d_weight', forbidden)
    bins = torch.randint(0, 21, (1, 24, 40), device='cuda', dtype=torch.uint8).transpose(1, 2)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = layer(bins)
    output.square().mean().backward()
    assert (layer.weight.grad is not None) == weight_grad
    assert (layer.bias.grad is not None) == bias_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_foreach_cuda_tensor_divisor():
    from src.training.gradients import GradientNormalizer
    p = torch.nn.Parameter(torch.zeros(1024, device='cuda'))
    p.grad = torch.randn_like(p)
    divisor = torch.tensor(16., device='cuda', dtype=torch.float64)
    expected = p.grad.clone().div_(divisor)
    GradientNormalizer.divide([p], divisor, foreach=True)
    torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
