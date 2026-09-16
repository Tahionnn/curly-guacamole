"""Fused categorical convolution on CUDA; imported only when Triton is available."""

import torch
import triton
import triton.language as tl


@triton.jit
def _category_conv(bins, weight, bias, output, height, width: tl.constexpr, BLOCK: tl.constexpr):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch_channel = tl.program_id(1)
    channel = batch_channel % 64
    batch = batch_channel // 64
    y, x = position // width, position % width
    valid = position < height * width
    value = tl.full((BLOCK,), 0, tl.float32) + tl.load(bias + channel).to(tl.float32)
    for row in tl.static_range(3):
        for col in tl.static_range(3):
            yy, xx = y + (row - 1) * 8, x + (col - 1) * 8
            inside = valid & (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            category = tl.load(bins + batch * height * width + yy * width + xx,
                               mask=inside, other=0).to(tl.int32)
            offset = channel * 21 * 9 + category * 9 + row * 3 + col
            value += tl.load(weight + offset, mask=inside, other=0).to(tl.float32)
    tl.store(output + batch_channel * height * width + position, value, mask=valid)


# Runtime width variant for controlled comparisons on variable-size JPEGs.
@triton.jit
def _category_conv_dynamic_width(bins, weight, bias, output, height, width, BLOCK: tl.constexpr):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch_channel = tl.program_id(1)
    channel = batch_channel % 64
    batch = batch_channel // 64
    y, x = position // width, position % width
    valid = position < height * width
    value = tl.full((BLOCK,), 0, tl.float32) + tl.load(bias + channel).to(tl.float32)
    for row in tl.static_range(3):
        for col in tl.static_range(3):
            yy, xx = y + (row - 1) * 8, x + (col - 1) * 8
            inside = valid & (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            category = tl.load(bins + batch * height * width + yy * width + xx,
                               mask=inside, other=0).to(tl.int32)
            offset = channel * 21 * 9 + category * 9 + row * 3 + col
            value += tl.load(weight + offset, mask=inside, other=0).to(tl.float32)
    tl.store(output + batch_channel * height * width + position, value, mask=valid)


@triton.jit
def _category_weight_partials(bins, grad, partials, height, width, total,
                              SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    split = tl.program_id(0)
    offset = tl.program_id(1)
    channels = tl.program_id(2) * 32 + tl.arange(0, 32)
    categories = tl.arange(0, 32)
    accumulator = tl.zeros((32, 32), tl.float32)
    for start in range(split * BLOCK, total, SPLITS * BLOCK):
        positions = start + tl.arange(0, BLOCK)
        frame = positions // (height * width)
        local = positions % (height * width)
        y = local // width + (offset // 3 - 1) * 8
        x = local % width + (offset % 3 - 1) * 8
        valid = (positions < total) & (y >= 0) & (y < height) & (x >= 0) & (x < width)
        category = tl.load(bins + frame * height * width + y * width + x, valid, other=255)
        values = tl.load(grad + (frame[:, None] * 64 + channels[None, :]) * height * width
                         + local[:, None], positions[:, None] < total, other=0)
        # Only a register/shared-memory tile is expanded, never a full image volume.
        indicator = (categories[:, None] == category[None, :]).to(values.dtype)
        accumulator += tl.dot(indicator, values, input_precision='ieee')
    index = ((split * 64 + channels[None, :]) * 21 + categories[:, None]) * 9 + offset
    tl.store(partials + index, accumulator, categories[:, None] < 21)


def categorical_weight_gradient(bins, grad_output):
    """Bounded FP32 partial sums; deterministic reduction without global atomics."""
    batch, height, width = bins.shape
    total = batch * height * width
    splits = min(128, triton.cdiv(total, 256))
    partials = torch.empty((splits, 64, 21, 9), device=bins.device, dtype=torch.float32)
    with torch.cuda.device(bins.device):
        _category_weight_partials[(splits, 9, 2)](
            bins.contiguous(), grad_output.contiguous(), partials, height, width, total,
            SPLITS=splits, BLOCK=256)
    return partials.sum(0).reshape(64, 21, 3, 3).to(grad_output.dtype)


class CUDACategoryConv(torch.autograd.Function):
    """Gather forward with selectable dense or tiled categorical backward.

    Only bins and the small weight tensor are saved. The default backward
    reconstructs one-hot; the opt-in tiled variant avoids that full volume.
    """

    @staticmethod
    def forward(ctx, bins, weight, bias, specialize_width=True, triton_backward=False):
        ctx.input_count = len(ctx.needs_input_grad)
        ctx.triton_backward = triton_backward
        ctx.save_for_backward(bins, weight)
        batch, height, width = bins.shape
        output = torch.empty((batch, 64, height, width), device=bins.device, dtype=weight.dtype)
        with torch.cuda.device(bins.device):
            kernel = _category_conv if specialize_width else _category_conv_dynamic_width
            kernel[(triton.cdiv(height * width, 256), batch * 64)](
                bins, weight, bias, output, height, width, BLOCK=256)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bins, weight = ctx.saved_tensors
        grad_weight = grad_bias = None
        with torch.autocast(device_type='cuda', enabled=False):
            grad_output = grad_output.to(weight.dtype).contiguous()
            if ctx.needs_input_grad[1]:
                if ctx.triton_backward:
                    grad_weight = categorical_weight_gradient(bins, grad_output)
                else:
                    categories = torch.arange(21, device=bins.device, dtype=torch.uint8)[None, :, None, None]
                    volume = (bins[:, None] == categories).to(weight.dtype)
                    grad_weight = torch.nn.grad.conv2d_weight(
                        volume, weight.shape, grad_output, padding=8, dilation=8)
            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3))
        return (None, grad_weight, grad_bias, None, None)[:ctx.input_count]
