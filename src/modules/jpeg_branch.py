"""CAT-Net artifact stem followed by our lightweight forensic pyramid."""

import logging
from functools import lru_cache

import numpy as np
import torch
from numpy.core.multiarray import scalar as numpy_scalar
from torch import nn
from torch.nn import functional as F

from src.modules.dws_conv2d import DWSConv2d


@lru_cache(maxsize=1)
def _cuda_category_conv():
    try:
        from src.modules.jpeg_lookup_cuda import CUDACategoryConv
    except ModuleNotFoundError as error:
        if error.name != 'triton':
            raise
        logging.getLogger(__name__).warning(
            'Triton unavailable: JPEG lookup uses the slower PyTorch fallback on CUDA.')
        return None
    return CUDACategoryConv


class JPEGFrameBatchNorm2d(nn.BatchNorm2d):
    """Use native-frame statistics in eval, matching this branch's training.

    Keep standard training behavior and checkpoint buffers for compatibility.
    Evaluation neither reads nor updates the running statistics.
    """

    def forward(self, x):
        if self.training:
            return super().forward(x)
        return F.batch_norm(x, None, None, self.weight, self.bias,
                            training=True, momentum=0.0, eps=self.eps)

    @classmethod
    def replace_in(cls, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                replacement = cls(child.num_features, eps=child.eps,
                                  momentum=child.momentum, affine=child.affine,
                                  track_running_stats=child.track_running_stats)
                replacement.load_state_dict(child.state_dict())
                setattr(module, name, replacement)
            else:
                cls.replace_in(child)


class _PointwiseConvWithoutCuDNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        with torch.backends.cudnn.flags(enabled=False):
            return F.conv2d(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # PyTorch queries cuDNN flags again in backward, independently of forward.
        with torch.backends.cudnn.flags(enabled=False):
            grad_x, grad_weight, _ = torch.ops.aten.convolution_backward(
                grad_output, x, weight, None, [1, 1], [0, 0], [1, 1],
                False, [0, 0], 1, [*ctx.needs_input_grad, False],
            )
        return grad_x, grad_weight


class JPEGPointwiseConv2d(nn.Conv2d):
    """Keep Conv2d checkpoint keys while avoiding per-shape cuDNN compilation."""

    def __init__(self):
        super().__init__(64, 4, 1, bias=False)
        self.use_matmul = True

    def forward(self, x):
        if not self.use_matmul:
            if x.device.type == 'meta':
                return super().forward(x)
            weight = self.weight
            if torch.is_autocast_enabled(x.device.type):
                dtype = torch.get_autocast_dtype(x.device.type)
                x, weight = x.to(dtype), weight.to(dtype)
            with torch.autocast(device_type=x.device.type, enabled=False):
                return _PointwiseConvWithoutCuDNN.apply(x, weight)
        # A 1x1 convolution is a matrix product over channels at each position.
        # Native autograd handles both gradients without a convolution backend.
        return (self.weight.flatten(1) @ x.flatten(2)).reshape(x.shape[0], 4, *x.shape[-2:])


class JPEGCategoryConv2d(nn.Conv2d):
    """Evaluate the CAT-Net one-hot convolution by gathering category weights.

    Keep the original Conv2d parameters for strict checkpoint compatibility.
    BxHxW category inputs use lookup; Bx21xHxW dense inputs remain supported
    for reference comparisons. Category 21 is reserved for zero padding.
    """

    def __init__(self):
        super().__init__(21, 64, 3, dilation=8, padding=8)
        self.specialize_width = True
        self.triton_backward = False

    def forward(self, bins):
        if bins.ndim == 4:
            return super().forward(bins)
        batch, height, width = bins.shape
        dtype = self.weight.dtype
        if bins.device.type != 'meta' and torch.is_autocast_enabled(bins.device.type):
            dtype = torch.get_autocast_dtype(bins.device.type)
        weight, bias = self.weight.to(dtype), self.bias.to(dtype)
        if bins.device.type == 'cuda' and dtype in (torch.float32, torch.float16, torch.bfloat16):
            kernel = _cuda_category_conv()
            if kernel is not None:
                return kernel.apply(bins.contiguous(), weight.contiguous(), bias.contiguous(),
                                    self.specialize_width, self.triton_backward)
        indices = F.pad(bins.long(), (8, 8, 8, 8), value=21)
        # Match convolution's low-precision inputs with FP32 accumulation.
        if dtype in (torch.float16, torch.bfloat16):
            weight, bias = weight.float(), bias.float()
        result = bias[None, None, None, :].expand(batch, height, width, -1)
        for row in range(3):
            for col in range(3):
                table = F.pad(weight[:, :, row, col].T, (0, 0, 0, 1))
                neighbors = indices[:, row*8:row*8+height, col*8:col*8+width]
                result = result + F.embedding(neighbors, table)
        return result.to(dtype).permute(0, 3, 1, 2).contiguous()


class JPEGArtifactModule(nn.Module):
    """Frequency-wise CAT-Net stem; input is min(abs(quantized Y DCT), 20)."""

    def __init__(self):
        super().__init__()
        self.dc_layer0_dil = nn.Sequential(
            JPEGCategoryConv2d(),
            nn.BatchNorm2d(64, momentum=0.01), nn.ReLU(inplace=True))
        self.dc_layer1_tail = nn.Sequential(
            JPEGPointwiseConv2d(),
            nn.BatchNorm2d(4, momentum=0.01), nn.ReLU(inplace=True))

    def forward(self, bins, qtable):
        x = self.dc_layer1_tail(self.dc_layer0_dil(bins))
        b, c, h, w = x.shape
        frequencies = x.reshape(b, c, h//8, 8, w//8, 8).permute(0, 1, 3, 5, 2, 4)
        raw = frequencies.reshape(b, c*64, h//8, w//8)
        weighted = (frequencies * qtable.to(x.dtype)[:, None, :, :, None, None]).reshape_as(raw)
        return torch.cat((raw, weighted), dim=1)

    def load_pretrained(self, path):
        """Load the complete CAT-Net artifact stem; ignore its HRNet and heads."""
        # Official CAT-Net v2 includes a NumPy scalar validation score. Allow
        # only its numeric types, with both NumPy 1.x and 2.x pickle paths.
        numeric_metadata = [
            (numpy_scalar, 'numpy.core.multiarray.scalar'),
            (numpy_scalar, 'numpy._core.multiarray.scalar'),
            np.dtype, type(np.dtype('float32')), type(np.dtype('float64')),
        ]
        with torch.serialization.safe_globals(numeric_metadata):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        expected = self.state_dict()
        selected = {key: state[key] for key in expected if key in state}
        self.load_state_dict(selected, strict=True)
        logging.getLogger(__name__).info('Loaded JPEG artifact weights from %s (%d tensors)', path, len(selected))


class JPEGBranch(nn.Module):
    """Process each native frame without batch padding; align features for fusion."""

    def __init__(self, channels):
        super().__init__()
        a, b, c = channels
        self.channels_by_stride = dict(zip((8, 16, 32), channels, strict=True))
        self.artifact = JPEGArtifactModule()
        self.project = nn.Sequential(nn.Conv2d(512, a, 1, bias=False), nn.BatchNorm2d(a), nn.GELU())
        self.refine = DWSConv2d(a, a)
        self.down16 = DWSConv2d(a, b, stride=2)
        self.down32 = DWSConv2d(b, c, stride=2)
        # Native images run one at a time, so training BN is per-frame even
        # when the outer RGB batch contains multiple images.
        JPEGFrameBatchNorm2d.replace_in(self)

    @staticmethod
    def align(feature, geometry, stride, target_size, *, orientation=1, source_size=None):
        top, left, height, width, rotations, horizontal, vertical = geometry
        # Crop/scale in pixel coordinates: partial JPEG blocks must not stretch
        # their padded area across the valid image. Rotation is applied afterwards.
        out_h, out_w = target_size[::-1] if rotations % 2 else target_size
        y = top + (torch.arange(out_h, device=feature.device, dtype=torch.float32) + .5) * height / out_h
        x = left + (torch.arange(out_w, device=feature.device, dtype=torch.float32) + .5) * width / out_w
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        if orientation != 1:
            h, w = source_size
            # Inverse EXIF mapping, from displayed image to stored JPEG coordinates.
            xx, yy = {2: (w-xx, yy), 3: (w-xx, h-yy), 4: (xx, h-yy),
                      5: (yy, xx), 6: (yy, h-xx), 7: (w-yy, h-xx), 8: (w-yy, xx)}[orientation]
        yy = 2*yy/(stride*feature.shape[-2])-1
        xx = 2*xx/(stride*feature.shape[-1])-1
        grid = torch.stack((xx, yy), dim=-1)[None]
        aligned = F.grid_sample(feature.float(), grid, padding_mode='border', align_corners=False).to(feature.dtype)
        aligned = torch.rot90(aligned, rotations, dims=(-2, -1))
        if horizontal:
            aligned = aligned.flip(-1)
        if vertical:
            aligned = aligned.flip(-2)
        return aligned

    def forward(self, inputs, target_sizes):
        outputs = {stride: [] for stride in self.channels_by_stride}
        for sample in inputs:
            artifact = self.artifact(sample['bins'][None], sample['qtable'][None])
            x = self.project(artifact)
            x = self.refine(x)
            x16 = self.down16(x)
            features = {8: x, 16: x16, 32: self.down32(x16)}
            for stride, feature in features.items():
                outputs[stride].append(self.align(feature, sample['geometry'], stride, target_sizes[stride],
                                                   orientation=sample.get('orientation', 1),
                                                   source_size=sample.get('source_size')))
        return {stride: torch.cat(features) for stride, features in outputs.items()}
