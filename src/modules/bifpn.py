"""Four-level BiFPN neck; preserves EMCAD's feature shapes and channel widths.

Adapted from EfficientDet (https://arxiv.org/abs/1911.09070): fast normalized
scalar fusion, top-down and bottom-up paths, separable convolutions. This neck
uses strides 4/8/16/32, shared input projections and output projections back to
encoder widths. Repeats have independent parameters. No cross-attention.
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.decoders.layers import make_norm


class WeightedFusion(nn.Module):
    """Learn positive input weights shared across images, channels and positions."""

    def __init__(self, inputs):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(inputs))

    def forward(self, features):
        weights = self.weights.float().relu()
        weights = weights / (weights.sum() + 1e-4)
        result = features[0] * weights[0].to(features[0].dtype)
        for weight, feature in zip(weights[1:], features[1:], strict=True):
            result = result + feature * weight.to(feature.dtype)
        return result


class FusionNode(nn.Module):
    def __init__(self, channels, inputs, norm):
        super().__init__()
        self.fusion = WeightedFusion(inputs)
        self.refine = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_norm(norm, channels),
        )

    def forward(self, features):
        return self.refine(self.fusion(features))


class BiFPNLayer(nn.Module):
    """32->16->8->4, then 4->8->16->32, with same-level input connections."""

    def __init__(self, channels, norm):
        super().__init__()
        self.top_down = nn.ModuleList([FusionNode(channels, 2, norm) for _ in range(3)])
        self.bottom_up = nn.ModuleList([FusionNode(channels, n, norm) for n in (3, 3, 2)])

    def forward(self, features):
        top = list(features)
        for index in (2, 1, 0):
            context = F.interpolate(top[index + 1], size=features[index].shape[-2:], mode='nearest')
            top[index] = self.top_down[index]([features[index], context])
        result = [top[0]]
        for index in (1, 2, 3):
            context = F.max_pool2d(result[-1], 3, stride=2, padding=1)
            if context.shape[-2:] != features[index].shape[-2:]:
                raise ValueError('BiFPN expects successive ceil-halved spatial grids')
            inputs = [features[index], top[index], context] if index < 3 else [features[index], context]
            result.append(self.bottom_up[index - 1](inputs))
        return result


class BiFPN(nn.Module):
    """Mix a pyramid at common width, then restore its original channel contract."""

    def __init__(self, encoder_channels, encoder_strides, *, width=64, repeats=1, norm='batch'):
        super().__init__()
        if tuple(encoder_strides) != (4, 8, 16, 32) or len(encoder_channels) != 4:
            raise ValueError('BiFPN requires four levels at strides 4/8/16/32')
        if any(type(x) is not int or x < 1 for x in (width, repeats, *encoder_channels)):
            raise ValueError('BiFPN widths and repeats must be positive integers')
        self.input_projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, width, 1, bias=False), make_norm(norm, width))
            for c in encoder_channels
        ])
        self.layers = nn.ModuleList([BiFPNLayer(width, norm) for _ in range(repeats)])
        self.output_projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, c, 1, bias=False), make_norm(norm, c))
            for c in encoder_channels
        ])

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('BiFPN requires four input feature maps')
        mixed = [project(feature) for project, feature in zip(self.input_projections, features, strict=True)]
        for layer in self.layers:
            mixed = layer(mixed)
        return [project(feature) for project, feature in zip(self.output_projections, mixed, strict=True)]
