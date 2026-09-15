"""DG-Force-inspired patch/edge supervision and gated pyramid fusion.

Each selected encoder stage has a channel bottleneck for region cues and a
spatial depthwise-separable bottleneck for boundary cues. Training-only heads
supervise both. Fuse mode adds their spatially balanced residual through a
zero-initialized channel gate. Optional reduced-width cross-attention transfers
features from the next finer stage. Rectangular grids are supported explicitly.

Reference: Yao et al., DG-Force, ECCV 2026, pp. 147-164.
https://doi.org/10.1007/978-3-032-37432-5_9
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.dws_conv2d import DWSConv2d

MODES = ('supervision', 'fuse')


def bottleneck_width(channels: int, reduction: int) -> int:
    width = channels // reduction
    if width < 1:
        raise ValueError(f'reduction {reduction} leaves an empty bottleneck for {channels} channels')
    return width


class PatchDisentangle(nn.Module):
    """Channel bottleneck; the compressed tensor carries the patch-level cue."""

    def __init__(self, channels: int, reduction: int, *, inject: bool):
        super().__init__()
        width = bottleneck_width(channels, reduction)
        self.down = nn.Sequential(nn.Conv2d(channels, width, 1), nn.GELU())
        self.up = nn.Conv2d(width, channels, 1) if inject else None

    def forward(self, x):
        compressed = self.down(x)
        return (self.up(compressed) if self.up is not None else None), compressed


class EdgeDisentangle(nn.Module):
    """The same bottleneck with spatial support, kept depthwise-separable."""

    def __init__(self, channels: int, reduction: int, *, inject: bool, norm: str = 'batch'):
        super().__init__()
        width = bottleneck_width(channels, reduction)
        self.down = DWSConv2d(channels, width, norm=norm)
        # The projection back has no activation: the result is a residual and
        # must keep both signs, unlike the reference's activated bottleneck.
        self.up = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.Conv2d(width, channels, 1),
        ) if inject else None

    def forward(self, x):
        compressed = self.down(x)
        return (self.up(compressed) if self.up is not None else None), compressed


class DisentangleLevel(nn.Module):
    """One pyramid level: both cues, their per-token balance, one gate.

    In 'supervision' mode only the bottlenecks and their heads exist, and the
    heads run in training alone - the level then costs zero GFLOPs at inference.
    """

    def __init__(self, channels: int, reduction: int, *, inject: bool, norm: str = 'batch'):
        super().__init__()
        self.inject = inject
        self.patch = PatchDisentangle(channels, reduction, inject=inject)
        self.edge = EdgeDisentangle(channels, reduction, inject=inject, norm=norm)
        width = bottleneck_width(channels, reduction)
        self.patch_head = nn.Conv2d(width, 1, 1)
        self.edge_head = nn.Conv2d(width, 1, 1)
        if inject:
            self.balance = nn.Conv2d(channels, 2, 1)
            self.channel_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, *, supervise: bool):
        if not self.inject and not supervise:
            # Supervision mode outside training: nothing here reaches the output.
            return x, (None, None)
        patch, patch_compressed = self.patch(x)
        edge, edge_compressed = self.edge(x)
        logits = ((self.patch_head(patch_compressed), self.edge_head(edge_compressed))
                  if supervise else (None, None))
        if not self.inject:
            return x, logits
        weights = self.balance(x).softmax(dim=1)
        residual = weights[:, :1] * patch + weights[:, 1:] * edge
        return x + self.channel_gate * residual, logits


class CrossScaleFuse(nn.Module):
    """The coarse level attends to the finer one at reduced width (MFT-cross).

    Direction follows the reference: the query is the coarser stage and the keys
    come from the finer stage, downsampled onto the query grid.
    """

    def __init__(self, coarse_channels: int, fine_channels: int, width: int, heads: int,
                 *, pool_context: bool = True):
        super().__init__()
        self.pool_context = pool_context
        if width % heads:
            raise ValueError('attention width must be divisible by the head count')
        self.query = nn.Conv2d(coarse_channels, width, 1)
        self.align = nn.Conv2d(fine_channels, width, 1)
        # LayerNorm before the projections keeps q @ k.T inside fp16 range; the
        # reference needed the same guard and this project lost a run to it.
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Conv2d(width, coarse_channels, 1)
        self.channel_gate = nn.Parameter(torch.zeros(1, coarse_channels, 1, 1))

    def forward(self, coarse, fine):
        batch, _, height, width = coarse.shape
        context = self.align(F.adaptive_avg_pool2d(fine, (height, width))
                             if self.pool_context else fine)
        query = self.query_norm(self.query(coarse).flatten(2).transpose(1, 2))
        keys = self.context_norm(context.flatten(2).transpose(1, 2))
        attended, _ = self.attention(query, keys, keys, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, -1, height, width)
        return coarse + self.channel_gate * self.output(attended)


class ForensicDisentangle(nn.Module):
    """Applies the disentangling levels, and cross-scale fusion where configured."""

    def __init__(self, encoder_strides, encoder_channels, levels, *, reduction=16,
                 mode='fuse', cross_strides=(), attention_width=128, attention_heads=4,
                 norm='batch', return_to_stride4=False, parallel_16_32=False):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'disentangle mode must be one of {MODES}')
        levels, cross_strides = tuple(levels), tuple(cross_strides)
        if not levels:
            raise ValueError('disentangle requires at least one level')
        if len(set(levels)) != len(levels):
            raise ValueError('disentangle levels must be unique')
        missing = [stride for stride in levels if stride not in encoder_strides]
        if missing:
            raise ValueError(f'encoder is missing disentangle strides: {missing}')
        if attention_width % attention_heads:
            raise ValueError('attention width must be divisible by the head count')
        if cross_strides and mode != 'fuse':
            raise ValueError('cross-scale fusion requires disentangle mode=fuse')
        for stride in cross_strides:
            if stride not in levels:
                raise ValueError(f'cross stride {stride} must also be a disentangle level')
            if stride // 2 not in encoder_strides:
                raise ValueError(f'cross stride {stride} requires encoder stride {stride // 2}')
        if return_to_stride4 and (mode != 'fuse' or not {4, 32}.issubset(levels)):
            raise ValueError('return_to_stride4 requires fuse mode and levels 4 and 32')

        if parallel_16_32 and (mode != 'fuse' or cross_strides != (32,)
                               or not {16, 32}.issubset(levels)):
            raise ValueError('parallel_16_32 requires fuse, levels 16/32 and cross_strides [32]')
        self.encoder_strides = list(encoder_strides)
        self.levels = tuple(sorted(levels))
        self.cross_strides = tuple(sorted(cross_strides))
        self.mode = mode
        inject = mode == 'fuse'
        self.blocks = nn.ModuleDict({
            str(stride): DisentangleLevel(
                encoder_channels[self.encoder_strides.index(stride)], reduction,
                inject=inject, norm=norm)
            for stride in self.levels
        })
        self.cross = nn.ModuleDict({
            str(stride): CrossScaleFuse(
                encoder_channels[self.encoder_strides.index(stride)],
                encoder_channels[self.encoder_strides.index(stride // 2)],
                attention_width, attention_heads)
            for stride in self.cross_strides
        })
        self.return_to_stride4 = CrossScaleFuse(
            encoder_channels[self.encoder_strides.index(4)],
            encoder_channels[self.encoder_strides.index(32)],
            attention_width, attention_heads, pool_context=False,
        ) if return_to_stride4 else None

        self.parallel_return16 = CrossScaleFuse(
            encoder_channels[self.encoder_strides.index(16)],
            encoder_channels[self.encoder_strides.index(32)],
            attention_width, attention_heads, pool_context=False,
        ) if parallel_16_32 else None

    def gate_stats(self) -> dict[str, float]:
        gates = [block.channel_gate for block in self.blocks.values() if block.inject]
        gates += [block.channel_gate for block in self.cross.values()]
        if self.parallel_return16 is not None:
            gates.append(self.parallel_return16.channel_gate)
        if self.return_to_stride4 is not None:
            gates.append(self.return_to_stride4.channel_gate)
        return {'max_abs': max((float(gate.detach().abs().max()) for gate in gates), default=0.0)}

    def forward(self, features, *, supervise: bool):
        """Returns the updated pyramid plus per-stride patch and edge logits."""
        features = list(features)
        patch_logits, edge_logits = {}, {}
        for stride in self.levels:
            index = self.encoder_strides.index(stride)
            features[index], (patch, edge) = self.blocks[str(stride)](
                features[index], supervise=supervise)
            if supervise:
                patch_logits[stride], edge_logits[stride] = patch, edge
            if str(stride) in self.cross and self.parallel_return16 is None:
                fine = features[self.encoder_strides.index(stride // 2)]
                features[index] = self.cross[str(stride)](features[index], fine)
        if self.parallel_return16 is not None:
            i16, i32 = (self.encoder_strides.index(s) for s in (16, 32))
            # Both directions read the same post-DG, pre-attention snapshot.
            f16, f32 = features[i16], features[i32]
            features[i32] = self.cross['32'](f32, f16)
            features[i16] = self.parallel_return16(f16, f32)
        # Read the final coarse state, after all ascending cross-scale updates.
        if self.return_to_stride4 is not None:
            fine_index = self.encoder_strides.index(4)
            coarse_index = self.encoder_strides.index(32)
            features[fine_index] = self.return_to_stride4(
                features[fine_index], features[coarse_index])
        return features, patch_logits, edge_logits
