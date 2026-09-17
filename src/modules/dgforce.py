"""Layer-level DG-Force building blocks for the PVT experiment."""

import torch
from torch import nn
from torch.nn import functional as F


def _bottleneck_width(channels: int, reduction: int) -> int:
    if type(reduction) is not int or reduction < 1:
        raise ValueError('reduction must be a positive integer')
    width = channels // reduction
    if width < 1:
        raise ValueError(f'reduction {reduction} leaves an empty bottleneck for {channels} channels')
    return width


class PatchForensicDisentangle(nn.Module):
    """Per-position MLP bottleneck with no spatial mixing."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        width = _bottleneck_width(channels, reduction)
        self.down = nn.Sequential(nn.Conv2d(channels, width, 1), nn.GELU())
        self.up = nn.Sequential(nn.Conv2d(width, channels, 1), nn.GELU())

    def forward(self, x):
        compressed = self.down(x)
        return self.up(compressed), compressed


class EdgeForensicDisentangle(nn.Module):
    """Spatial convolution bottleneck for boundary-sensitive cues."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        width = _bottleneck_width(channels, reduction)
        self.down = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.Conv2d(width, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        compressed = self.down(x)
        return self.up(compressed), compressed


class DFDGLevel(nn.Module):
    """Disentangle patch/edge cues and gather them with token-wise weights."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        width = _bottleneck_width(channels, reduction)
        self.patch = PatchForensicDisentangle(channels, reduction)
        self.edge = EdgeForensicDisentangle(channels, reduction)
        self.patch_head = nn.Conv2d(width, 1, 1)
        self.edge_head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(width, 1, 1))
        self.evaluator = nn.Sequential(
            nn.Conv2d(channels, width, 1), nn.GELU(), nn.Conv2d(width, 2, 1))

    def disentangle(self, x, *, supervise=True):
        patch, patch_compressed = self.patch(x)
        edge, edge_compressed = self.edge(x)
        logits = ((self.patch_head(patch_compressed), self.edge_head(edge_compressed))
                  if supervise else (None, None))
        return patch, edge, *logits

    def forward(self, x, patch_transfer=None, edge_transfer=None, *, supervise=True):
        patch, edge, patch_logits, edge_logits = self.disentangle(x, supervise=supervise)
        if patch_transfer is not None:
            patch = patch_transfer(patch)
        if edge_transfer is not None:
            edge = edge_transfer(edge)
        weights = self.evaluator(x).softmax(dim=1)
        enriched = x + weights[:, :1] * patch + weights[:, 1:] * edge
        return enriched, patch, edge, patch_logits, edge_logits, weights


class IntraScaleTransfer(nn.Module):
    """Gated residual transfer between shallow and deep cues at one scale."""

    def __init__(self, channels: int):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, padding=1), nn.GELU())
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, current, shallow):
        if current.shape != shallow.shape:
            raise ValueError('intra-scale cues must have identical shapes')
        update = self.fuse(torch.cat((self.project(shallow), current), dim=1))
        return current + self.gate * update


class CrossScaleTransfer(nn.Module):
    """Fine-to-coarse cross-attention followed by a gated residual."""

    def __init__(self, target_channels: int, source_channels: int, width: int, heads: int):
        super().__init__()
        if type(width) is not int or width < 1 or type(heads) is not int or heads < 1:
            raise ValueError('attention width and heads must be positive integers')
        if width % heads:
            raise ValueError('attention width must be divisible by the head count')
        self.query = nn.Conv2d(target_channels, width, 1)
        self.source = nn.Sequential(nn.Conv2d(source_channels, width, 3, padding=1), nn.GELU())
        self.query_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Conv2d(width, target_channels, 1)
        self.gate = nn.Parameter(torch.zeros(1, target_channels, 1, 1))

    def forward(self, target, source):
        batch, _, height, width = target.shape
        source = F.adaptive_avg_pool2d(source, (height, width))
        query = self.query_norm(self.query(target).flatten(2).transpose(1, 2))
        context = self.source_norm(self.source(source).flatten(2).transpose(1, 2))
        update, _ = self.attention(query, context, context, need_weights=False)
        update = update.transpose(1, 2).reshape(batch, -1, height, width)
        return target + self.gate * self.output(update)
