import math

import torch
import torch.nn.functional as F
from torch import nn


class JPEGSimilarity(nn.Module):
    """Add a coarse within-image similarity cue to stride-32 JPEG features.

    Inspired by IIS; uses the existing forensic encoder instead of a Laplacian
    and GFNet. Aggregate dot products before softmax, as in the authors' code.
    The zero gate preserves baseline behavior at initialization.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.embedding = nn.Conv2d(channels, 64, 1)
        self.output = nn.Conv2d(1, channels, 1, bias=False)
        self.channel_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def patch_scores(tokens):
        # sum_j(z_i dot z_j), excluding j=i, without allocating an N x N matrix.
        # Keep reductions in fp32 under AMP.
        tokens = tokens.float()
        totals = (tokens * (tokens.sum(dim=-1, keepdim=True) - tokens)).sum(dim=1)
        scores = (totals / math.sqrt(tokens.shape[1])).softmax(dim=-1)
        # A uniform image yields zero; scale does not shrink with patch count.
        return scores * tokens.shape[-1] - 1

    def forward(self, features):
        size = tuple(min(20, value) for value in features.shape[-2:])
        pooled = F.adaptive_avg_pool2d(features, size)
        tokens = self.embedding(pooled).flatten(2)
        scores = self.patch_scores(tokens).reshape(features.shape[0], 1, *size)
        scores = F.interpolate(scores, features.shape[-2:], mode='bilinear', align_corners=False)
        residual = self.output(scores.to(features.dtype))
        return features + self.channel_gate * residual
