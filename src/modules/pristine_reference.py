"""Learn within-image pristine references without GT inputs or pairwise attention."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class PristineReferenceHead(nn.Module):
    def __init__(self, channels: int, *, width: int = 16, references: int = 4):
        super().__init__()
        if references != 4:
            raise ValueError('pristine references currently uses four spatially seeded slots')
        self.embedding = nn.Sequential(nn.Conv2d(channels, width, 1), nn.GELU())
        self.authenticity = nn.Conv2d(width, 1, 1)
        self.selection = nn.Conv2d(width, references, 1)
        self.correction = nn.Sequential(nn.Conv2d(width + references + 1, width, 1),
                                        nn.GELU(), nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.register_buffer('centers', torch.tensor([[-.5, -.5], [-.5, .5],
                                                      [.5, -.5], [.5, .5]]), persistent=False)

    def forward(self, features):
        embedding = self.embedding(features)
        batch, _, height, width = embedding.shape
        authentic = self.authenticity(embedding).float()
        y = torch.linspace(-1, 1, height, device=features.device)
        x = torch.linspace(-1, 1, width, device=features.device)
        # Soft spatial seeds break slot symmetry without restricting a slot to a region.
        prior = -((y[None, :, None] - self.centers[:, 0, None, None]).square()
                  + (x[None, None, :] - self.centers[:, 1, None, None]).square())
        scores = self.selection(embedding).float() + prior[None] + F.logsigmoid(authentic)
        weights = scores.flatten(2).softmax(-1)
        # Explicit FP32 reductions also under AMP; no N x N attention matrix.
        precision = (nullcontext() if features.device.type == 'meta'
                     else torch.autocast(device_type=features.device.type, enabled=False))
        with precision:
            tokens = F.normalize(embedding.float().flatten(2), dim=1, eps=1e-6)
            references = F.normalize(weights @ tokens.transpose(1, 2), dim=-1, eps=1e-6)
            similarities = (references @ tokens).reshape(batch, 4, height, width)
            support = (weights * authentic.sigmoid().flatten(2)).sum(-1).mean(1)[:, None, None, None]
        inputs = torch.cat((embedding, similarities.to(embedding.dtype),
                            authentic.sigmoid().to(embedding.dtype)), dim=1)
        return {'correction': self.correction(inputs) * support,
                'authenticity_logits': authentic,
                'reference_weights': weights.reshape(batch, 4, height, width),
                'similarities': similarities}

    @staticmethod
    def supervision(output, target, available):
        """Supervise selection only in the loss; skip reference terms without clean cells."""
        logits = output['authenticity_logits'].float()
        occupancy = F.adaptive_avg_pool2d(target.float(), logits.shape[-2:])
        valid = available.to(device=logits.device, dtype=torch.float32)
        clean_valid = valid * (occupancy.flatten(1).amin(1) < 1e-6)

        def average(values, mask):
            return (values * mask).sum() / mask.sum().clamp_min(1)

        bce = F.binary_cross_entropy_with_logits(logits, 1 - occupancy, reduction='none')
        weights = output['reference_weights'].float().flatten(2)
        clean_mass = (weights * (1 - occupancy.flatten(2))).sum(-1)
        contamination = -clean_mass.clamp_min(1e-6).log().mean(1)
        normalized = F.normalize(weights, dim=-1, eps=1e-6)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            overlaps = normalized @ normalized.transpose(1, 2)
        slots = weights.shape[1]
        off_diagonal = 1 - torch.eye(slots, device=logits.device)
        diversity = (overlaps * off_diagonal).sum((1, 2)) / (slots * (slots - 1))
        return {'reference_authenticity': average(bce.flatten(1).mean(1), valid),
                'reference_contamination': average(contamination, clean_valid),
                'reference_diversity': .1 * average(diversity, clean_valid)}
