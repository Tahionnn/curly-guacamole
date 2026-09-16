import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def soft_dice_loss(logits, targets, smooth=1.0, valid_mask=None):
    probs = torch.sigmoid(logits.float()).flatten(1)
    targets = targets.float().flatten(1)
    inter = probs * targets
    if valid_mask is not None:
        valid = valid_mask.float().flatten(1)
        inter = inter * valid
        probs = probs * valid
        targets = targets * valid
    inter = inter.sum(1)
    return (1.0 - (2.0 * inter + smooth) / (probs.sum(1) + targets.sum(1) + smooth)).mean()


def bce_loss(logits, targets, valid_mask=None):
    if valid_mask is not None:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted = (loss * valid_mask).flatten(1).sum(1)
        return (weighted / valid_mask.flatten(1).sum(1).clamp_min(1)).mean()
    return F.binary_cross_entropy_with_logits(logits, targets)


def coarse_target(target, size):
    """Soft occupancy of the target inside each cell of a coarser grid."""
    return F.interpolate(target, size=size, mode='area')


def edge_band_target(occupancy, width):
    """Morphological gradient of the occupied cells: dilation minus erosion.

    Both are max pooling, so the band needs no loader work and follows whatever
    geometry the batch already has.
    """
    occupied = (occupancy > 0).float()
    dilated = F.max_pool2d(occupied, width, stride=1, padding=width // 2)
    eroded = -F.max_pool2d(-occupied, width, stride=1, padding=width // 2)
    return dilated - eroded


def edge_loss(logits, band, max_pos_weight):
    """Balance sparse boundary positives, capping their weight for stability."""
    positive = band.sum()
    pos_weight = ((band.numel() - positive) / positive.clamp_min(1)).clamp(1, max_pos_weight)
    return F.binary_cross_entropy_with_logits(logits, band, pos_weight=pos_weight)


@dataclass(frozen=True)
class LossResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, tuple[torch.Tensor, torch.Tensor]]


class BoundaryBandLoss(torch.nn.Module):
    """Per-image BCE in a two-sided square band on the final target grid.

    Only band construction binarizes soft targets. Pooling ignores the frame
    exterior, so uniform masks have no contour and contribute zero. Average
    over all images, including zero contributions from empty bands.
    """

    def __init__(self, radius=4):
        super().__init__()
        if type(radius) is not int or radius < 1:
            raise ValueError('boundary_radius must be a positive integer')
        self.radius = radius

    def forward(self, logits, target):
        foreground = (target > .5).float()
        width = 2 * self.radius + 1
        dilated = F.max_pool2d(foreground, width, stride=1, padding=self.radius)
        eroded = -F.max_pool2d(-foreground, width, stride=1, padding=self.radius)
        return bce_loss(logits.float(), target.float(), valid_mask=dilated - eroded)


class SegmentationLoss(torch.nn.Module):
    """BCE + all-image Dice, classifier BCE and decoder auxiliary supervision."""

    def __init__(self, *, dice_weight=1.0, aux_weight=.4, patch_weight=0., edge_weight=0.,
                 edge_band=3, edge_max_pos_weight=50., reference_weight=0.,
                 boundary_weight=0., boundary_radius=4):
        super().__init__()
        for value in (dice_weight, aux_weight, patch_weight, edge_weight, reference_weight, boundary_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Loss weights must be finite and nonnegative')
        if type(edge_band) is not int or edge_band < 1 or not edge_band % 2:
            raise ValueError('edge_band must be a positive odd integer')
        if not math.isfinite(edge_max_pos_weight) or edge_max_pos_weight < 1:
            raise ValueError('edge_max_pos_weight must be finite and at least 1')
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.patch_weight = patch_weight
        self.edge_weight = edge_weight
        self.edge_band = edge_band
        self.edge_max_pos_weight = edge_max_pos_weight
        self.reference_weight = reference_weight
        self.boundary_weight = boundary_weight
        self.boundary_loss = BoundaryBandLoss(boundary_radius)

    @staticmethod
    def _dice(logits, target):
        probs = logits.float().sigmoid().flatten(1)
        target = target.float().flatten(1)
        positive = target.sum(1) > 0
        losses = 1 - (2 * (probs * target).sum(1) + 1) / (probs.sum(1) + target.sum(1) + 1)
        return losses.mean(), losses, positive

    def forward(self, out, batch):
        target = batch['mask'].float()
        dice, per_image, positive = self._dice(out['logits'], target)
        labels = batch.get('label')
        if labels is None:
            labels = positive.float().reshape_as(out['cls_logits'])
        components = {
            'bce': bce_loss(out['logits'].float(), target),
            'dice': self.dice_weight * dice,
            'cls': .3 * bce_loss(out['cls_logits'].float(), labels.float()),
        }
        if self.boundary_weight > 0:
            components['boundary_bce'] = self.boundary_weight * self.boundary_loss(out['logits'], target)
        if self.aux_weight > 0 and 'aux_logits' in out:
            logits = out['aux_logits'].float()
            components['aux_bce'] = self.aux_weight * bce_loss(logits, target)
            components['aux_dice'] = self.aux_weight * self.dice_weight * self._dice(logits, target)[0]
        if self.patch_weight > 0 and out.get('patch_logits'):
            pixel, overlap = [], []
            for logits in out['patch_logits'].values():
                logits = logits.float()
                occupancy = coarse_target(target, logits.shape[-2:])
                pixel.append(bce_loss(logits, occupancy))
                overlap.append(self._dice(logits, occupancy)[0])
            components['patch_bce'] = self.patch_weight * torch.stack(pixel).mean()
            components['patch_dice'] = self.patch_weight * self.dice_weight * torch.stack(overlap).mean()
        if self.edge_weight > 0 and out.get('edge_logits'):
            band = []
            for logits in out['edge_logits'].values():
                logits = logits.float()
                occupancy = coarse_target(target, logits.shape[-2:])
                band.append(edge_loss(logits, edge_band_target(occupancy, self.edge_band),
                                      self.edge_max_pos_weight))
            components['edge_bce'] = self.edge_weight * torch.stack(band).mean()
        if self.training and self.reference_weight > 0:
            if 'reference' not in out:
                raise ValueError('reference_weight requires training output from a pristine reference head')
            from src.modules.pristine_reference import PristineReferenceHead
            terms = PristineReferenceHead.supervision(out['reference'], target, out['reference']['available'])
            components.update({key: self.reference_weight * value for key, value in terms.items()})
        diagnostics = {'dice_pos': ((per_image * positive).sum(), positive.sum()),
                       'dice_neg': ((per_image * ~positive).sum(), (~positive).sum())}
        return LossResult(sum(components.values()), components, diagnostics)


class LossMeter:
    """Epoch means; conditional Dice diagnostics use actual group counts.

    The objective and its contributions are averaged by batch image count,
    matching train/loss. Missing diagnostic groups are omitted, not logged as 0.
    Detached sums remain on device until compute(), avoiding per-term GPU sync.
    """

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, result: LossResult, batch_size: int):
        for key, value in {"total": result.total, **result.components}.items():
            self._add(key, value.detach() * batch_size, batch_size)
        for key, (value, count) in result.diagnostics.items():
            self._add(key, value.detach(), count.detach())

    def _add(self, key, value, count):
        self.sums[key] = self.sums.get(key, 0) + value
        self.counts[key] = self.counts.get(key, 0) + count

    def compute(self):
        result = {key: float(value / self.counts[key]) for key, value in self.sums.items()
                  if float(self.counts[key]) > 0}
        return result


def compute_loss(out, batch, aux_weight: float = 0.0):
    """Scalar convenience API for the baseline objective."""
    return SegmentationLoss(aux_weight=aux_weight)(out, batch).total
