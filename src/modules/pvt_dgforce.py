"""Explicit PVT-v2-B2 execution with layer-level DG-Force modules."""

from functools import partial

import timm
from torch import nn

from src.modules.dgforce import CrossScaleTransfer, DFDGLevel, IntraScaleTransfer


class PVTDGForceEncoder(nn.Module):
    """PVT-v2-B2 with DFDG after every block in stages one through three."""

    EXPECTED_DEPTHS = [3, 4, 6, 3]
    EXPECTED_STRIDES = [4, 8, 16, 32]
    INTRA_PAIRS = (((0, 2),), ((0, 3),), ((0, 4), (1, 5)))

    def __init__(self, *, pretrained=True, reduction=16, attention_width=128,
                 attention_heads=4, encoder='pvt_v2_b2'):
        super().__init__()
        if encoder != 'pvt_v2_b2':
            raise ValueError("PVT-DGForce currently supports only encoder='pvt_v2_b2'")
        self.backbone = timm.create_model(encoder, pretrained=pretrained)
        self.depths = [len(stage.blocks) for stage in self.backbone.stages]
        self.strides = [item['reduction'] for item in self.backbone.feature_info]
        self.channels = [item['num_chs'] for item in self.backbone.feature_info]
        if self.depths != self.EXPECTED_DEPTHS or self.strides != self.EXPECTED_STRIDES:
            raise ValueError(
                f'unexpected {encoder} topology: depths={self.depths}, strides={self.strides}')
        self.intra_pairs = self.INTRA_PAIRS
        self.dfdg = nn.ModuleDict({
            self._key(stage, block): DFDGLevel(self.channels[stage - 1], reduction)
            for stage, depth in enumerate(self.depths[:3], start=1)
            for block in range(depth)
        })
        self.intra_transfers = nn.ModuleDict({
            self._transfer_key(stage, shallow, deep, cue): IntraScaleTransfer(
                self.channels[stage - 1])
            for stage, pairs in enumerate(self.intra_pairs, start=1)
            for shallow, deep in pairs
            for cue in ('patch', 'edge')
        })
        self.cross_transfers = nn.ModuleDict({
            f's{stage}_{cue}': CrossScaleTransfer(
                self.channels[stage - 1], self.channels[stage - 2],
                attention_width, attention_heads)
            for stage in (2, 3)
            for cue in ('patch', 'edge')
        })

    @staticmethod
    def _key(stage, block):
        return f's{stage}_b{block}'

    @staticmethod
    def _transfer_key(stage, shallow, deep, cue):
        return f's{stage}_b{shallow}_to_b{deep}_{cue}'

    def _cue_transfer(self, stage, block, cue, saved, previous):
        transfers = []
        if block == 0 and stage > 1:
            transfers.append(partial(self.cross_transfers[f's{stage}_{cue}'],
                                     source=previous[cue]))
        for shallow, deep in self.intra_pairs[stage - 1]:
            if block == deep:
                module = self.intra_transfers[
                    self._transfer_key(stage, shallow, deep, cue)]
                transfers.append(partial(module, shallow=saved[(shallow, cue)]))
        if not transfers:
            return None

        def apply(current):
            for transfer in transfers:
                current = transfer(current)
            return current
        return apply

    def forward(self, image, *, supervise):
        x = self.backbone.patch_embed(image)
        features, patch_logits, edge_logits = [], {}, {}
        previous = None
        for stage_index, stage_module in enumerate(self.backbone.stages, start=1):
            if stage_module.downsample is not None:
                x = stage_module.downsample(x)
            batch, height, width, channels = x.shape
            size = (height, width)
            tokens = x.reshape(batch, -1, channels)
            saved = {}
            last_cues = None
            for block_index, block in enumerate(stage_module.blocks):
                tokens = block(tokens, size)
                if stage_index <= 3:
                    feature_map = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
                    key = self._key(stage_index, block_index)
                    result = self.dfdg[key](
                        feature_map,
                        self._cue_transfer(stage_index, block_index, 'patch', saved, previous),
                        self._cue_transfer(stage_index, block_index, 'edge', saved, previous),
                        supervise=supervise,
                    )
                    feature_map, patch, edge, patch_logit, edge_logit, _ = result
                    tokens = feature_map.permute(0, 2, 3, 1).reshape(batch, -1, channels)
                    for shallow, _ in self.intra_pairs[stage_index - 1]:
                        if block_index == shallow:
                            saved[(shallow, 'patch')] = patch
                            saved[(shallow, 'edge')] = edge
                    if supervise:
                        patch_logits[key], edge_logits[key] = patch_logit, edge_logit
                    last_cues = {'patch': patch, 'edge': edge}
            tokens = stage_module.norm(tokens)
            x = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
            features.append(x)
            if stage_index <= 3:
                previous = last_cues
        return features, patch_logits, edge_logits
