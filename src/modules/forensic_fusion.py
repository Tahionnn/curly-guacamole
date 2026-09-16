from torch import nn

from src.modules.gated_fuse import GatedFuse
from src.modules.jpeg_branch import JPEGBranch


class ForensicFusion(nn.Module):
    """Align native JPEG features and add local residuals to image features."""

    FUSION_STRIDES = (8, 16, 32)

    def __init__(self, encoder_strides, encoder_channels, jpeg_channels):
        super().__init__()
        if len(encoder_strides) != len(encoder_channels):
            raise ValueError("encoder strides and channels must have the same length")
        if len(set(encoder_strides)) != len(encoder_strides):
            raise ValueError("encoder strides must be unique")
        if len(jpeg_channels) != len(self.FUSION_STRIDES):
            raise ValueError("expected three JPEG channel groups")
        missing = [stride for stride in self.FUSION_STRIDES if stride not in encoder_strides]
        if missing:
            raise ValueError(f"encoder is missing fusion strides: {missing}")
        self.encoder_strides = encoder_strides
        self.fusion_strides = self.FUSION_STRIDES
        self.branch = JPEGBranch(jpeg_channels)
        self.fusion_blocks = nn.ModuleDict({
            str(stride): GatedFuse(encoder_channels[encoder_strides.index(stride)],
                                   self.branch.channels_by_stride[stride])
            for stride in self.fusion_strides
        })

    def gate_stats(self) -> dict[str, float]:
        return {"max_abs": max(float(block.channel_gate.detach().abs().max())
                               for block in self.fusion_blocks.values())}

    def forward(self, encoder_features, *, jpeg, return_jpeg8=False):
        available = [i for i, sample in enumerate(jpeg) if sample.get('available', True)]
        if not available:
            return (encoder_features, None) if return_jpeg8 else encoder_features
        if len(available) != len(jpeg):
            # PNG samples bypass both the JPEG branch and fusion.
            subset = [feature[available] for feature in encoder_features]
            updated = self.forward(subset, jpeg=[jpeg[i] for i in available], return_jpeg8=return_jpeg8)
            if return_jpeg8:
                updated, subset_jpeg8 = updated
            result = [feature.clone() for feature in encoder_features]
            for original, fused in zip(result, updated, strict=True):
                original[available] = fused
            if return_jpeg8:
                jpeg8 = subset_jpeg8.new_zeros(len(jpeg), *subset_jpeg8.shape[1:])
                jpeg8[available] = subset_jpeg8
                return result, jpeg8
            return result
        sizes = {s: encoder_features[self.encoder_strides.index(s)].shape[-2:]
                 for s in self.fusion_strides}
        jpeg_features = self.branch(jpeg, sizes)
        if hasattr(self, 'similarity'):
            jpeg_features[32] = self.similarity(jpeg_features[32])
        for stride in self.fusion_strides:
            index = self.encoder_strides.index(stride)
            encoder_features[index] = self.fusion_blocks[str(stride)](
                encoder_features[index], jpeg_features[stride])
        return (encoder_features, jpeg_features[8]) if return_jpeg8 else encoder_features
