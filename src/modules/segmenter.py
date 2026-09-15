import torch.nn.functional as F
from torch import nn

from src.decoders import EMCADDecoder
from src.modules.forensic_disentangle import ForensicDisentangle
from src.modules.forensic_fusion import ForensicFusion
from src.modules.gate_head import GateHead
from src.modules.input_normalization import ImageNetInputNormalization
from src.modules.utils import build_timm_encoder


class Segmenter(nn.Module):
    """PVT image encoder with native JPEG fusion and an EMCAD mask decoder."""

    def __init__(self, encoder="pvt_v2_b2", jpeg_channels=(64, 96, 128),
                 aux_weight=0.4, *, pretrained=True, jpeg_similarity=False,
                 disentangle_levels=(), disentangle_mode='fuse', disentangle_reduction=16,
                 disentangle_cross_strides=(), disentangle_attention_width=128,
                 disentangle_attention_heads=4):
        super().__init__()
        # Construction order is part of reproducible baseline initialization.
        self.encoder, self.strides, self.channels = build_timm_encoder(
            encoder, pretrained=pretrained)
        self.forensic_fusion = ForensicFusion(self.strides, self.channels, jpeg_channels)
        self.decoder = EMCADDecoder(self.channels, self.strides, use_aux=aux_weight > 0)
        self.segmentation_head = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)
        self.classification_head = GateHead(self.channels[-1])
        self.aux_weight = aux_weight
        self.input_normalization = ImageNetInputNormalization()
        # Construct after baseline modules to preserve their seeded initialization.
        if jpeg_similarity:
            from src.modules.jpeg_similarity import JPEGSimilarity
            self.forensic_fusion.similarity = JPEGSimilarity(jpeg_channels[-1])
        self.disentangle = ForensicDisentangle(
            self.strides, self.channels, disentangle_levels,
            reduction=disentangle_reduction, mode=disentangle_mode,
            cross_strides=disentangle_cross_strides,
            attention_width=disentangle_attention_width,
            attention_heads=disentangle_attention_heads,
        ) if disentangle_levels else None

    def disentangle_gate_stats(self) -> dict[str, float]:
        """Detached residual-gate statistics, zero when the module is disabled."""
        return self.disentangle.gate_stats() if self.disentangle is not None else {'max_abs': 0.0}

    def forensic_gate_stats(self) -> dict[str, float]:
        """Detached channel-gate statistics for training logs."""
        return self.forensic_fusion.gate_stats()

    def forward(self, image, *, jpeg):
        if not isinstance(jpeg, (list, tuple)) or len(jpeg) != image.shape[0]:
            raise ValueError('jpeg inputs must contain one native frame per image')
        input_size = image.shape[-2:]
        image = self.input_normalization(image)
        encoder_features = self.forensic_fusion(list(self.encoder(image)), jpeg=jpeg)
        patch_logits, edge_logits = {}, {}
        if self.disentangle is not None:
            encoder_features, patch_logits, edge_logits = self.disentangle(
                encoder_features, supervise=self.training)
        decoder_features, aux_logits = self.decoder(encoder_features)
        result = {
            "logits": self._resize(self.segmentation_head(decoder_features), input_size),
            "cls_logits": self.classification_head(encoder_features[-1]),
        }
        if self.training and aux_logits is not None:
            result["aux_logits"] = self._resize(aux_logits, input_size)
        if patch_logits:
            result['patch_logits'] = patch_logits
        if edge_logits:
            result['edge_logits'] = edge_logits
        return result

    @staticmethod
    def _resize(features, size):
        if features.shape[-2:] == size:
            return features
        return F.interpolate(features, size=size, mode="bilinear", align_corners=False)
