"""Independent PVT-DGForce experiment model."""

import torch.nn.functional as F
from torch import nn

from src.decoders import EMCADDecoder
from src.modules.forensic_fusion import ForensicFusion
from src.modules.gate_head import GateHead
from src.modules.input_normalization import ImageNetInputNormalization
from src.modules.pvt_dgforce import PVTDGForceEncoder


class PVTDGForceSegmenter(nn.Module):
    """PVT-v2-B2 with layer-level DFDG/MFT and the project decoder stack."""

    def __init__(self, encoder='pvt_v2_b2', jpeg_channels=(64, 96, 128),
                 aux_weight=0.0, *, pretrained=True, reduction=16,
                 attention_width=128, attention_heads=4):
        super().__init__()
        self.encoder = PVTDGForceEncoder(
            encoder=encoder, pretrained=pretrained, reduction=reduction,
            attention_width=attention_width, attention_heads=attention_heads)
        self.strides, self.channels = self.encoder.strides, self.encoder.channels
        self.forensic_fusion = ForensicFusion(self.strides, self.channels, jpeg_channels)
        self.decoder = EMCADDecoder(self.channels, self.strides, use_aux=aux_weight > 0)
        self.segmentation_head = nn.Conv2d(self.decoder.out_channels, 1, 1)
        self.classification_head = GateHead(self.channels[-1])
        self.input_normalization = ImageNetInputNormalization()
        self.aux_weight = aux_weight

    def forensic_gate_stats(self):
        return self.forensic_fusion.gate_stats()

    @staticmethod
    def disentangle_gate_stats():
        return {'max_abs': 0.0}

    def forward(self, image, *, jpeg):
        if not isinstance(jpeg, (list, tuple)) or len(jpeg) != image.shape[0]:
            raise ValueError('jpeg inputs must contain one native frame per image')
        input_size = image.shape[-2:]
        image = self.input_normalization(image)
        features, patch_logits, edge_logits = self.encoder(image, supervise=self.training)
        features = self.forensic_fusion(features, jpeg=jpeg)
        decoded, aux_logits = self.decoder(features)
        result = {
            'logits': F.interpolate(self.segmentation_head(decoded), input_size,
                                    mode='bilinear', align_corners=False),
            'cls_logits': self.classification_head(features[-1]),
        }
        if self.training and aux_logits is not None:
            result['aux_logits'] = F.interpolate(
                aux_logits, input_size, mode='bilinear', align_corners=False)
        if patch_logits:
            result['patch_logits'] = patch_logits
            result['edge_logits'] = edge_logits
        return result
