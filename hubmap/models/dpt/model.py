"""
Vision Transformers for Dense Prediction
========================================

Authors: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
Paper: https://arxiv.org/abs/2103.13413
Code: https://github.com/isl-org/DPT/tree/main
"""
import torch
import torch.nn as nn

from hubmap.models.dpt.encoder import make_encoder
from hubmap.models.dpt.encoder import Backbone
from hubmap.models.dpt.fusion_block import make_fusion_block
from hubmap.models.dpt.forward_vit import forward_vit
from hubmap.models.dpt.core import Interpolate


class DPT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        features: int = 256,
        backbone: Backbone = Backbone.vitb_rn50_384,
        readout: str = "project",
        channels_last: bool = False,
        use_bn: bool = True,
        enable_attention_hooks: bool = False,
        use_pretrained: bool = False,
    ):
        super().__init__()

        self.channels_last = channels_last

        hooks = {
            Backbone.vitb_rn50_384: [0, 1, 8, 11],
            Backbone.vitb16_384: [2, 5, 8, 11],
            Backbone.vitl16_384: [5, 11, 17, 23],
            # # resnext101_wsl: None, as it does not use hooks.
            # # Addition to the code of the source.
            # Backbone.resnext101_wsl: None,
        }
        self.num_classes = num_classes

        self.pretrained, self.scratch = make_encoder(
            backbone,
            features,
            use_pretrained,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, x):
        if self.channels_last is True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out
