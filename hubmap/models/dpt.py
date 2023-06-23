"""
Vision Transformers for Dense Prediction
========================================

Authors: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
Paper: https://arxiv.org/abs/2103.13413
Code: https://github.com/isl-org/DPT/tree/main
"""
from functools import partial
from typing import Optional

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from hubmap.models.dpt_block import (
    Backbone,
    make_encoder,
    make_fusion_block,
    forward_vit,
)


class Interpolate(nn.Module):
    """Interpolation module from: https://github.com/isl-org/DPT/tree/main"""

    def __init__(
        self, scale_factor: int, mode: str, align_corners: Optional[bool] = False
    ):
        super().__init__()

        self.interpolate = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class DPT(pl.LightningModule):
    def __init__(
        self,
        # MODEL PARAMETERS
        num_classes: int,
        features: int = 256,
        backbone: Backbone = Backbone.VITB_RN50_384,
        use_pretrained: bool = False,
        readout: str = "project",
        enable_attention_hooks: bool = False,
        # TRAINING PARAMETERS
        criterion: Optional = None,
        # Either a partially initialized optimizer or None. In case of None a default
        # optimizer is initialized.
        optimizer: Optional[partial] = None,
    ):
        """
        DPT
        ---

        Authors: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        Paper: https://arxiv.org/abs/2103.13413

        Code based on: : https://github.com/isl-org/DPT/tree/main
        """
        super().__init__()

        # USED FOR TRAIING
        self.criterion = criterion
        self.optim = optimizer

        # SETUP THE MODEL ITSELF
        hooks = {}

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

        self.scratch.refinenet1 = make_fusion_block(features, True)
        self.scratch.refinenet2 = make_fusion_block(features, True)
        self.scratch.refinenet3 = make_fusion_block(features, True)
        self.scratch.refinenet4 = make_fusion_block(features, True)

        #  HEAD
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        # THIS CODE SEEMS TO BE UNUSED IN THE ORIGINAL REPO.
        # self.auxlayer = nn.Sequential(
        #     nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(features),
        #     nn.ReLU(True),
        #     nn.Dropout(0.1, False),
        #     nn.Conv2d(features, num_classes, kernel_size=1),
        # )

    def forward(self, x):
        layer1, layer2, layer3, layer4 = forward_vit(self.pretrained, x)

        layer1_rn = self.scratch.layer1_rn(layer1)
        layer2_rn = self.scratch.layer2_rn(layer2)
        layer3_rn = self.scratch.layer3_rn(layer3)
        layer4_rn = self.scratch.layer4_rn(layer4)

        path4 = self.scratch.refinenet4(layer4_rn)
        path3 = self.scratch.refinenet3(path4, layer3_rn)
        path2 = self.scratch.refinenet2(path3, layer2_rn)
        path1 = self.scratch.refinenet1(path2, layer1_rn)

        out = self.scratch.output_conv(path1)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # For Accurcay - needs to be changed for the task at hand.
        # preds = torch.argmax(logits, dim=1)
        # self.val_accuracy.update(preds, y)

        # From lightning documentation: Calling self.log will surface up scalars for
        # you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # From lightning documentation: Calling self.log will surface up scalars for
        # you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        # self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters())
        return optimizer
