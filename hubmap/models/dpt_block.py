from functools import partial
from typing import Optional, List, Dict
from enum import Enum

import torch.nn.functional as F

import math
import timm
import types
import torch
import torch.nn as nn


class Backbone(Enum):
    VITL16_384: str = "vitl16_384"
    VITB_RN50_384: str = "vitb_rn50_384"
    VITB16_384: str = "vitb16_384"
    RESNEXT101_wsl: str = "resnext101_wsl"

    def __str__(self):
        return self.value


activations = {}


def get_activation(name: str):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}


def get_attention(name: str):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, _ = (qkv[0], qkv[1], qkv[2])

        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attention[name] = attn

    return hook


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super().__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super().__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze()


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super().__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)
        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [ProjectReadout(vit_features, start_index) for _ in features]
    else:
        raise ValueError(
            f"Operation '{use_readout}' is unkown, "
            "choose one of 'ignore', 'add', 'project'"
        )
    return readout_oper


def make_pretrained_vitl16_384(
    use_pretrained: bool, hooks: List, use_readout: str, enable_attention_hooks: bool
):
    features = [96, 192, 384, 768]
    size = [384, 384]
    vit_features = 768
    start_index = 1

    model = timm.create_model("vit_large_patch16_384", pretrained=use_pretrained)
    hooks = [5, 11, 17, 23] if hooks is None else hooks

    pretrained = nn.Module()
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # FROM THE ORIGINAL IMPLEMENTATION:
    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the
    # library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def make_pretrained_vitb_rn50_384(
    use_pretrained: bool, hooks: List, use_readout: str, enable_attention_hooks: bool
):
    raise NotImplementedError


def make_pretrained_vitb16_384(
    use_pretrained: bool, hooks: List, use_readout: str, enable_attention_hooks: bool
):
    raise NotImplementedError


def make_pretrained_resnext101_wsl(
    use_pretrained: bool, hooks: List, use_readout: str, enable_attention_hooks: bool
):
    raise NotImplementedError


def make_scratch(in_shape, out_shape, groups: int = 1, expand: bool = False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape * 2 if expand else out_shape
    out_shape3 = out_shape * 4 if expand else out_shape
    out_shape4 = out_shape * 8 if expand else out_shape

    base_layer = partial(
        nn.Conv2d,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer1_rn = base_layer(in_channels=in_shape[0], out_channels=out_shape1)
    scratch.layer2_rn = base_layer(in_channels=in_shape[1], out_channels=out_shape2)
    scratch.layer3_rn = base_layer(in_channels=in_shape[2], out_channels=out_shape3)
    scratch.layer4_rn = base_layer(in_channels=in_shape[3], out_channels=out_shape4)
    return scratch


__KNOWN_BACKBONES: Dict = {
    Backbone.VITL16_384: (
        make_pretrained_vitl16_384,
        partial(make_scratch, in_shape=[256, 512, 1024, 1024]),
    ),
    Backbone.VITB_RN50_384: (
        make_pretrained_vitb_rn50_384,
        partial(make_scratch, in_shape=[256, 512, 768, 768]),
    ),
    Backbone.VITB16_384: (
        make_pretrained_vitb16_384,
        partial(make_scratch, in_shape=[96, 192, 384, 768]),
    ),
    Backbone.RESNEXT101_wsl: (
        make_pretrained_resnext101_wsl,
        partial(make_scratch, in_shape=[256, 512, 1024, 2048]),
    ),
}


def make_encoder(
    backbone: Backbone,
    features: int,
    use_pretrained: bool,
    groups: int = 1,
    expand: bool = False,
    exportable: bool = True,
    hooks: Optional[List] = None,
    use_vit_only: bool = False,
    use_readout: str = "ignore",
    enable_attention_hooks: bool = False,
):
    make_encoder = __KNOWN_BACKBONES.get(backbone, None)
    if make_encoder is None:
        raise ValueError(f"Backbone '{str(backbone)}' not implemented.")

    make_pretrained, make_scratch = make_encoder
    pretrained = make_pretrained(
        use_pretrained,
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    scratch = make_scratch(out_shape=features, groups=groups, expand=expand)
    return pretrained, scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )
        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features,
        activation,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
    ):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features

        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConvUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConvUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConvUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConvUnit2(output)
        output = F.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output


def make_fusion_block(features, use_bn):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


def forward_vit(pretrained, x):
    b, c, h, w = x.shape

    # glob = pretrained.model.forward_flex(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4
