import torch
import timm
from hubmap.models.dpt.backbone import make_vit_b_rn50_backbone
from hubmap.models.dpt.backbone import make_vit_b16_backbone

# from hubmap.models.dpt.backbone import make_resnet_backbone


def make_pretrained_vitb_rn50_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
):
    model = timm.create_model(
        "vit_base_r50_s16_384.orig_in21k_ft_in1k", pretrained=pretrained
    )

    hooks = [0, 1, 8, 11] if hooks is None else hooks
    return make_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def make_pretrained_vitl16_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
):
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)
    hooks = [5, 11, 17, 23] if hooks is None else hooks
    return make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def make_pretrained_vitb16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


# def make_pretrained_resnext101_wsl(use_pretrained):
#     resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
#     return make_resnet_backbone(resnet)
