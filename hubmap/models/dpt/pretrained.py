import timm
from hubmap.models.dpt.backbone import make_vit_b_rn50_backbone


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