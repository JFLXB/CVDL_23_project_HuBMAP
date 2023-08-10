from enum import StrEnum
from hubmap.models.dpt.pretrained import make_pretrained_vitb_rn50_384
from hubmap.models.dpt.pretrained import make_pretrained_vitl16_384
from hubmap.models.dpt.pretrained import make_pretrained_vitb16_384

# from hubmap.models.dpt.pretrained import make_pretrained_resnext101_wsl
from hubmap.models.dpt.scratch import make_scratch


class Backbone(StrEnum):
    vitl16_384 = "vitl16_384"
    vitb_rn50_384 = "vitb_rn50_384"
    vitb16_384 = "vitb16_384"
    # resnext101_wsl = "resnext101_wsl"


def make_encoder(
    backbone: Backbone,
    features,
    use_pretrained,
    groups=1,
    expand=False,
    exportable=True,
    hooks=None,
    use_vit_only=False,
    use_readout="ignore",
    enable_attention_hooks=False,
):
    if backbone == Backbone.vitl16_384:
        pretrained = make_pretrained_vitl16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )
    elif backbone == Backbone.vitb_rn50_384:
        pretrained = make_pretrained_vitb_rn50_384(
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )
    elif backbone == Backbone.vitb16_384:
        pretrained = make_pretrained_vitb16_384(
            use_pretrained,
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        scratch = make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )
    # elif backbone == Backbone.resnext101_wsl:
    #     pretrained = make_pretrained_resnext101_wsl(use_pretrained)
    #     scratch = make_scratch(
    #         [256, 512, 1024, 2048], features, groups=groups, expand=expand
    #     )
    else:
        raise NotImplementedError
    return pretrained, scratch
