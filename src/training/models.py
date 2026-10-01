"""Build torchvision models with a new classification head, or as frozen feature extractors."""

from __future__ import annotations

from typing import Tuple, Union

from torch import nn
from torchvision import models as tvm

from src.training.config import FAMILIES


DEFAULT_INPUT_SIZE = 384
VIT_INPUT_SIZE = 224

_BUILDERS = {
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.DEFAULT),
    "vgg16": (tvm.vgg16, tvm.VGG16_Weights.DEFAULT),
    "resnet152": (tvm.resnet152, tvm.ResNet152_Weights.DEFAULT),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT),
    "googlenet": (tvm.googlenet, tvm.GoogLeNet_Weights.DEFAULT),
    "vit_b16": (tvm.vit_b_16, tvm.ViT_B_16_Weights.DEFAULT),
    "alexnet": (tvm.alexnet, tvm.AlexNet_Weights.DEFAULT),
}


def input_size(family: str) -> int:
    return VIT_INPUT_SIZE if family == "vit_b16" else DEFAULT_INPUT_SIZE


def _backbone(family: str, pretrained: bool) -> nn.Module:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}")
    builder, weights = _BUILDERS[family]
    if family == "googlenet":
        # Pretrained GoogLeNet drops its auxiliary heads by itself; passing aux_logits there raises.
        if pretrained:
            return builder(weights=weights)
        return builder(weights=None, aux_logits=False, init_weights=True)
    return builder(weights=weights if pretrained else None)


def _head_slot(model: nn.Module, family: str) -> Tuple[nn.Module, Union[str, int]]:
    """Return (parent module, key) of the final linear layer."""
    if family in ("resnet152", "googlenet"):
        return model, "fc"
    if family == "vit_b16":
        return model.heads, "head"
    return model.classifier, len(model.classifier) - 1


def _get_head(model: nn.Module, family: str) -> nn.Linear:
    parent, key = _head_slot(model, family)
    return parent[key] if isinstance(key, int) else getattr(parent, key)


def _set_head(model: nn.Module, family: str, new: nn.Module) -> None:
    parent, key = _head_slot(model, family)
    if isinstance(key, int):
        parent[key] = new
    else:
        setattr(parent, key, new)


def build_model(family: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Pretrained backbone with its last linear layer replaced by a ``num_classes`` head."""
    model = _backbone(family, pretrained)
    _set_head(model, family, nn.Linear(_get_head(model, family).in_features, num_classes))
    return model


def feature_extractor(family: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Backbone with its last linear layer removed, and the size of the feature vector it returns."""
    model = _backbone(family, pretrained)
    feature_dim = _get_head(model, family).in_features
    _set_head(model, family, nn.Identity())
    return model, feature_dim
