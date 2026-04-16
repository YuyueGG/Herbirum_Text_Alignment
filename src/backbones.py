from __future__ import annotations

from torch import nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    convnext_base,
    resnet50,
    swin_t,
)


def build_backbone(arch: str) -> tuple[nn.Module, int]:
    arch = arch.lower()
    if arch == "resnet":
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_dim

    if arch == "convnext":
        backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        in_dim = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
        return backbone, in_dim

    if arch == "swin":
        backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_dim = backbone.head.in_features
        backbone.head = nn.Identity()
        return backbone, in_dim

    raise ValueError(f"Unknown architecture: {arch}")


def freeze_low_level_layers(backbone: nn.Module, arch: str, train_from: str) -> None:
    arch = arch.lower()
    train_from = train_from.lower()

    if train_from == "all":
        return

    if arch == "resnet":
        if train_from == "layer2":
            for name, param in backbone.named_parameters():
                if name.startswith("conv1") or name.startswith("bn1") or name.startswith("layer1."):
                    param.requires_grad = False
            return
        if train_from == "layer3":
            for name, param in backbone.named_parameters():
                if not any(key in name for key in ["layer3", "layer4"]):
                    param.requires_grad = False
            return

    if arch == "convnext":
        if train_from == "layer2":
            allowed = ["stages.1", "stages.2", "stages.3"]
            for name, param in backbone.named_parameters():
                if not any(key in name for key in allowed):
                    param.requires_grad = False
            return
        if train_from == "layer3":
            allowed = ["stages.2", "stages.3"]
            for name, param in backbone.named_parameters():
                if not any(key in name for key in allowed):
                    param.requires_grad = False
            return

    if arch == "swin":
        if train_from == "layer2":
            allowed = ["stages.1", "stages.2", "stages.3"]
            for name, param in backbone.named_parameters():
                if not any(key in name for key in allowed):
                    param.requires_grad = False
            return
        if train_from == "layer3":
            allowed = ["stages.2", "stages.3"]
            for name, param in backbone.named_parameters():
                if not any(key in name for key in allowed):
                    param.requires_grad = False
            return

    raise ValueError(f"Unsupported freeze setting: arch={arch}, train_from={train_from}")
