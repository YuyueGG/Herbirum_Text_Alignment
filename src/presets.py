from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class TrainingConfig:
    preset_name: str
    arch: str
    epochs: int
    batch_size: int
    base_lr: float
    weight_decay: float
    momentum: float
    head_lr_mult: float
    head_type: str
    head_scale: float
    label_smoothing: float
    cls_proj_dim: int
    align_dim: int
    share_img_proj: bool
    text_pool: str
    align_type: str
    w_align: float
    warmup_align_epochs: int
    align_ramp_epochs: int
    align_backbone_grad_scale: float
    min_text_tokens: int
    proto_max_per_class: int
    proto_batch_size: int
    proto_recompute_every: int
    hard_neg_alpha: float
    balanced_sampler: bool
    cb_loss: bool
    cb_beta: float
    freeze_img_low: bool
    freeze_txt_low: bool
    freeze_txt_all: bool
    img_train_from: str
    amp: bool
    tta: bool
    augment: bool
    deterministic: bool
    max_len: int
    tokenizer_name: str
    text_model: str
    dropout: float
    save_best: bool
    learnable_temperature: bool
    temp_init: float
    grad_clip_val: Optional[float]
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


BASE_CONFIG = TrainingConfig(
    preset_name="base",
    arch="resnet",
    epochs=80,
    batch_size=32,
    base_lr=3e-4,
    weight_decay=5e-3,
    momentum=0.9,
    head_lr_mult=5.0,
    head_type="cosine",
    head_scale=8.0,
    label_smoothing=0.0,
    cls_proj_dim=512,
    align_dim=512,
    share_img_proj=False,
    text_pool="masked_mean",
    align_type="inst",
    w_align=0.25,
    warmup_align_epochs=8,
    align_ramp_epochs=0,
    align_backbone_grad_scale=1.0,
    min_text_tokens=5,
    proto_max_per_class=64,
    proto_batch_size=64,
    proto_recompute_every=1,
    hard_neg_alpha=0.0,
    balanced_sampler=True,
    cb_loss=True,
    cb_beta=0.999,
    freeze_img_low=True,
    freeze_txt_low=True,
    freeze_txt_all=False,
    img_train_from="layer3",
    amp=True,
    tta=True,
    augment=True,
    deterministic=True,
    max_len=64,
    tokenizer_name="distilbert-base-uncased",
    text_model="distilbert-base-uncased",
    dropout=0.1,
    save_best=True,
    learnable_temperature=True,
    temp_init=0.07,
    grad_clip_val=1.0,
    seed=42,
)


PRESETS = {
    # ResNet-50, 44 classes
    # Coupled Instance (InfoNCE), shared projection, learnable temperature
    "resnet44": replace(
        BASE_CONFIG,
        preset_name="resnet44",
        arch="resnet",
        epochs=80,
        batch_size=32,
        base_lr=3e-4,
        weight_decay=5e-3,
        head_lr_mult=5.0,
        head_type="cosine",
        head_scale=8.0,
        label_smoothing=0.0,
        cls_proj_dim=512,
        align_dim=512,
        share_img_proj=True,
        align_type="inst",
        w_align=0.25,
        warmup_align_epochs=8,
        align_ramp_epochs=0,
        align_backbone_grad_scale=1.0,
        hard_neg_alpha=0.0,
        freeze_img_low=True,
        freeze_txt_low=True,
        freeze_txt_all=False,
        img_train_from="layer3",
        amp=True,
        tta=True,
        augment=True,
        learnable_temperature=True,
        temp_init=0.05,
        grad_clip_val=1.0,
    ),

    # ResNet-50, 9 classes
    # Coupled Instance (InfoNCE), shared projection, learnable temperature
    "resnet9": replace(
        BASE_CONFIG,
        preset_name="resnet9",
        arch="resnet",
        epochs=80,
        batch_size=32,
        base_lr=3e-4,
        weight_decay=5e-3,
        head_lr_mult=5.0,
        head_type="cosine",
        head_scale=8.0,
        label_smoothing=0.0,
        cls_proj_dim=512,
        align_dim=512,
        share_img_proj=True,
        align_type="inst",
        w_align=0.20,
        warmup_align_epochs=12,
        align_ramp_epochs=0,
        align_backbone_grad_scale=1.0,
        hard_neg_alpha=0.0,
        freeze_img_low=True,
        freeze_txt_low=True,
        freeze_txt_all=False,
        img_train_from="layer3",
        amp=True,
        tta=True,
        augment=True,
        learnable_temperature=True,
        temp_init=0.06,
        grad_clip_val=1.0,
    ),

    # ConvNeXt, 44 classes
    # Coupled Instance (InfoNCE), shared projection, fixed temperature
    "convnext44": replace(
        BASE_CONFIG,
        preset_name="convnext44",
        arch="convnext",
        epochs=80,
        batch_size=64,
        base_lr=3e-4,
        weight_decay=5e-4,
        head_lr_mult=10.0,
        head_type="cosine",
        head_scale=8.0,
        label_smoothing=0.0,
        cls_proj_dim=512,
        align_dim=512,
        share_img_proj=True,
        align_type="inst",
        w_align=0.20,
        warmup_align_epochs=0,
        align_ramp_epochs=0,
        align_backbone_grad_scale=1.0,
        hard_neg_alpha=0.0,
        freeze_img_low=False,
        freeze_txt_low=False,
        freeze_txt_all=False,
        img_train_from="layer3",
        amp=True,
        tta=True,
        augment=True,
        learnable_temperature=False,
        temp_init=0.07,
        grad_clip_val=1.0,
    ),

    # ConvNeXt, 9 classes
    # Decoupled Prototype softmax, no shared projection, fixed temperature
    "convnext9": replace(
        BASE_CONFIG,
        preset_name="convnext9",
        arch="convnext",
        epochs=80,
        batch_size=64,
        base_lr=3e-4,
        weight_decay=5e-3,
        head_lr_mult=10.0,
        head_type="cosine",
        head_scale=8.0,
        label_smoothing=0.0,
        cls_proj_dim=0,
        align_dim=512,
        share_img_proj=False,
        align_type="proto",
        w_align=0.25,
        warmup_align_epochs=0,
        align_ramp_epochs=0,
        align_backbone_grad_scale=1.0,
        hard_neg_alpha=0.0,
        freeze_img_low=False,
        freeze_txt_low=False,
        freeze_txt_all=False,
        img_train_from="layer3",
        amp=True,
        tta=True,
        augment=True,
        learnable_temperature=False,
        temp_init=0.07,
        grad_clip_val=1.0,
    ),

    # Swin-T, 44 classes
    # Decoupled Prototype DCL, no shared projection, fixed temperature, no grad clip
    "swin44": replace(
        BASE_CONFIG,
        preset_name="swin44",
        arch="swin",
        epochs=80,
        batch_size=64,
        base_lr=3e-4,
        weight_decay=5e-4,
        head_lr_mult=5.0,
        head_type="cosine",
        head_scale=8.0,
        label_smoothing=0.0,
        cls_proj_dim=0,
        align_dim=768,
        share_img_proj=False,
        align_type="proto_dcl",
        w_align=0.25,
        warmup_align_epochs=12,
        align_ramp_epochs=0,
        align_backbone_grad_scale=1.0,
        hard_neg_alpha=1.0,
        freeze_img_low=False,
        freeze_txt_low=False,
        freeze_txt_all=False,
        img_train_from="layer3",
        amp=True,
        tta=True,
        augment=True,
        learnable_temperature=False,
        temp_init=0.06,
        grad_clip_val=None,
    ),

    # Swin-T, 9 classes
    # Topology-3 with the same confirmed hyperparameter pack as Swin44
    "swin9": replace(
        BASE_CONFIG,
        preset_name="swin9",
        arch="swin",
        epochs=80,
        batch_size=64,
        base_lr=3e-4,
        weight_decay=5e-4,
        head_lr_mult=5.0,
        head_type="cosine",
        head_scale=8.0,
        label_smoothing=0.0,
        cls_proj_dim=0,
        align_dim=768,
        share_img_proj=False,
        align_type="proto_dcl",
        w_align=0.25,
        warmup_align_epochs=12,
        align_ramp_epochs=0,
        align_backbone_grad_scale=1.0,
        hard_neg_alpha=1.0,
        freeze_img_low=False,
        freeze_txt_low=False,
        freeze_txt_all=False,
        img_train_from="layer3",
        amp=True,
        tta=True,
        augment=True,
        learnable_temperature=False,
        temp_init=0.06,
        grad_clip_val=None,
    ),
}


PRESET_NAMES = tuple(PRESETS.keys())


def build_preset(name: str, seed: int) -> TrainingConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    return replace(PRESETS[name], seed=seed)