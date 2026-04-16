from __future__ import annotations

import math
import os
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from .backbones import build_backbone, freeze_low_level_layers


def make_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


class ImageEncoder(nn.Module):
    def __init__(
        self,
        arch: str,
        cls_proj_dim: int,
        align_dim: int,
        dropout: float,
        share_img_proj: bool,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.backbone, self.backbone_dim = build_backbone(self.arch)
        self.share_img_proj = bool(share_img_proj)

        if self.share_img_proj:
            if align_dim <= 0:
                raise ValueError("align_dim must be positive when share_img_proj is enabled.")
            self.shared_dim = int(align_dim)
            self.shared_proj = nn.Identity() if self.shared_dim == self.backbone_dim else make_mlp(self.backbone_dim, self.shared_dim, dropout)
            self.cls_dim = self.shared_dim
            self.align_dim = self.shared_dim
            self.cls_proj = nn.Identity()
            self.align_proj = nn.Identity()
        else:
            self.shared_dim = None
            self.shared_proj = None

            if cls_proj_dim <= 0 or cls_proj_dim == self.backbone_dim:
                self.cls_dim = self.backbone_dim
                self.cls_proj = nn.Identity()
            else:
                self.cls_dim = int(cls_proj_dim)
                self.cls_proj = make_mlp(self.backbone_dim, self.cls_dim, dropout)

            if align_dim <= 0 or align_dim == self.backbone_dim:
                self.align_dim = self.backbone_dim
                self.align_proj = nn.Identity()
            else:
                self.align_dim = int(align_dim)
                self.align_proj = make_mlp(self.backbone_dim, self.align_dim, dropout)

    def projection_parameters(self):
        if self.share_img_proj:
            return list(self.shared_proj.parameters())
        return list(self.cls_proj.parameters()) + list(self.align_proj.parameters())

    def forward(self, pixel_values: torch.Tensor):
        features = self.backbone(pixel_values)
        if self.share_img_proj:
            shared = self.shared_proj(features)
            return features, shared, shared
        cls_features = self.cls_proj(features)
        align_features = self.align_proj(features)
        return features, cls_features, align_features


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_dim: int,
        dropout: float,
        pool: str,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_dim = int(self.backbone.config.hidden_size)
        self.pool = pool
        if out_dim <= 0 or out_dim == self.hidden_dim:
            self.out_dim = self.hidden_dim
            self.proj = nn.Identity()
        else:
            self.out_dim = int(out_dim)
            self.proj = make_mlp(self.hidden_dim, self.out_dim, dropout)

    @staticmethod
    def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if self.pool == "cls":
            pooled = outputs.last_hidden_state[:, 0]
        else:
            pooled = self.masked_mean(outputs.last_hidden_state, attention_mask)
        return self.proj(pooled)


class LinearHead(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CosineHead(nn.Module):
    def __init__(self, dim: int, n_classes: int, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return self.scale * (x_norm @ w_norm.t())


class HerbariumTextAlignmentModel(nn.Module):
    def __init__(
        self,
        arch: str,
        n_classes: int,
        cls_proj_dim: int,
        align_dim: int,
        dropout: float,
        head_type: str,
        head_scale: float,
        freeze_img_low: bool,
        freeze_txt_low: bool,
        freeze_txt_all: bool,
        img_train_from: str,
        temp_init: float,
        learnable_temperature: bool,
        share_img_proj: bool,
        text_model: str,
        text_pool: str,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.n_classes = int(n_classes)

        self.img_enc = ImageEncoder(
            arch=self.arch,
            cls_proj_dim=cls_proj_dim,
            align_dim=align_dim,
            dropout=dropout,
            share_img_proj=share_img_proj,
        )
        self.txt_enc = TextEncoder(
            model_name=text_model,
            out_dim=self.img_enc.align_dim,
            dropout=dropout,
            pool=text_pool,
        )

        if head_type == "cosine":
            self.classifier = CosineHead(self.img_enc.cls_dim, self.n_classes, scale=head_scale)
        elif head_type == "linear":
            self.classifier = LinearHead(self.img_enc.cls_dim, self.n_classes)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.learnable_temperature = bool(learnable_temperature)
        if self.learnable_temperature:
            self.log_temp = nn.Parameter(torch.tensor(math.log(float(temp_init)), dtype=torch.float32))
        else:
            self.register_buffer("fixed_temp", torch.tensor(float(temp_init), dtype=torch.float32), persistent=True)
            self.log_temp = None

        if freeze_img_low:
            freeze_low_level_layers(self.img_enc.backbone, self.arch, img_train_from)

        if freeze_txt_all:
            for _, param in self.txt_enc.backbone.named_parameters():
                param.requires_grad = False
        elif freeze_txt_low:
            for name, param in self.txt_enc.backbone.named_parameters():
                if not any(key in name for key in ["layer.4", "layer.5"]):
                    param.requires_grad = False

    @property
    def temperature(self) -> torch.Tensor:
        if self.learnable_temperature:
            return self.log_temp.exp().clamp(1e-3, 1.0)
        return self.fixed_temp.clamp(1e-3, 1.0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        _, cls_features, align_features = self.img_enc(pixel_values)
        logits = self.classifier(cls_features)
        if input_ids is None or attention_mask is None:
            return logits, cls_features, align_features, None
        text_features = self.txt_enc(input_ids, attention_mask)
        return logits, cls_features, align_features, text_features
