#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualize_model_gradcam.py

Generate a single Grad-CAM overlay for one model and one specimen image.

Design goals:
- One model per run
- No hard-coded checkpoint paths
- Repository-relative path resolution
- Default input image: examples/sample_images/1.png
- Same input geometry for all models
- No internal project acronyms in user-facing outputs

Supported model families:
1) ours
   - Image-text alignment model trained with the released presets
   - Loads only the image encoder and classifier weights from the checkpoint

2) baseline
   - Pure vision baseline model
   - Requires an explicit architecture name

Output:
- One Grad-CAM overlay image
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms as T
from torchvision.models import (
    convnext_base,
    resnet50,
    swin_t,
)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.presets import PRESET_NAMES, build_preset


REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_IMAGE = REPO_ROOT / "examples" / "sample_images" / "1.png"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "gradcam"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BACKBONE_DIMS = {
    "resnet": 2048,
    "convnext": 1024,
    "swin": 768,
}


# -------------------------------------------------------------------------
# Path and metadata helpers
# -------------------------------------------------------------------------
def resolve_repo_path(path_value: Path) -> Path:
    """Resolve a path relative to the repository root."""
    if path_value.is_absolute():
        return path_value
    return (REPO_ROOT / path_value).resolve()


def default_label2id_path(preset: Optional[str]) -> Path:
    """Choose the default label2id file from the repository examples."""
    if preset is not None and preset.endswith("9"):
        return REPO_ROOT / "examples" / "sample_jsonl" / "label2id9.json"
    return REPO_ROOT / "examples" / "sample_jsonl" / "label2id.json"


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def id2label_from_label2id(label2id: dict) -> Dict[int, str]:
    """Convert label2id into id2label."""
    return {int(v): k for k, v in label2id.items()}


def extract_specimen_id(image_path: Path) -> str:
    """Extract a stable identifier from the image filename."""
    stem = image_path.stem
    if "__" in stem:
        return stem.split("__")[-1]
    return stem


def get_label_from_filename(image_path: Path) -> str:
    """Parse genus_species from the image filename."""
    stem = image_path.stem
    main = stem.split("__")[0]
    parts = [p for p in main.split("_") if p]

    if len(parts) >= 2:
        return f"{parts[0].lower()}_{parts[1].lower()}"

    return stem.lower()


def pretty_arch_name(arch: str) -> str:
    """Convert an internal architecture name into a display name."""
    mapping = {
        "resnet": "ResNet",
        "convnext": "ConvNeXt",
        "swin": "Swin",
    }
    return mapping.get(arch.lower(), arch)


def pretty_model_name(family: str, arch: str) -> str:
    """Build a user-facing model name."""
    arch_name = pretty_arch_name(arch)

    if family == "ours":
        return f"{arch_name} VLM-KD"

    if family == "baseline":
        return f"{arch_name} Baseline"

    return arch_name


def model_file_slug(family: str, arch: str) -> str:
    """Build a clean file-name slug."""
    suffix = "ours" if family == "ours" else "baseline"
    return f"{arch.lower()}_{suffix}"


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------
def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a leading module prefix if the checkpoint was saved from DataParallel."""
    if not state_dict:
        return state_dict

    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}

    return state_dict


def extract_state_dict(checkpoint_obj) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from common checkpoint formats."""
    if isinstance(checkpoint_obj, dict):
        if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
            return strip_module_prefix(checkpoint_obj["model"])

        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return strip_module_prefix(checkpoint_obj["state_dict"])

        if all(torch.is_tensor(value) for value in checkpoint_obj.values()):
            return strip_module_prefix(checkpoint_obj)

    raise RuntimeError("No valid state_dict was found in the checkpoint.")


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load a checkpoint state_dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return extract_state_dict(checkpoint)


def keep_image_only_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only the image encoder and classifier weights."""
    keep_prefixes = ("img_enc.", "classifier.")
    return {key: value for key, value in state_dict.items() if key.startswith(keep_prefixes)}


# -------------------------------------------------------------------------
# Image preprocessing
# -------------------------------------------------------------------------
class ResizeShortSideAndPad:
    """Resize the shorter side to 224 and pad/crop to a square image."""

    def __init__(self, size: int = 224, fill: int = 128):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size

        if width < height:
            new_width = self.size
            new_height = int(height * (self.size / width))
        else:
            new_height = self.size
            new_width = int(width * (self.size / height))

        image = image.resize((new_width, new_height), Image.BICUBIC)

        pad_left = max(0, (self.size - new_width) // 2)
        pad_right = max(0, self.size - new_width - pad_left)
        pad_top = max(0, (self.size - new_height) // 2)
        pad_bottom = max(0, self.size - new_height - pad_top)

        if any([pad_left, pad_right, pad_top, pad_bottom]):
            image = ImageOps.expand(
                image,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=self.fill,
            )

        if image.size != (self.size, self.size):
            image = image.crop((0, 0, self.size, self.size))

        return image


def build_same_input_transform():
    """Build the same input transform for all models."""
    return T.Compose(
        [
            ResizeShortSideAndPad(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_input(image_path: Path, device: torch.device) -> torch.Tensor:
    """Load one input image."""
    image = Image.open(image_path).convert("RGB")
    x = build_same_input_transform()(image)
    return x.unsqueeze(0).to(device)


def denorm_to_rgb(x: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor image into an RGB NumPy array."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)

    image = x.detach().clone() * std + mean
    image = image.clamp(0, 1)
    image = image[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    return image


def normalize_cam_np(cam: np.ndarray) -> np.ndarray:
    """Normalize a CAM to [0, 1]."""
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


# -------------------------------------------------------------------------
# Baseline models
# -------------------------------------------------------------------------
def load_baseline_model(
    arch: str,
    checkpoint_path: Path,
    n_classes: int,
    device: torch.device,
) -> nn.Module:
    """Load a pure vision baseline model."""
    arch = arch.lower()

    if arch == "resnet":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    elif arch == "convnext":
        model = convnext_base(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, n_classes)

    elif arch == "swin":
        model = swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, n_classes)

    else:
        raise ValueError(f"Unsupported baseline architecture: {arch}")

    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    return model


# -------------------------------------------------------------------------
# Image-only reconstruction for the proposed model
# -------------------------------------------------------------------------
def build_backbone(arch: str) -> Tuple[nn.Module, int]:
    """Build a vision backbone without classifier weights."""
    arch = arch.lower()

    if arch == "resnet":
        model = resnet50(weights=None)
        in_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_dim

    if arch == "convnext":
        model = convnext_base(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        return model, in_dim

    if arch == "swin":
        model = swin_t(weights=None)
        in_dim = model.head.in_features
        model.head = nn.Identity()
        return model, in_dim

    raise ValueError(f"Unsupported architecture: {arch}")


def make_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Module:
    """Build the projection MLP used by the released training code."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


class ImageEncoder(nn.Module):
    """Image encoder with shared or decoupled projection heads."""

    def __init__(
        self,
        arch: str,
        cls_proj_dim: int,
        align_dim: int,
        dropout: float = 0.1,
        share_img_proj: bool = False,
    ):
        super().__init__()

        self.arch = arch.lower()
        self.backbone, self.backbone_dim = build_backbone(self.arch)
        self.share_img_proj = bool(share_img_proj)

        if self.share_img_proj:
            self.shared_dim = int(cls_proj_dim)
            self.shared_proj = (
                nn.Identity()
                if self.shared_dim == self.backbone_dim
                else make_mlp(self.backbone_dim, self.shared_dim, dropout)
            )
            self.cls_dim = self.shared_dim
            self.align_dim = self.shared_dim
            self.cls_proj = nn.Identity()
            self.align_proj = nn.Identity()
        else:
            self.cls_dim = int(cls_proj_dim)
            self.align_dim = int(align_dim)

            self.cls_proj = (
                nn.Identity()
                if self.cls_dim == self.backbone_dim
                else make_mlp(self.backbone_dim, self.cls_dim, dropout)
            )
            self.align_proj = (
                nn.Identity()
                if self.align_dim == self.backbone_dim
                else make_mlp(self.backbone_dim, self.align_dim, dropout)
            )

            self.shared_dim = None
            self.shared_proj = None

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)

        if self.share_img_proj:
            shared = self.shared_proj(feat)
            return feat, shared, shared

        cls_feat = self.cls_proj(feat)
        align_feat = self.align_proj(feat)
        return feat, cls_feat, align_feat


class LinearHead(nn.Module):
    """Linear classifier head."""

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CosineHead(nn.Module):
    """Cosine classifier head."""

    def __init__(self, dim: int, n_classes: int, scale: float = 16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim))
        nn.init.xavier_normal_(self.W)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.W, dim=1)
        return self.scale * (x_norm @ w_norm.t())


class OursImageOnlyModel(nn.Module):
    """Image-only inference model reconstructed from the released checkpoint."""

    def __init__(
        self,
        arch: str,
        n_classes: int,
        cls_proj_dim: int,
        align_dim: int,
        share_img_proj: bool,
        head_type: str,
        head_scale: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.arch = arch.lower()
        self.n_classes = int(n_classes)

        self.img_enc = ImageEncoder(
            arch=arch,
            cls_proj_dim=cls_proj_dim,
            align_dim=align_dim,
            dropout=dropout,
            share_img_proj=share_img_proj,
        )

        if head_type == "cosine":
            self.classifier = CosineHead(
                dim=self.img_enc.cls_dim,
                n_classes=self.n_classes,
                scale=head_scale,
            )
        elif head_type == "linear":
            self.classifier = LinearHead(
                dim=self.img_enc.cls_dim,
                n_classes=self.n_classes,
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, cls_feat, _ = self.img_enc(x)
        logits = self.classifier(cls_feat)
        return logits


@dataclass(frozen=True)
class OursCheckpointConfig:
    """Minimal model structure inferred from an image-text checkpoint."""

    arch: str
    n_classes: int
    cls_proj_dim: int
    align_dim: int
    share_img_proj: bool
    head_type: str


def infer_ours_checkpoint_config(
    state_dict: Dict[str, torch.Tensor],
    arch: str,
    share_img_proj: Optional[bool] = None,
) -> OursCheckpointConfig:
    """Infer the image-side architecture from checkpoint keys."""
    arch = arch.lower()
    backbone_dim = BACKBONE_DIMS[arch]

    if "classifier.W" in state_dict:
        head_type = "cosine"
        n_classes, cls_dim = state_dict["classifier.W"].shape
    elif "classifier.fc.weight" in state_dict:
        head_type = "linear"
        n_classes, cls_dim = state_dict["classifier.fc.weight"].shape
    else:
        raise RuntimeError("Cannot infer classifier type from checkpoint.")

    if share_img_proj is None:
        has_shared = any(key.startswith("img_enc.shared_proj.") for key in state_dict)
        has_decoupled = any(
            key.startswith("img_enc.cls_proj.") or key.startswith("img_enc.align_proj.")
            for key in state_dict
        )

        if has_shared:
            share_img_proj = True
        elif has_decoupled:
            share_img_proj = False
        else:
            if int(cls_dim) == backbone_dim:
                raise RuntimeError(
                    f"Cannot infer projection topology for arch={arch}. "
                    "Use a preset or provide a checkpoint with projection keys."
                )
            share_img_proj = False

    if share_img_proj:
        align_dim = int(cls_dim)
    else:
        if "img_enc.align_proj.0.weight" in state_dict:
            align_dim = int(state_dict["img_enc.align_proj.0.weight"].shape[0])
        else:
            align_dim = backbone_dim

    return OursCheckpointConfig(
        arch=arch,
        n_classes=int(n_classes),
        cls_proj_dim=int(cls_dim),
        align_dim=int(align_dim),
        share_img_proj=bool(share_img_proj),
        head_type=head_type,
    )


def load_ours_model(
    checkpoint_path: Path,
    preset_name: str,
    n_classes: int,
    device: torch.device,
) -> OursImageOnlyModel:
    """Load the proposed model from a released preset checkpoint."""
    preset = build_preset(preset_name, seed=42)
    raw_state = load_checkpoint_state_dict(checkpoint_path, device)

    inferred = infer_ours_checkpoint_config(
        state_dict=raw_state,
        arch=preset.arch,
        share_img_proj=preset.share_img_proj,
    )

    if inferred.n_classes != n_classes:
        raise RuntimeError(
            f"Checkpoint has {inferred.n_classes} classes, but label2id has {n_classes} classes."
        )

    model = OursImageOnlyModel(
        arch=inferred.arch,
        n_classes=inferred.n_classes,
        cls_proj_dim=inferred.cls_proj_dim,
        align_dim=inferred.align_dim,
        share_img_proj=inferred.share_img_proj,
        head_type=inferred.head_type,
        head_scale=preset.head_scale,
        dropout=preset.dropout,
    )

    image_state = keep_image_only_state_dict(raw_state)
    model.load_state_dict(image_state, strict=True)
    model.to(device).eval()

    return model


# -------------------------------------------------------------------------
# Grad-CAM
# -------------------------------------------------------------------------
def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape Swin activations into CAM-compatible spatial maps."""
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2).contiguous()

    if tensor.dim() == 3:
        batch, length, channels = tensor.shape
        height = width = int(math.sqrt(length))
        return tensor.permute(0, 2, 1).reshape(batch, channels, height, width).contiguous()

    raise ValueError(f"Unexpected Swin tensor shape: {tuple(tensor.shape)}")


def get_target_layer(model: nn.Module, family: str, arch: str):
    """Select the target layer for Grad-CAM."""
    if family == "baseline":
        if arch == "resnet":
            return model.layer4[-1]
        if arch == "convnext":
            return model.features[-1][-1]
        if arch == "swin":
            return model.features[-1][-1].norm1

    if family == "ours":
        if arch == "resnet":
            return model.img_enc.backbone.layer4[-1]
        if arch == "convnext":
            return model.img_enc.backbone.features[-1][-1]
        if arch == "swin":
            return model.img_enc.backbone.features[-1][-1].norm1

    raise ValueError(f"Unsupported family/architecture combination: {family}, {arch}")


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """Predict the top-1 class."""
    logits = model(x)
    pred_id = int(logits.argmax(dim=1).item())
    return pred_id, logits


def run_gradcam(
    model: nn.Module,
    family: str,
    arch: str,
    x: torch.Tensor,
    target_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Grad-CAM and return RGB input, normalized CAM, and overlay."""
    target_layer = get_target_layer(model, family, arch)
    rgb = denorm_to_rgb(x)

    x_for_cam = x.clone().detach().requires_grad_(True)

    if arch == "swin":
        cam_obj = GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=swin_reshape_transform,
        )
    else:
        cam_obj = GradCAM(
            model=model,
            target_layers=[target_layer],
        )

    with cam_obj:
        grayscale_cam = cam_obj(
            input_tensor=x_for_cam,
            targets=[ClassifierOutputTarget(target_class)],
        )[0]

    grayscale_cam = normalize_cam_np(grayscale_cam)
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    return rgb, grayscale_cam, overlay


def save_overlay(overlay: np.ndarray, out_path: Path) -> None:
    """Save a CAM overlay image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a single-model Grad-CAM overlay.")

    parser.add_argument(
        "--family",
        choices=["ours", "baseline"],
        required=True,
        help="Model family to visualize.",
    )
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default=None,
        help="Preset name for the proposed model. Required when --family ours.",
    )
    parser.add_argument(
        "--arch",
        choices=["resnet", "convnext", "swin"],
        default=None,
        help="Backbone architecture for a baseline model. Required when --family baseline.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Path to the input specimen image.",
    )
    parser.add_argument(
        "--label2id",
        type=Path,
        default=None,
        help="Path to label2id JSON. If omitted, the default example label2id file is used.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save the Grad-CAM overlay.",
    )
    parser.add_argument(
        "--target_mode",
        choices=["pred", "gt"],
        default="pred",
        help="Explain the predicted class or the ground-truth class parsed from the filename.",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Optional explicit target class id. Overrides --target_mode.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.family == "ours" and args.preset is None:
        raise ValueError("--preset is required when --family ours.")

    if args.family == "baseline" and args.arch is None:
        raise ValueError("--arch is required when --family baseline.")

    image_path = resolve_repo_path(args.image)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    out_dir = resolve_repo_path(args.out_dir)

    if args.label2id is None:
        label2id_path = default_label2id_path(args.preset)
    else:
        label2id_path = resolve_repo_path(args.label2id)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not label2id_path.is_file():
        raise FileNotFoundError(f"label2id file not found: {label2id_path}")

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    label2id = load_json(label2id_path)
    id2label = id2label_from_label2id(label2id)
    n_classes = len(label2id)

    if args.family == "ours":
        preset = build_preset(args.preset, seed=42)
        arch = preset.arch
        model = load_ours_model(
            checkpoint_path=checkpoint_path,
            preset_name=args.preset,
            n_classes=n_classes,
            device=device,
        )
    else:
        arch = args.arch
        model = load_baseline_model(
            arch=arch,
            checkpoint_path=checkpoint_path,
            n_classes=n_classes,
            device=device,
        )

    x = load_input(image_path, device)
    pred_id, _ = predict(model, x)
    pred_name = id2label.get(pred_id, str(pred_id))

    gt_name = get_label_from_filename(image_path)
    gt_id = label2id.get(gt_name)

    if args.target_class is not None:
        target_class = int(args.target_class)
        target_name = id2label.get(target_class, str(target_class))
        target_source = "manual"
    elif args.target_mode == "gt":
        if gt_id is None:
            raise RuntimeError(
                f"Ground-truth label '{gt_name}' was not found in label2id. "
                "Use --target_mode pred or provide --target_class."
            )
        target_class = int(gt_id)
        target_name = gt_name
        target_source = "ground truth"
    else:
        target_class = pred_id
        target_name = pred_name
        target_source = "prediction"

    _, _, overlay = run_gradcam(
        model=model,
        family=args.family,
        arch=arch,
        x=x,
        target_class=target_class,
    )

    specimen_id = extract_specimen_id(image_path)
    clean_model_name = pretty_model_name(args.family, arch)
    clean_slug = model_file_slug(args.family, arch)
    out_path = out_dir / f"{clean_slug}_gradcam_{target_source.replace(' ', '_')}_{specimen_id}.png"

    save_overlay(overlay, out_path)

    print(f"[Done] Model: {clean_model_name}")
    print(f"[Done] Device: {device}")
    print(f"[Done] Image: {image_path}")
    print(f"[Done] Prediction: {pred_id} ({pred_name})")
    print(f"[Done] CAM target: {target_class} ({target_name})")
    print(f"[Done] Target source: {target_source}")
    print(f"[Done] Saved: {out_path}")


if __name__ == "__main__":
    main()