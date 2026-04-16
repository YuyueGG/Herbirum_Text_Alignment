#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualize_model_xai.py

Generate a single-model XAI panel for a herbarium specimen image.

Design goals:
- One model per run
- No hard-coded checkpoint paths
- Repository-relative path resolution
- Support both baseline and alignment models
- Paper-ready outputs with clean overlays

Supported model families:
1) alignment
   - Rebuilds the model from a released preset
   - Loads a checkpoint produced by this repository

2) baseline
   - Loads a pure vision checkpoint
   - Requires an explicit architecture name

Outputs:
- A PNG panel
- A PDF panel

The panel contains:
1) input image
2) grayscale activation map
3) CAM overlay
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms as T
from torchvision.io import read_image
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    convnext_base,
    resnet50,
    swin_t,
)

from src.model import HerbariumTextAlignmentModel
from src.presets import PRESET_NAMES, build_preset


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = REPO_ROOT / "examples" / "sample_images" / "1.png"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "xai"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DARK_GRAY = "#444444"


class ResizeShortSideAndPad:
    """Resize the shorter side to the target size and pad to a square canvas."""

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

def pretty_arch_name(arch: str) -> str:
    """Convert an internal architecture name into a display-friendly name."""
    mapping = {
        "resnet": "ResNet",
        "convnext": "ConvNeXt",
        "swin": "Swin",
    }
    return mapping.get(arch.lower(), arch)


def pretty_model_name(family: str, arch: str) -> str:
    """Build a user-facing model name without internal project acronyms."""
    arch_name = pretty_arch_name(arch)

    if family == "alignment":
        return f"{arch_name} Ours"

    if family == "baseline":
        return f"{arch_name} Baseline"

    return arch_name

def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def id2label_from_label2id(label2id: dict) -> Dict[int, str]:
    """Convert label2id into id2label."""
    return {int(v): k for k, v in label2id.items()}


def extract_specimen_id(image_path: Path) -> str:
    """Extract a stable specimen identifier from the image filename."""
    stem = image_path.stem
    if "__" in stem:
        return stem.split("__")[-1]
    return stem


def get_label_from_filename(image_path: Path) -> str:
    """Parse genus_species from the image filename."""
    stem = image_path.stem
    main = stem.split("__")[0]
    parts = main.split("_")
    if len(parts) >= 2:
        genus = parts[0].lower()
        species = parts[1].lower()
        return f"{genus}_{species}"
    return stem.lower()


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a leading 'module.' prefix if present."""
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def extract_state_dict(checkpoint_obj) -> Dict[str, torch.Tensor]:
    """Extract a model state dict from a checkpoint object."""
    if isinstance(checkpoint_obj, dict):
        if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
            return strip_module_prefix(checkpoint_obj["model"])
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return strip_module_prefix(checkpoint_obj["state_dict"])
        if all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return strip_module_prefix(checkpoint_obj)
    raise RuntimeError("No valid state_dict was found in the checkpoint file.")


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load and normalize a checkpoint state dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return extract_state_dict(checkpoint)


def normalize_cam_np(cam: np.ndarray) -> np.ndarray:
    """Normalize a CAM into the range [0, 1]."""
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam


def denorm_to_rgb(x: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor image into an RGB NumPy image."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    image = x.detach().clone() * std + mean
    image = image.clamp(0, 1)
    image = image[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return image


def build_alignment_transform():
    """Build the evaluation transform used by alignment models."""
    return T.Compose(
        [
            ResizeShortSideAndPad(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_baseline_tensor_transform():
    """Build the exact tensor-space evaluation transform used by baseline models."""
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_alignment_input(image_path: Path, device: torch.device) -> torch.Tensor:
    """Load an input image using the alignment-model evaluation pipeline."""
    image = Image.open(image_path).convert("RGB")
    x = build_alignment_transform()(image)
    return x.unsqueeze(0).to(device)


def load_baseline_input(image_path: Path, device: torch.device) -> torch.Tensor:
    """Load an input image using the baseline evaluation pipeline."""
    x = read_image(str(image_path)).float() / 255.0
    x = build_baseline_tensor_transform()(x)
    return x.unsqueeze(0).to(device)


def load_baseline_model(
    arch: str,
    checkpoint_path: Path,
    n_classes: int,
    device: torch.device,
) -> nn.Module:
    """Load a baseline checkpoint."""
    arch = arch.lower()

    if arch == "resnet":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "convnext":
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, n_classes)
    elif arch == "swin":
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, n_classes)
    else:
        raise ValueError(f"Unsupported baseline architecture: {arch}")

    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def load_alignment_model(
    preset_name: str,
    checkpoint_path: Path,
    n_classes: int,
    device: torch.device,
) -> HerbariumTextAlignmentModel:
    """Load an alignment-model checkpoint from a released preset."""
    config = build_preset(preset_name, seed=42)

    model = HerbariumTextAlignmentModel(
        arch=config.arch,
        n_classes=n_classes,
        cls_proj_dim=config.cls_proj_dim,
        align_dim=config.align_dim,
        dropout=config.dropout,
        head_type=config.head_type,
        head_scale=config.head_scale,
        freeze_img_low=config.freeze_img_low,
        freeze_txt_low=config.freeze_txt_low,
        freeze_txt_all=config.freeze_txt_all,
        img_train_from=config.img_train_from,
        temp_init=config.temp_init,
        learnable_temperature=config.learnable_temperature,
        share_img_proj=config.share_img_proj,
        text_model=config.text_model,
        text_pool=config.text_pool,
    )

    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_no_tta(model: nn.Module, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """Run a forward pass without test-time augmentation."""
    outputs = model(x)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    pred = int(logits.argmax(dim=1).item())
    return pred, logits


@torch.no_grad()
def predict_with_hflip_tta(model: nn.Module, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """Run horizontal-flip test-time augmentation and average logits."""
    outputs_1 = model(x)
    logits_1 = outputs_1[0] if isinstance(outputs_1, tuple) else outputs_1

    x_flip = torch.flip(x, dims=[-1])
    outputs_2 = model(x_flip)
    logits_2 = outputs_2[0] if isinstance(outputs_2, tuple) else outputs_2

    logits = (logits_1 + logits_2) / 2.0
    pred = int(logits.argmax(dim=1).item())
    return pred, logits


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape Swin features into CAM-compatible spatial maps."""
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2).contiguous()

    if tensor.dim() == 3:
        batch, length, channels = tensor.shape
        height = width = int(math.sqrt(length))
        return tensor.permute(0, 2, 1).reshape(batch, channels, height, width).contiguous()

    raise ValueError(f"Unexpected Swin tensor shape: {tuple(tensor.shape)}")


def get_target_layer(model: nn.Module, family: str, arch: str):
    """Select a suitable CAM target layer."""
    if family == "baseline":
        if arch == "resnet":
            return model.layer4[-1]
        if arch == "convnext":
            return model.features[-1][-1]
        if arch == "swin":
            return model.features[-1][-1].norm1

    if family == "alignment":
        if arch == "resnet":
            return model.img_enc.backbone.layer4[-1]
        if arch == "convnext":
            return model.img_enc.backbone.features[-1][-1]
        if arch == "swin":
            return model.img_enc.backbone.features[-1][-1].norm1

    raise ValueError(f"Unsupported family/arch combination: {family}, {arch}")


def run_cnn_cam(
    model: nn.Module,
    x: torch.Tensor,
    target_layer: nn.Module,
    target_class: int,
    use_tta: bool,
):
    """Run HiResCAM for CNN-based models."""
    rgb = denorm_to_rgb(x)

    if not use_tta:
        x_single = x.clone().detach().requires_grad_(True)
        with HiResCAM(model=model, target_layers=[target_layer]) as cam_obj:
            cam = cam_obj(
                input_tensor=x_single,
                targets=[ClassifierOutputTarget(target_class)],
            )[0]
        cam = normalize_cam_np(cam)
        overlay = show_cam_on_image(rgb, cam, use_rgb=True)
        return rgb, cam, overlay

    x_1 = x.clone().detach().requires_grad_(True)
    x_2 = torch.flip(x, dims=[-1]).clone().detach().requires_grad_(True)

    with HiResCAM(model=model, target_layers=[target_layer]) as cam_obj_1:
        cam_1 = cam_obj_1(
            input_tensor=x_1,
            targets=[ClassifierOutputTarget(target_class)],
        )[0]

    with HiResCAM(model=model, target_layers=[target_layer]) as cam_obj_2:
        cam_2 = cam_obj_2(
            input_tensor=x_2,
            targets=[ClassifierOutputTarget(target_class)],
        )[0]

    cam_2 = np.flip(cam_2, axis=1)
    cam = normalize_cam_np((cam_1 + cam_2) / 2.0)
    overlay = show_cam_on_image(rgb, cam, use_rgb=True)
    return rgb, cam, overlay


def run_swin_cam(
    model: nn.Module,
    x: torch.Tensor,
    target_layer: nn.Module,
    target_class: int,
    use_tta: bool,
):
    """Run GradCAM for Swin-based models."""
    rgb = denorm_to_rgb(x)

    if not use_tta:
        x_single = x.clone().detach().requires_grad_(True)
        with GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=swin_reshape_transform,
        ) as cam_obj:
            cam = cam_obj(
                input_tensor=x_single,
                targets=[ClassifierOutputTarget(target_class)],
            )[0]
        cam = normalize_cam_np(cam)
        overlay = show_cam_on_image(rgb, cam, use_rgb=True)
        return rgb, cam, overlay

    x_1 = x.clone().detach().requires_grad_(True)
    x_2 = torch.flip(x, dims=[-1]).clone().detach().requires_grad_(True)

    with GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=swin_reshape_transform,
    ) as cam_obj_1:
        cam_1 = cam_obj_1(
            input_tensor=x_1,
            targets=[ClassifierOutputTarget(target_class)],
        )[0]

    with GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=swin_reshape_transform,
    ) as cam_obj_2:
        cam_2 = cam_obj_2(
            input_tensor=x_2,
            targets=[ClassifierOutputTarget(target_class)],
        )[0]

    cam_2 = np.flip(cam_2, axis=1)
    cam = normalize_cam_np((cam_1 + cam_2) / 2.0)
    overlay = show_cam_on_image(rgb, cam, use_rgb=True)
    return rgb, cam, overlay


def run_cam(
    model: nn.Module,
    family: str,
    arch: str,
    x: torch.Tensor,
    target_class: int,
    use_tta: bool,
):
    """Dispatch CAM generation by backbone family."""
    target_layer = get_target_layer(model, family, arch)

    if arch in ("resnet", "convnext"):
        return run_cnn_cam(
            model=model,
            x=x,
            target_layer=target_layer,
            target_class=target_class,
            use_tta=use_tta,
        )

    if arch == "swin":
        return run_swin_cam(
            model=model,
            x=x,
            target_layer=target_layer,
            target_class=target_class,
            use_tta=use_tta,
        )

    raise ValueError(f"Unsupported architecture: {arch}")


def save_xai_panel(
    rgb: np.ndarray,
    cam: np.ndarray,
    overlay: np.ndarray,
    title_input: str,
    title_cam: str,
    title_overlay: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Save a three-panel XAI figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(rgb)
    axes[0].set_title(title_input, fontsize=10, color=DARK_GRAY)
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title(title_cam, fontsize=10, color=DARK_GRAY)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(title_overlay, fontsize=10, color=DARK_GRAY)
    axes[2].axis("off")

    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a single-model XAI panel.")

    parser.add_argument(
        "--family",
        choices=["alignment", "baseline"],
        required=True,
        help="Model family to visualize.",
    )
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default=None,
        help="Preset name for an alignment model.",
    )
    parser.add_argument(
        "--arch",
        choices=["resnet", "convnext", "swin"],
        default=None,
        help="Backbone architecture for a baseline model.",
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
        help="Path to label2id JSON. If omitted, a repository example path is selected automatically.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save the XAI panel.",
    )
    parser.add_argument(
        "--target_mode",
        choices=["pred", "gt"],
        default="pred",
        help="Explain the predicted class or the ground-truth class parsed from the filename.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    parser.add_argument(
        "--disable_tta",
        action="store_true",
        help="Disable test-time augmentation even if the alignment preset normally uses it.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.family == "alignment" and args.preset is None:
        raise ValueError("--preset is required when --family alignment is used.")

    if args.family == "baseline" and args.arch is None:
        raise ValueError("--arch is required when --family baseline is used.")

    image_path = resolve_repo_path(args.image)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    output_dir = resolve_repo_path(args.output_dir)

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

    if args.family == "alignment":
        config = build_preset(args.preset, seed=42)
        arch = config.arch
        use_tta = bool(config.tta) and not args.disable_tta
        model = load_alignment_model(
            preset_name=args.preset,
            checkpoint_path=checkpoint_path,
            n_classes=n_classes,
            device=device,
        )
        x = load_alignment_input(image_path, device)
    else:
        arch = args.arch
        use_tta = False
        model = load_baseline_model(
            arch=arch,
            checkpoint_path=checkpoint_path,
            n_classes=n_classes,
            device=device,
        )
        x = load_baseline_input(image_path, device)

    model_display_name = pretty_model_name(args.family, arch)
    model_file_name = f"{arch}_{'ours' if args.family == 'alignment' else 'baseline'}"

    gt_name = get_label_from_filename(image_path)
    gt_id = label2id.get(gt_name, None)

    if use_tta:
        pred_id, _ = predict_with_hflip_tta(model, x)
    else:
        pred_id, _ = predict_no_tta(model, x)

    if args.target_mode == "gt":
        if gt_id is None:
            raise RuntimeError(f"Ground-truth label '{gt_name}' was not found in label2id.")
        target_id = gt_id
    else:
        target_id = pred_id

    rgb, cam, overlay = run_cam(
        model=model,
        family=args.family,
        arch=arch,
        x=x,
        target_class=target_id,
        use_tta=use_tta,
    )

    specimen_id = extract_specimen_id(image_path)
    output_stem = f"{model_file_name}_{args.target_mode}_{specimen_id}"
    out_png = output_dir / f"{output_stem}.png"
    out_pdf = output_dir / f"{output_stem}.pdf"

    pred_text = id2label.get(pred_id, str(pred_id))
    target_text = id2label.get(target_id, str(target_id))
    gt_text = gt_name if gt_id is not None else "N/A"

    title_input = "Input image"
    title_cam = f"CAM | target={target_text}"
    title_overlay = f"{model_display_name} | pred={pred_text} | gt={gt_text}"

    save_xai_panel(
        rgb=rgb,
        cam=cam,
        overlay=overlay,
        title_input=title_input,
        title_cam=title_cam,
        title_overlay=title_overlay,
        out_png=out_png,
        out_pdf=out_pdf,
    )

    print(f"[Done] Model: {model_display_name}")
    print(f"[Done] Device: {device}")
    print(f"[Done] Prediction: {pred_id} ({pred_text})")
    print(f"[Done] CAM target: {target_id} ({target_text})")
    print(f"[Done] Ground truth: {gt_text}")
    print(f"[Done] Saved PNG: {out_png}")
    print(f"[Done] Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()