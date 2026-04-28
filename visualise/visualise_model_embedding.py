#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualize_model_embedding.py

Visualize embeddings for a single model from a JSONL dataset.

Design goals:
- No hard-coded checkpoint paths
- Repository-relative path resolution
- One model per run
- Compatible with both baseline and alignment models
- Clean outputs for paper figures or qualitative analysis

Supported model families:
1) alignment
   - Uses a released training preset to rebuild the model
   - Loads a checkpoint produced by this repository

2) baseline
   - Loads a pure vision backbone checkpoint
   - Requires an architecture name

Input JSONL format:
Each line should contain at least:
{
    "image": "path/to/image.jpg",
    "label_id": 0
}

The script saves both PNG and PDF outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
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


class JsonlImageDataset(Dataset):
    """Read images and labels from a JSONL file."""

    def __init__(self, rows: List[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label_id"])
        return image, label


class FeatureCapture:
    """Capture the tensor passed into a classifier layer."""

    def __init__(self, module: torch.nn.Module):
        self.feature = None
        self.hook = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x = inputs[0]
        if isinstance(x, torch.Tensor):
            self.feature = x.detach()

    def close(self):
        try:
            self.hook.remove()
        except Exception:
            pass


def resolve_repo_path(path_value: Path) -> Path:
    """Resolve a path relative to the repository root."""
    if path_value.is_absolute():
        return path_value
    return (REPO_ROOT / path_value).resolve()


def default_label2id_path(preset: Optional[str]) -> Path:
    """Choose the default label2id path from the repository examples."""
    if preset is not None and preset.endswith("9"):
        return REPO_ROOT / "examples" / "sample_jsonl" / "label2id9.json"
    return REPO_ROOT / "examples" / "sample_jsonl" / "label2id.json"


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_baseline_transform():
    """Build the evaluation transform used by baseline models."""
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_alignment_transform():
    """Build the evaluation transform used by alignment models."""
    return T.Compose(
        [
            ResizeShortSideAndPad(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_jsonl_rows(jsonl_path: Path) -> List[dict]:
    """Load rows from a JSONL file."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def filter_rows_by_class_count(rows: List[dict], min_class_count: int) -> List[dict]:
    """Keep only classes with at least the requested number of samples."""
    counts = Counter(int(row["label_id"]) for row in rows)
    kept_labels = {label for label, count in counts.items() if count >= int(min_class_count)}
    return [row for row in rows if int(row["label_id"]) in kept_labels]


def subsample_rows_by_class(rows: List[dict], max_per_class: int, seed: int) -> List[dict]:
    """Optionally subsample each class for faster plotting."""
    if max_per_class <= 0:
        return rows

    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        buckets[int(row["label_id"])].append(row)

    sampled_rows: List[dict] = []
    for label_id in sorted(buckets):
        class_rows = buckets[label_id]
        if len(class_rows) <= max_per_class:
            sampled_rows.extend(class_rows)
        else:
            selected_indices = rng.choice(len(class_rows), size=max_per_class, replace=False)
            sampled_rows.extend([class_rows[i] for i in sorted(selected_indices.tolist())])

    return sampled_rows


def remap_labels_to_compact(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remap arbitrary label ids to a compact range 0..K-1."""
    unique_labels = np.unique(labels).astype(np.int64)
    mapping = {int(old): int(new) for new, old in enumerate(unique_labels.tolist())}
    remapped = np.array([mapping[int(label)] for label in labels.tolist()], dtype=np.int64)
    return remapped, unique_labels


def build_vivid_color_table(n_classes: int, seed: int, counts: Optional[np.ndarray] = None) -> np.ndarray:
    """Build a vivid color table for class visualization."""
    if n_classes <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    hues = np.linspace(0.0, 1.0, n_classes, endpoint=False).astype(np.float32)
    colors = np.zeros((n_classes, 3), dtype=np.float32)

    for i, hue in enumerate(hues):
        colors[i] = mpl.colors.hsv_to_rgb((float(hue), 1.0, 0.98))

    rng = np.random.default_rng(seed)
    colors = colors[rng.permutation(n_classes)]

    if counts is not None and counts.size == n_classes and counts.max() > 0:
        weights = counts.astype(np.float32) / (counts.max() + 1e-12)
        weights = np.power(weights, 0.6)

        adjusted = np.zeros_like(colors)
        for i in range(n_classes):
            r, g, b = colors[i]
            h, s, v = mpl.colors.rgb_to_hsv((r, g, b))
            s_new = 0.80 + (1.0 - 0.80) * float(weights[i])
            v_new = 0.92 + (1.0 - 0.92) * float(weights[i])
            adjusted[i] = mpl.colors.hsv_to_rgb((h, np.clip(s_new, 0, 1), np.clip(v_new, 0, 1)))
        colors = adjusted

    return np.clip(colors, 0, 1)

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
        return f"{arch_name} VLM-KD"

    if family == "baseline":
        return f"{arch_name} Baseline"

    return arch_name

def set_square_limits(ax, embedding: np.ndarray, pad: float = 0.06) -> None:
    """Set square axes for cleaner visual comparison."""
    x = embedding[:, 0]
    y = embedding[:, 1]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min) * (1.0 + pad)

    ax.set_xlim(x_mid - 0.5 * span, x_mid + 0.5 * span)
    ax.set_ylim(y_mid - 0.5 * span, y_mid + 0.5 * span)
    ax.set_box_aspect(1)


def reduce_to_2d(
    features: np.ndarray,
    method: str,
    seed: int,
    pca_dim: int,
    perplexity: float,
) -> np.ndarray:
    """Reduce embeddings to 2D for visualization."""
    if features.shape[0] < 3:
        raise ValueError("At least three samples are required for dimensionality reduction.")

    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)

    if pca_dim > 0 and features.shape[1] > pca_dim:
        features = PCA(n_components=int(pca_dim), random_state=seed).fit_transform(features)

    method = method.lower()
    if method == "pca2":
        return PCA(n_components=2, random_state=seed).fit_transform(features)

    if method == "umap":
        try:
            import umap  # type: ignore

            return umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=seed,
            ).fit_transform(features)
        except Exception:
            method = "tsne"

    n_samples = features.shape[0]
    effective_perplexity = float(perplexity)
    if effective_perplexity >= n_samples:
        effective_perplexity = max(5.0, (n_samples - 1.0) / 3.0)
        effective_perplexity = min(effective_perplexity, 50.0)

    tsne_kwargs = dict(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        random_state=seed,
        learning_rate="auto",
    )

    try:
        reducer = TSNE(max_iter=1000, **tsne_kwargs)
    except TypeError:
        reducer = TSNE(n_iter=1000, **tsne_kwargs)

    return reducer.fit_transform(features)


def plot_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    color_table: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    title: str,
    plot_centers: bool,
) -> None:
    """Create and save a single-model embedding plot."""
    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(DARK_GRAY)

    set_square_limits(ax, embedding, pad=0.06)

    if not plot_centers:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=color_table[labels],
            s=16,
            alpha=0.92,
            linewidths=0.35,
            edgecolors="white",
            rasterized=True,
        )
    else:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=12,
            alpha=0.22,
            c="#9a9a9a",
            linewidths=0,
            rasterized=True,
        )

        unique_labels = np.unique(labels)
        centers = np.zeros((unique_labels.size, 2), dtype=np.float32)
        center_colors = np.zeros((unique_labels.size, 3), dtype=np.float32)

        for idx, class_id in enumerate(unique_labels):
            class_points = embedding[labels == class_id]
            centers[idx] = class_points.mean(axis=0)
            center_colors[idx] = color_table[int(class_id)]

        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=20,
            alpha=0.98,
            c=center_colors,
            linewidths=0.6,
            edgecolors="white",
            zorder=5,
        )

    ax.set_title(title, fontsize=16, color=DARK_GRAY)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def clean_state_dict(state_dict: dict) -> dict:
    """Remove a leading 'module.' prefix if present."""
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    """Load a checkpoint and return the model state dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    return clean_state_dict(state_dict)


def load_baseline_model(
    arch: str,
    checkpoint_path: Path,
    n_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load a baseline model checkpoint."""
    arch = arch.lower()

    if arch == "resnet":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    elif arch == "convnext":
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, n_classes)
    elif arch == "swin":
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, n_classes)
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
    """Load an alignment model checkpoint from a released preset."""
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
def extract_baseline_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    arch: str,
    use_tta: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract baseline pre-logit features."""
    if arch == "resnet":
        capture = FeatureCapture(model.fc)
    elif arch == "convnext":
        capture = FeatureCapture(model.classifier[-1])
    elif arch == "swin":
        capture = FeatureCapture(model.head)
    else:
        raise ValueError(f"Unsupported baseline architecture: {arch}")

    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        _ = model(images)
        feat_1 = capture.feature

        if use_tta:
            _ = model(torch.flip(images, dims=[-1]))
            feat_2 = capture.feature
            features = 0.5 * (feat_1 + feat_2)
        else:
            features = feat_1

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy().astype(np.int64))

    capture.close()

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return features_np, labels_np


@torch.no_grad()
def extract_alignment_features(
    model: HerbariumTextAlignmentModel,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract classifier-space features from an alignment model."""
    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        _, cls_feat_1, _, _ = model(images)

        if use_tta:
            _, cls_feat_2, _, _ = model(torch.flip(images, dims=[-1]))
            features = 0.5 * (cls_feat_1 + cls_feat_2)
        else:
            features = cls_feat_1

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy().astype(np.int64))

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return features_np, labels_np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize embeddings for one model.")

    parser.add_argument(
        "--family",
        choices=["alignment", "baseline"],
        required=True,
        help="Model family to evaluate.",
    )
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default=None,
        help="Preset name for an alignment model. Not used for baseline models.",
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
        "--data_jsonl",
        type=Path,
        required=True,
        help="Path to the JSONL file used for embedding extraction.",
    )
    parser.add_argument(
        "--label2id",
        type=Path,
        default=None,
        help="Path to the label2id JSON file. If omitted, a repository example path is selected automatically.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("outputs/embedding_visualization/model_embedding"),
        help="Output file prefix. Both PNG and PDF will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--min_class_count",
        type=int,
        default=5,
        help="Keep only classes with at least this many samples.",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="Optional per-class subsampling limit. Set 0 to disable.",
    )
    parser.add_argument(
        "--reduce",
        choices=["pca2", "tsne", "umap"],
        default="tsne",
        help="Dimensionality reduction method.",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=50,
        help="Optional PCA dimension before t-SNE or UMAP.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable horizontal-flip test-time augmentation during feature extraction.",
    )
    parser.add_argument(
        "--plot_centers",
        action="store_true",
        help="Plot class centers on top of gray points instead of plotting all points directly by class color.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.family == "alignment" and args.preset is None:
        raise ValueError("--preset is required when --family alignment is used.")

    if args.family == "baseline" and args.arch is None:
        raise ValueError("--arch is required when --family baseline is used.")

    checkpoint_path = resolve_repo_path(args.checkpoint)
    data_jsonl_path = resolve_repo_path(args.data_jsonl)
    output_prefix = resolve_repo_path(args.output_prefix)

    if args.label2id is None:
        label2id_path = default_label2id_path(args.preset)
    else:
        label2id_path = resolve_repo_path(args.label2id)

    if args.family == "alignment":
        effective_arch = build_preset(args.preset, seed=args.seed).arch
    else:
        effective_arch = args.arch

    device = torch.device(
        args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"
    )

    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    n_classes = len(label2id)

    rows = load_jsonl_rows(data_jsonl_path)
    rows = filter_rows_by_class_count(rows, args.min_class_count)
    rows = subsample_rows_by_class(rows, args.max_per_class, args.seed)

    if len(rows) == 0:
        raise RuntimeError("No samples remain after filtering.")

    if args.family == "alignment":
        transform = build_alignment_transform()
        model = load_alignment_model(
            preset_name=args.preset,
            checkpoint_path=checkpoint_path,
            n_classes=n_classes,
            device=device,
        )
    else:
        transform = build_baseline_transform()
        model = load_baseline_model(
            arch=effective_arch,
            checkpoint_path=checkpoint_path,
            n_classes=n_classes,
            device=device,
        )

    dataset = JsonlImageDataset(rows=rows, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if args.family == "alignment":
        features, labels = extract_alignment_features(
            model=model,
            loader=loader,
            device=device,
            use_tta=args.tta,
        )
    else:
        features, labels = extract_baseline_features(
            model=model,
            loader=loader,
            device=device,
            arch=effective_arch,
            use_tta=args.tta,
        )

    labels, kept_labels = remap_labels_to_compact(labels)
    class_counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    color_table = build_vivid_color_table(
        n_classes=int(labels.max()) + 1,
        seed=args.seed,
        counts=class_counts,
    )

    embedding_2d = reduce_to_2d(
        features=features,
        method=args.reduce,
        seed=args.seed,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
    )

    title = pretty_model_name(args.family, effective_arch)

    model_slug = f"{effective_arch}_{'VLM-KD' if args.family == 'alignment' else 'baseline'}"
    output_prefix = output_prefix.parent / f"{output_prefix.stem}_{model_slug}"
    out_png = output_prefix.with_suffix(".png")
    out_pdf = output_prefix.with_suffix(".pdf")

    plot_embedding(
        embedding=embedding_2d,
        labels=labels,
        color_table=color_table,
        out_png=out_png,
        out_pdf=out_pdf,
        title=title,
        plot_centers=args.plot_centers,
    )

    print(f"[Done] Saved PNG: {out_png}")
    print(f"[Done] Saved PDF: {out_pdf}")
    print(f"[Info] Kept classes: {len(kept_labels)}")
    print(f"[Info] Samples plotted: {len(labels)}")


if __name__ == "__main__":
    main()