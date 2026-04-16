#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from src.presets import PRESET_NAMES, build_preset
from src.trainer import train_model


REPO_ROOT = Path(__file__).resolve().parent


def resolve_repo_path(path_value: Path) -> Path:
    """Resolve a path relative to the repository root."""
    if path_value.is_absolute():
        return path_value
    return (REPO_ROOT / path_value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a preset herbarium text-alignment model."
    )
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        required=True,
        help="Preset name.",
    )
    parser.add_argument(
        "--train_jsonl",
        type=Path,
        required=True,
        help="Path to the training JSONL file. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--test_jsonl",
        type=Path,
        required=True,
        help="Path to the validation/test JSONL file. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--label2id",
        type=Path,
        required=True,
        help="Path to the label2id JSON file. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory used to save logs and checkpoints. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_jsonl = resolve_repo_path(args.train_jsonl)
    test_jsonl = resolve_repo_path(args.test_jsonl)
    label2id_path = resolve_repo_path(args.label2id)
    save_dir = resolve_repo_path(args.save_dir)

    config = build_preset(args.preset, seed=args.seed)

    train_model(
        config=config,
        train_jsonl=train_jsonl,
        test_jsonl=test_jsonl,
        label2id_path=label2id_path,
        save_dir=save_dir,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()