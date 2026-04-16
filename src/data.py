from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from .transforms import build_transforms
from .utils import seed_worker


class MMJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        transform,
        max_len: int = 64,
        n_classes: Optional[int] = None,
    ):
        self.rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

        if n_classes is not None:
            for row in self.rows:
                label_id = int(row["label_id"])
                if not 0 <= label_id < n_classes:
                    raise ValueError(f"label_id out of range: {label_id}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = Image.open(row["image"]).convert("RGB")
        pixel_values = self.transform(image)

        text = (row.get("text") or "").strip()
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }


def build_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def count_labels(rows: list[dict]) -> Counter:
    return Counter(int(row["label_id"]) for row in rows)


def build_train_sampler(rows: list[dict], n_classes: int) -> WeightedRandomSampler:
    label_counter = count_labels(rows)
    class_counts = torch.tensor([label_counter.get(class_id, 0) for class_id in range(n_classes)], dtype=torch.float)
    class_counts[class_counts == 0] = 1.0
    sample_weights = [1.0 / class_counts[int(row["label_id"])] for row in rows]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_dataloaders(
    train_jsonl: Path,
    test_jsonl: Path,
    tokenizer_name: str,
    batch_size: int,
    max_len: int,
    n_classes: int,
    num_workers: int,
    augment: bool,
    balanced_sampler: bool,
):
    tokenizer = build_tokenizer(tokenizer_name)

    train_dataset = MMJsonlDataset(
        jsonl_path=train_jsonl,
        tokenizer=tokenizer,
        transform=build_transforms(augment=augment),
        max_len=max_len,
        n_classes=n_classes,
    )
    val_dataset = MMJsonlDataset(
        jsonl_path=test_jsonl,
        tokenizer=tokenizer,
        transform=build_transforms(augment=False),
        max_len=max_len,
        n_classes=n_classes,
    )

    sampler = build_train_sampler(train_dataset.rows, n_classes) if balanced_sampler else None
    shuffle = sampler is None
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=persistent_workers,
    )

    return tokenizer, train_dataset, val_dataset, train_loader, val_loader
