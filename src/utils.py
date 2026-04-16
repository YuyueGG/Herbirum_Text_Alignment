from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def load_label2id(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_better_model(
    curr_acc: float,
    curr_f1: float,
    curr_loss: float,
    best_acc: float,
    best_f1: float,
    best_loss: float,
    eps: float = 1e-6,
) -> bool:
    if curr_acc > best_acc + eps:
        return True
    if abs(curr_acc - best_acc) <= eps:
        if curr_f1 > best_f1 + eps:
            return True
        if abs(curr_f1 - best_f1) <= eps and curr_loss < best_loss - eps:
            return True
    return False


@torch.no_grad()
def macro_f1_from_preds(n_classes: int, preds: torch.Tensor, labels: torch.Tensor) -> float:
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for pred, label in zip(preds, labels):
        confusion[label, pred] += 1

    true_pos = confusion.diag()
    false_pos = confusion.sum(dim=0) - true_pos
    false_neg = confusion.sum(dim=1) - true_pos

    precision = true_pos / (true_pos + false_pos + 1e-8)
    recall = true_pos / (true_pos + false_neg + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    valid = confusion.sum(dim=1) > 0
    return f1[valid].mean().item() if valid.any() else 0.0


def to_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [to_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
