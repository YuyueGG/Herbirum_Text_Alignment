from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def ce_per_sample(logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    if smoothing > 0.0:
        num_classes = logits.size(-1)
        with torch.no_grad():
            target = torch.full_like(logits, smoothing / (num_classes - 1))
            target.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        log_prob = F.log_softmax(logits, dim=1)
        return (-target * log_prob).sum(dim=1)
    return F.cross_entropy(logits, labels, reduction="none")


def class_balanced_weights(counts: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    effective_num = (1 - beta) / (1 - torch.pow(beta, counts.clamp(min=1.0)))
    weights = effective_num / effective_num.sum() * len(counts)
    return weights


def grad_scale_trick(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale >= 0.999:
        return x
    return x * scale + x.detach() * (1.0 - scale)


def align_weight(epoch: int, w_align: float, warmup_epochs: int, ramp_epochs: int) -> float:
    if w_align <= 0.0:
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs > 0:
        progress = min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))
        return w_align * progress
    return w_align


def inst_infonce_loss(
    img_z: torch.Tensor,
    txt_z: torch.Tensor,
    temperature: torch.Tensor,
    dcl: bool = False,
) -> torch.Tensor:
    num_samples = img_z.size(0)
    if num_samples <= 1:
        return img_z.new_tensor(0.0)

    img = F.normalize(img_z.float(), dim=1)
    txt = F.normalize(txt_z.float(), dim=1)
    logits = (img @ txt.t()) / temperature
    targets = torch.arange(num_samples, device=logits.device)

    if not dcl:
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))

    eye = torch.eye(num_samples, device=logits.device, dtype=torch.bool)
    logits_neg = logits.masked_fill(eye, float("-inf"))
    positives = logits.diag()
    loss_i = (torch.logsumexp(logits_neg, dim=1) - positives).mean()

    logits_t = logits.t()
    logits_neg_t = logits_t.masked_fill(eye, float("-inf"))
    loss_t = (torch.logsumexp(logits_neg_t, dim=1) - positives).mean()

    return 0.5 * (loss_i + loss_t)


@torch.no_grad()
def compute_text_prototypes(
    txt_enc,
    tokenizer,
    rows: list[dict],
    n_classes: int,
    device: torch.device,
    max_len: int = 64,
    min_text_tokens: int = 5,
    max_per_class: int = 64,
    batch_size: int = 64,
    amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts_by_class = [[] for _ in range(n_classes)]
    for row in rows:
        label_id = int(row["label_id"])
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if len(texts_by_class[label_id]) < max_per_class:
            texts_by_class[label_id].append(text)

    pairs: list[tuple[int, str]] = []
    for class_id, texts in enumerate(texts_by_class):
        for text in texts:
            pairs.append((class_id, text))

    prototype_sum = torch.zeros(n_classes, txt_enc.out_dim, device=device, dtype=torch.float32)
    counts = torch.zeros(n_classes, device=device, dtype=torch.long)

    if not pairs:
        valid = torch.zeros(n_classes, device=device, dtype=torch.bool)
        return prototype_sum, valid

    txt_enc.eval()
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        labels = torch.tensor([item[0] for item in chunk], device=device, dtype=torch.long)
        texts = [item[1] for item in chunk]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        token_lengths = attention_mask.sum(dim=1)
        keep = token_lengths >= min_text_tokens
        if keep.sum().item() == 0:
            continue

        with torch.amp.autocast(amp_device, enabled=(amp and amp_device == "cuda")):
            features = txt_enc(input_ids[keep], attention_mask[keep])
        features = F.normalize(features.float(), dim=1)

        labels = labels[keep]
        prototype_sum.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))

    valid = counts > 0
    prototypes = torch.zeros_like(prototype_sum)
    if valid.any():
        prototypes[valid] = prototype_sum[valid] / counts[valid].unsqueeze(1).float()
    prototypes = F.normalize(prototypes, dim=1)
    return prototypes, valid


@torch.no_grad()
def _proto_sim_weights(proto: torch.Tensor, labels: torch.Tensor, alpha: float) -> Optional[torch.Tensor]:
    if alpha <= 0.0:
        return None
    sim_matrix = (proto @ proto.t()).clamp(-1.0, 1.0)
    return torch.exp(alpha * sim_matrix.index_select(0, labels))


def proto_softmax_loss(
    img_z: torch.Tensor,
    labels: torch.Tensor,
    proto: torch.Tensor,
    valid_class: torch.Tensor,
    temperature: torch.Tensor,
    dcl: bool = False,
    hard_neg_alpha: float = 0.0,
) -> torch.Tensor:
    if img_z.numel() == 0:
        return img_z.new_tensor(0.0)

    img = F.normalize(img_z.float(), dim=1)
    proto = F.normalize(proto.float(), dim=1)
    logits = (img @ proto.t()) / temperature

    if valid_class is not None:
        invalid = ~valid_class
        if invalid.any():
            logits = logits.masked_fill(invalid.unsqueeze(0), float("-inf"))

    neg_weights = _proto_sim_weights(proto, labels, hard_neg_alpha)

    if not dcl:
        if neg_weights is None:
            return F.cross_entropy(logits, labels)
        log_num = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        exp_logits = torch.exp(logits) * neg_weights
        denom = torch.log(exp_logits.sum(dim=1).clamp(min=1e-12))
        return (denom - log_num).mean()

    log_pos = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    logits_neg = logits.clone()
    logits_neg.scatter_(1, labels.unsqueeze(1), float("-inf"))

    if neg_weights is None:
        denom = torch.logsumexp(logits_neg, dim=1)
        return (denom - log_pos).mean()

    exp_neg = torch.exp(logits_neg) * neg_weights
    denom = torch.log(exp_neg.sum(dim=1).clamp(min=1e-12))
    return (denom - log_pos).mean()
