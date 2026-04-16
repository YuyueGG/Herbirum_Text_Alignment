from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import build_dataloaders, count_labels
from .losses import (
    align_weight,
    ce_per_sample,
    class_balanced_weights,
    compute_text_prototypes,
    grad_scale_trick,
    inst_infonce_loss,
    proto_softmax_loss,
)
from .model import HerbariumTextAlignmentModel
from .presets import TrainingConfig
from .utils import (
    append_jsonl,
    is_better_model,
    load_label2id,
    macro_f1_from_preds,
    seed_everything,
    to_jsonable,
    write_json,
)


@torch.no_grad()
def evaluate(model, loader, device: torch.device, label_smoothing: float, tta: bool) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    preds_all = []
    labels_all = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if not tta:
            logits, _, _, _ = model(pixel_values)
        else:
            logits_1, _, _, _ = model(pixel_values)
            logits_2, _, _, _ = model(torch.flip(pixel_values, dims=[-1]))
            logits = (logits_1 + logits_2) / 2.0

        loss = ce_per_sample(logits, labels, label_smoothing).mean()
        predictions = logits.argmax(dim=1)

        loss_sum += float(loss.item()) * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        preds_all.append(predictions.cpu())
        labels_all.append(labels.cpu())

    if total == 0:
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0}

    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    macro_f1 = macro_f1_from_preds(model.n_classes, preds_all, labels_all)
    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "macro_f1": macro_f1,
    }


def build_optimizer(model: HerbariumTextAlignmentModel, config: TrainingConfig):
    head_params = []
    head_params.extend(list(model.classifier.parameters()))
    head_params.extend(model.img_enc.projection_parameters())
    head_params.extend(list(model.txt_enc.proj.parameters()))

    head_param_ids = {id(param) for param in head_params}
    base_params = [param for param in model.parameters() if param.requires_grad and id(param) not in head_param_ids]

    if config.head_type == "cosine" or config.head_type == "linear":
        pass
    else:
        raise ValueError(f"Unsupported head_type: {config.head_type}")

    if config.base_lr <= 0.0:
        raise ValueError("base_lr must be positive.")

    if config.momentum < 0.0:
        raise ValueError("momentum must be non-negative.")

    optimizer_name = "adamw"
    if optimizer_name == "adamw":
        optimizer = AdamW(
            [
                {"params": head_params, "lr": config.base_lr * config.head_lr_mult},
                {"params": base_params, "lr": config.base_lr},
            ],
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = SGD(
            [
                {"params": head_params, "lr": config.base_lr},
                {"params": base_params, "lr": config.base_lr},
            ],
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.base_lr * 0.05)
    return optimizer, scheduler


def train_model(
    config: TrainingConfig,
    train_jsonl: Path,
    test_jsonl: Path,
    label2id_path: Path,
    save_dir: Path,
    num_workers: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed, deterministic=config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label2id = load_label2id(label2id_path)
    n_classes = len(label2id)

    tokenizer, train_dataset, _, train_loader, val_loader = build_dataloaders(
        train_jsonl=train_jsonl,
        test_jsonl=test_jsonl,
        tokenizer_name=config.tokenizer_name,
        batch_size=config.batch_size,
        max_len=config.max_len,
        n_classes=n_classes,
        num_workers=num_workers,
        augment=config.augment,
        balanced_sampler=config.balanced_sampler,
    )

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
    ).to(device)

    optimizer, scheduler = build_optimizer(model, config)

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=(config.amp and amp_device == "cuda"))

    label_counter = count_labels(train_dataset.rows)
    cb_weights = None
    if config.cb_loss:
        counts = torch.tensor([label_counter.get(class_id, 0) for class_id in range(n_classes)], dtype=torch.float, device=device)
        cb_weights = class_balanced_weights(counts, beta=config.cb_beta)

    metrics_csv_path = save_dir / "metrics.csv"
    metrics_jsonl_path = save_dir / "metrics.jsonl"
    summary_jsonl_path = save_dir / "summary.jsonl"
    best_ckpt_path = save_dir / "best.pt"
    config_json_path = save_dir / "config.json"

    write_json(
        config_json_path,
        {
            "preset_name": config.preset_name,
            "config": config.to_dict(),
            "train_jsonl": train_jsonl,
            "test_jsonl": test_jsonl,
            "label2id": label2id_path,
            "num_workers": num_workers,
        },
    )

    csv_header = [
        "epoch",
        "dt_sec",
        "train_loss",
        "train_ce",
        "train_align",
        "align_w",
        "train_acc",
        "train_macro_f1",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "temp",
        "head",
        "arch",
        "align_type",
        "cls_dim",
        "align_dim",
    ]

    best_acc = 0.0
    best_f1 = 0.0
    best_loss = float("inf")
    best_epoch = 0
    best_record = None
    prototypes = None
    prototype_valid = None

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=csv_header)
        writer.writeheader()

        for epoch in range(1, config.epochs + 1):
            start_time = time.time()
            model.train()

            if config.align_type in ["proto", "proto_dcl"] and config.w_align > 0.0 and not config.freeze_txt_all:
                every = max(0, config.proto_recompute_every)
                if every > 0 and (epoch == 1 or epoch % every == 0):
                    prototypes, prototype_valid = compute_text_prototypes(
                        txt_enc=model.txt_enc,
                        tokenizer=tokenizer,
                        rows=train_dataset.rows,
                        n_classes=n_classes,
                        device=device,
                        max_len=config.max_len,
                        min_text_tokens=config.min_text_tokens,
                        max_per_class=config.proto_max_per_class,
                        batch_size=config.proto_batch_size,
                        amp=config.amp,
                    )
                    print(f"[Proto] recompute @ epoch {epoch}: valid_classes={prototype_valid.float().mean().item():.2f}")

            current_align_weight = align_weight(
                epoch=epoch,
                w_align=config.w_align,
                warmup_epochs=config.warmup_align_epochs,
                ramp_epochs=config.align_ramp_epochs,
            )

            total = 0
            correct = 0
            total_loss = 0.0
            total_ce = 0.0
            total_align = 0.0
            train_preds = []
            train_labels = []

            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(amp_device, enabled=(config.amp and amp_device == "cuda")):
                    logits, _, img_align, txt_features = model(pixel_values, input_ids, attention_mask)

                    ce_all = ce_per_sample(logits, labels, config.label_smoothing)
                    if cb_weights is not None:
                        weights = cb_weights.index_select(0, labels)
                        ce_loss = (ce_all * weights).mean()
                    else:
                        ce_loss = ce_all.mean()

                    align_loss = logits.new_tensor(0.0)
                    if config.align_type != "none" and current_align_weight > 0.0:
                        token_lengths = attention_mask.sum(dim=1)
                        valid_text = token_lengths >= int(config.min_text_tokens)

                        if config.align_type in ["inst", "inst_dcl"]:
                            if txt_features is not None and valid_text.sum().item() >= 2:
                                image_features = grad_scale_trick(img_align[valid_text], config.align_backbone_grad_scale)
                                align_loss = inst_infonce_loss(
                                    image_features,
                                    txt_features[valid_text],
                                    temperature=model.temperature,
                                    dcl=(config.align_type == "inst_dcl"),
                                )

                        elif config.align_type in ["proto", "proto_dcl"]:
                            if prototypes is not None and prototype_valid is not None:
                                keep = prototype_valid.index_select(0, labels)
                                if keep.any():
                                    image_features = grad_scale_trick(img_align[keep], config.align_backbone_grad_scale)
                                    align_loss = proto_softmax_loss(
                                        image_features,
                                        labels[keep],
                                        prototypes,
                                        prototype_valid,
                                        temperature=model.temperature,
                                        dcl=(config.align_type == "proto_dcl"),
                                        hard_neg_alpha=config.hard_neg_alpha,
                                    )

                    loss = ce_loss + current_align_weight * align_loss

                scaler.scale(loss).backward()
                if config.grad_clip_val is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_val)
                scaler.step(optimizer)
                scaler.update()

                predictions = logits.argmax(dim=1)
                batch_size = labels.size(0)

                total += batch_size
                correct += (predictions == labels).sum().item()
                total_loss += float(loss.item()) * batch_size
                total_ce += float(ce_loss.item()) * batch_size
                total_align += float(align_loss.item()) * batch_size
                train_preds.append(predictions.detach().cpu())
                train_labels.append(labels.detach().cpu())

            scheduler.step()

            elapsed = time.time() - start_time
            train_preds = torch.cat(train_preds) if train_preds else torch.empty(0, dtype=torch.long)
            train_labels = torch.cat(train_labels) if train_labels else torch.empty(0, dtype=torch.long)
            train_acc = correct / max(1, total)
            train_f1 = macro_f1_from_preds(n_classes, train_preds, train_labels) if total > 0 else 0.0
            train_loss = total_loss / max(1, total)
            train_ce = total_ce / max(1, total)
            train_align = total_align / max(1, total)

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                label_smoothing=config.label_smoothing,
                tta=config.tta,
            )

            record = {
                "epoch": epoch,
                "dt_sec": elapsed,
                "train_loss": train_loss,
                "train_ce": train_ce,
                "train_align": train_align,
                "align_w": current_align_weight,
                "train_acc": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "temp": float(model.temperature.item()),
                "head": config.head_type,
                "arch": config.arch,
                "align_type": config.align_type,
                "cls_dim": model.img_enc.cls_dim,
                "align_dim": model.img_enc.align_dim,
            }

            writer.writerow(record)
            csv_handle.flush()
            append_jsonl(metrics_jsonl_path, {**record, "config": config.to_dict()})

            print(
                f"[Epoch {epoch:03d}] {elapsed:.1f}s | "
                f"Train loss {train_loss:.4f} (CE {train_ce:.4f}, Align {train_align:.4f}, w={current_align_weight:.3f}) "
                f"acc {train_acc:.4f} macroF1 {train_f1:.4f} | "
                f"Val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} macroF1 {val_metrics['macro_f1']:.4f} | "
                f"temp={model.temperature.item():.3f} head={config.head_type} align={config.align_type}"
            )

            if is_better_model(
                curr_acc=val_metrics["acc"],
                curr_f1=val_metrics["macro_f1"],
                curr_loss=val_metrics["loss"],
                best_acc=best_acc,
                best_f1=best_f1,
                best_loss=best_loss,
            ):
                best_acc = val_metrics["acc"]
                best_f1 = val_metrics["macro_f1"]
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                best_record = {**record, "best_epoch": best_epoch}

                if config.save_best:
                    checkpoint = {
                        "model": model.state_dict(),
                        "label2id": label2id,
                        "best_epoch": best_epoch,
                        "best_metrics": {
                            "val_acc": best_acc,
                            "val_macro_f1": best_f1,
                            "val_loss": best_loss,
                        },
                        "preset_name": config.preset_name,
                        "config": config.to_dict(),
                    }
                    torch.save(checkpoint, best_ckpt_path)

                append_jsonl(summary_jsonl_path, {"event": "new_best", **best_record})

        if best_record is not None:
            append_jsonl(summary_jsonl_path, {"event": "final_best", **best_record, "checkpoint": best_ckpt_path})
            print(
                f"[Best] epoch={best_epoch} acc={best_acc:.4f} macroF1={best_f1:.4f} "
                f"loss={best_loss:.4f} ckpt={best_ckpt_path}"
            )
        else:
            append_jsonl(summary_jsonl_path, {"event": "final_best", "message": "No best checkpoint was produced."})
