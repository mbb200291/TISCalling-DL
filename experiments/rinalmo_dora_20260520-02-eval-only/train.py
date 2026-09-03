'''
features:
- Pretrained model: RiNRiNALMo
- Trained on Peft (DoRA)
- Increase positive weight
- Increase non-annotated TIS
- Select model by dev evaluated only on non-annotated TIS
'''


from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(hyperparameters: dict):
    import torch

    requested = hyperparameters.get("device", "cuda:0")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _load_tokenizer(hyperparameters: dict):
    from multimolecule import RnaTokenizer

    model_id = hyperparameters.get("rinalmo_model_id", "multimolecule/rinalmo-micro")
    return RnaTokenizer.from_pretrained(model_id)


def _build_collate_fn(tokenizer, pad_token_id: int):
    import torch

    def collate_fn(batch):
        seqs, labels, sections = zip(*batch)
        encoding = tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        attention_mask = attention_mask.masked_fill(input_ids == pad_token_id, 0)
        return (
            input_ids,
            attention_mask,
            attention_mask.sum(dim=1),
            torch.tensor(labels, dtype=torch.float),
            torch.tensor(sections, dtype=torch.float),
        )

    return collate_fn


def _build_loader(df: pd.DataFrame, batch_size: int, collate_fn, shuffle: bool):
    from torch.utils.data import DataLoader, Dataset

    class RNADataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.sequences = frame["seq"].astype(str).tolist()
            self.labels = frame["label"].astype(int).tolist()
            self.sections = frame["section_id"].astype(int).tolist()

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx], self.sections[idx]

    return DataLoader(RNADataset(df), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _build_model(hyperparameters: dict, device):
    import torch
    import torch.nn as nn
    from multimolecule import RiNALMoModel
    from peft import LoraConfig, TaskType, get_peft_model

    class RiNALMoClassifier(nn.Module):
        def __init__(self, backbone: nn.Module):
            super().__init__()
            d_model = backbone.config.hidden_size
            pooling_mode = hyperparameters.get("pooling_mode", "attention")
            if pooling_mode not in {"marker", "attention"}:
                raise ValueError("pooling_mode must be marker or attention")

            self.backbone = backbone
            self.pooling_mode = pooling_mode
            if pooling_mode == "attention":
                self.marker_attention = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=int(hyperparameters.get("num_heads", 8)),
                    dropout=float(hyperparameters.get("dropout", 0.1)),
                    batch_first=True,
                )
            classifier_in = d_model if pooling_mode == "marker" else d_model * 2
            self.dropout = nn.Dropout(float(hyperparameters.get("dropout", 0.1)))
            self.classifier = nn.Linear(classifier_in, 1)

        def forward(self, input_ids, attention_mask, lengths):
            del lengths
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            cls_h = hidden[batch_idx, 0]
            if self.pooling_mode == "marker":
                features = cls_h
            else:
                global_h, _ = self.marker_attention(
                    query=cls_h.unsqueeze(1),
                    key=hidden,
                    value=hidden,
                    key_padding_mask=attention_mask == 0,
                    need_weights=False,
                )
                features = torch.cat([cls_h, global_h.squeeze(1)], dim=-1)
            return self.classifier(self.dropout(features)).squeeze(-1)

    base_model = RiNALMoModel.from_pretrained(hyperparameters.get("rinalmo_model_id", "multimolecule/rinalmo-micro"))
    use_peft = hyperparameters.get('use_peft', True)
    if use_peft:
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(hyperparameters.get("lora_r", 4)),
            lora_alpha=int(hyperparameters.get("lora_alpha", 32)),
            lora_dropout=float(hyperparameters.get("lora_dropout", 0.1)),
            use_dora=bool(hyperparameters.get("use_dora", True)),
            bias="none",
            target_modules=list(hyperparameters.get("target_modules", ["query", "key", "value", "dense"])),
        )
        backbone = get_peft_model(base_model, lora_cfg)
        backbone.enable_input_require_grads()
        if hyperparameters.get('gradient_checkpointing_enable', False):
            backbone.gradient_checkpointing_enable()
        backbone.print_trainable_parameters()
    else: 
        for p in base_model.parameters():  # won't train hidden layer
            p.requires_grad = False
        base_model.eval()  # to disable dropout
        backbone = base_model
        # if hyperparameters.get('gradient_checkpointing_enable', False):
        #     backbone.gradient_checkpointing_enable()
        #     if hasattr(backbone, "enable_input_require_grads"):
        #         backbone.enable_input_require_grads()
    model = RiNALMoClassifier(backbone).to(device)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[build_model] use_peft={use_peft}, trainable={total_trainable:,} / total={total_params:,} "
          f"({100 * total_trainable / total_params:.4f}%)")
    return model


def _loss_weight_stats(train_df: pd.DataFrame) -> dict:
    labels = train_df["label"].astype(int).to_numpy()
    sections = train_df["section_id"].astype(int).to_numpy()
    section_counts = Counter(sections)
    total_train = len(train_df)
    section_ratio = {int(section): total_train / count for section, count in section_counts.items()}

    section_label_counts = defaultdict(lambda: Counter())
    for section, label in zip(sections, labels):
        section_label_counts[int(section)][int(label)] += 1

    section_neg_pos_ratio = {}
    for section in section_counts:
        neg_count = section_label_counts[int(section)][0]
        pos_count = section_label_counts[int(section)][1]
        section_neg_pos_ratio[int(section)] = 1.0 if pos_count == 0 else neg_count / pos_count
    num_pos = int((labels == 1).sum())
    num_neg = int((labels == 0).sum())
    return {
        "pos_weight": num_neg / max(num_pos, 1),
        "num_pos": num_pos,
        "num_neg": num_neg,
        "section_ratio": section_ratio,
        "section_neg_pos_ratio": section_neg_pos_ratio,
    }


def _section_weights(train_df: pd.DataFrame) -> tuple[dict[int, float], dict[int, float]]:
    stats = _loss_weight_stats(train_df)
    return stats["section_ratio"], stats["section_neg_pos_ratio"]


def _fallback_noncanonical_weight(hyperparameters: dict, annotated_section_id: int) -> float:
    if "noncanonical_weight_fold" in hyperparameters:
        return float(hyperparameters["noncanonical_weight_fold"])
    section_weight_fold = hyperparameters.get("section_weight_fold", {})
    values = [
        float(weight)
        for section, weight in section_weight_fold.items()
        if int(section) != annotated_section_id
    ]
    return max(values) if values else 1.0


def _criterion(hyperparameters: dict, weight_stats: dict):
    import torch
    import torch.nn.functional as F

    annotated_section_id = int(hyperparameters.get("annotated_section_id", 0))
    noncanonical_weight_fold = _fallback_noncanonical_weight(hyperparameters, annotated_section_id)
    pos_weight = float(weight_stats["pos_weight"]) * float(hyperparameters.get("pos_weight_fold", 1.0))
    max_weight = hyperparameters.get("max_weight", 100)

    def criterion_fn(logits, labels, sections):
        sample_w = torch.ones_like(labels, dtype=torch.float32, device=logits.device)
        sample_w[sections != annotated_section_id] = noncanonical_weight_fold
        if max_weight is not None:
            sample_w = torch.clamp(sample_w, max=float(max_weight))
        return F.binary_cross_entropy_with_logits(
            logits,
            labels.float(),
            weight=sample_w,
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=logits.device),
        )

    return criterion_fn


def _predict_arrays(model, dataloader, device, threshold: float, ignore_sec: int):
    import torch

    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, lengths, labels, sections in dataloader:
            logits = model(input_ids.to(device), attention_mask.to(device), lengths.to(device))
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= threshold).float()
            keep = sections != ignore_sec
            all_probs.append(probs[keep])
            all_preds.append(preds[keep])
            all_labels.append(labels[keep])
    return torch.cat(all_probs), torch.cat(all_preds), torch.cat(all_labels)


def _compute_metrics(labels, preds, probs) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels_np = np.asarray(labels).astype(int)
    preds_np = np.asarray(preds).astype(int)
    probs_np = np.asarray(probs)
    out = {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "precision": float(precision_score(labels_np, preds_np, zero_division=0)),
        "recall": float(recall_score(labels_np, preds_np, zero_division=0)),
        "f1": float(f1_score(labels_np, preds_np, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels_np, preds_np)),
    }
    if len(np.unique(labels_np)) < 2:
        out["auc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auc"] = float(roc_auc_score(labels_np, probs_np))
        out["auprc"] = float(average_precision_score(labels_np, probs_np))
    out["aupr"] = out["auprc"]
    return out


def _metrics_from_logits(logits, labels, threshold: float) -> dict:
    import torch

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= threshold).astype(int)
    return _compute_metrics(labels.detach().cpu().numpy(), preds, probs)


def _evaluate_epoch(model, dataloader, device, threshold: float, ignore_sec: int) -> dict | None:
    import torch
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, lengths, labels, sections in dataloader:
            logits = model(input_ids.to(device), attention_mask.to(device), lengths.to(device))
            keep = sections != ignore_sec
            logits = logits[keep]
            labels = labels[keep]
            if labels.numel() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(logits.cpu(), labels.float())
            total_loss += float(loss.item()) * len(labels)
            total_samples += len(labels)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if total_samples == 0:
        return None
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = _metrics_from_logits(logits, labels, threshold)
    return {"loss": total_loss / total_samples, **metrics}


def _format_epoch_metrics(prefix: str, values: dict | None) -> str:
    if values is None:
        return f"{prefix}: no samples"
    keys = ["loss", "accuracy", "precision", "recall", "f1", "mcc", "auc", "auprc"]
    parts = []
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value:.4f}")
    return f"{prefix}: " + " ".join(parts)


def _format_counts(values: pd.Series) -> dict:
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def train(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame | None,
    hyperparameters: dict,
    model_path: Path,
    seed: int,
) -> dict:
    import torch

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _set_seed(seed)
    device = _device(hyperparameters)
    tokenizer = _load_tokenizer(hyperparameters)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("RnaTokenizer has no pad_token_id")

    collate_fn = _build_collate_fn(tokenizer, pad_token_id)
    train_loader = _build_loader(train_df, int(hyperparameters.get("batch_size", 16)), collate_fn, shuffle=True)
    dev_loader = _build_loader(dev_df, int(hyperparameters.get("batch_size", 16)), collate_fn, shuffle=False) if dev_df is not None else None
    weight_stats = _loss_weight_stats(train_df)
    section_ratio = weight_stats["section_ratio"]
    section_neg_pos_ratio = weight_stats["section_neg_pos_ratio"]
    criterion_fn = _criterion(hyperparameters, weight_stats)

    print(
        "[train_setup] "
        f"device={device} seed={seed} "
        f"train_rows={len(train_df)} dev_rows={0 if dev_df is None else len(dev_df)} "
        f"batch_size={hyperparameters.get('batch_size', 16)} max_epoch={hyperparameters.get('max_epoch', 50)}",
        flush=True,
    )
    print(f"[train_setup] train_label_counts={_format_counts(train_df['label'])}", flush=True)
    print(f"[train_setup] train_section_counts={_format_counts(train_df['section_id'])}", flush=True)
    if dev_df is not None:
        print(f"[train_setup] dev_label_counts={_format_counts(dev_df['label'])}", flush=True)
        print(f"[train_setup] dev_section_counts={_format_counts(dev_df['section_id'])}", flush=True)
    print(f"[train_setup] section_ratio={section_ratio}", flush=True)
    print(f"[train_setup] section_neg_pos_ratio={section_neg_pos_ratio}", flush=True)
    print(
        "[train_setup] loss_weights="
        f"pos_weight={weight_stats['pos_weight']:.4f} "
        f"pos_weight_fold={hyperparameters.get('pos_weight_fold', 1.0)} "
        f"noncanonical_weight_fold={_fallback_noncanonical_weight(hyperparameters, int(hyperparameters.get('annotated_section_id', 0)))}",
        flush=True,
    )

    model = _build_model(hyperparameters, device)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(hyperparameters.get("lr", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(hyperparameters.get("scheduler_factor", 0.5)),
        patience=int(hyperparameters.get("scheduler_patience", 3)),
        min_lr=float(hyperparameters.get("scheduler_min_lr", 1e-6)),
    )

    threshold = float(hyperparameters.get("threshold", 0.5))
    ignore_sec = int(hyperparameters.get("ignore_sec", -1))
    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    patience = int(hyperparameters.get("early_stopping_patience", 5))
    epoch_history = []
    log_every_n_batches = int(hyperparameters.get("log_every_n_batches", 0))

    for epoch in range(1, int(hyperparameters.get("max_epoch", 50)) + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        train_logits = []
        train_labels = []
        for batch_idx, (input_ids, attention_mask, lengths, labels, sections) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            logits = model(input_ids.to(device), attention_mask.to(device), lengths.to(device))
            labels_device = labels.to(device).float()
            loss = criterion_fn(logits, labels_device, sections.to(device).float())
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(labels)
            train_samples += len(labels)
            train_logits.append(logits.detach().cpu())
            train_labels.append(labels.detach().cpu())
            if log_every_n_batches > 0 and batch_idx % log_every_n_batches == 0:
                running_loss = train_loss / max(train_samples, 1)
                print(
                    f"Epoch {epoch:03d} batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"running_train_loss={running_loss:.4f}",
                    flush=True,
                )

        train_epoch_metrics = {
            "loss": train_loss / max(train_samples, 1),
            **_metrics_from_logits(
                logits=torch.cat(train_logits),
                labels=torch.cat(train_labels),
                threshold=threshold,
            ),
        }

        if dev_loader is None:
            dev_epoch_metrics = None
            score = -train_epoch_metrics["loss"]
        else:
            dev_epoch_metrics = _evaluate_epoch(model, dev_loader, device, threshold, ignore_sec)
            metrics = dev_epoch_metrics or {}
            score = float(metrics.get("auprc", float("nan")))
            if np.isnan(score):
                score = float(metrics.get("f1", 0.0))
        scheduler.step(score)

        lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "lr": float(lr),
            "train": train_epoch_metrics,
            "dev": dev_epoch_metrics,
            "score": score,
        }
        epoch_history.append(epoch_record)
        print(
            f"Epoch {epoch:03d} | lr={lr:.2e} | "
            f"{_format_epoch_metrics('train', train_epoch_metrics)} | "
            f"{_format_epoch_metrics('dev', dev_epoch_metrics)}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hyperparameters": hyperparameters,
                    "weight_stats": weight_stats,
                    "seed": seed,
                },
                model_path,
            )
            print(f"[checkpoint] saved best model epoch={epoch} score={score:.4f} path={model_path}", flush=True)
        else:
            no_improve += 1
            print(
                f"[early_stopping] no_improve={no_improve}/{patience} "
                f"best_epoch={best_epoch} best_score={best_score:.4f}",
                flush=True,
            )
            if no_improve >= patience:
                print(f"[early_stopping] stopped at epoch={epoch}", flush=True)
                break

    return {
        "train_rows": int(len(train_df)),
        "dev_rows": 0 if dev_df is None else int(len(dev_df)),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "weight_stats": weight_stats,
        "epoch_history": epoch_history,
    }
