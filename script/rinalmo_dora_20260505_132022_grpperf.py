import random
import math
from typing import List, Tuple
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from pprint import pprint

# ── RiNALMo (multimolecule) ───────────────────────────────────────────────────
from multimolecule import RnaTokenizer, RiNALMoModel

# ── PEFT / DoRA ───────────────────────────────────────────────────────────────
from peft import get_peft_model, LoraConfig, TaskType

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 23
random.seed(SEED)
torch.manual_seed(SEED)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# ── Model variant: rinalmo-micro (30M) | rinalmo-mega (150M) | rinalmo (650M) ─
# RINALMO_MODEL_ID = "multimolecule/rinalmo-mega"
RINALMO_MODEL_ID = "multimolecule/rinalmo-micro"

# ── 全域只載入一次 tokenizer ──────────────────────────────────────────────────
print(f"Loading tokenizer: {RINALMO_MODEL_ID}", flush=True)
_tokenizer = RnaTokenizer.from_pretrained(RINALMO_MODEL_ID)

# RiNALMo tokenizer will add [CLS] at front and [EOS] at the end, so nucleotide position should add 1 offset
_CLS_OFFSET = 1


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class RNADataset(Dataset):
    """
    Raw sequences + nucleotide-space positions + binary labels.
    Tokenisation is deferred to collate_fn.
    """

    def __init__(
        self, sequences: List[str], positions: List[int],
        labels: List[int], sections: List[int],
        ):
        assert len(sequences) == len(positions) == len(labels) #== len(sections)
        for seq, pos in zip(sequences, positions):
            assert len(seq) > pos, f"position {pos} out of range for seq len {len(seq)}"
        self.sequences = sequences
        self.positions = positions
        self.labels = labels
        self.sections = sections

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.positions[idx],
            torch.tensor(self.labels[idx], dtype=torch.float),
            self.sections[idx],
        )

def collate_fn(batch):
    seqs, positions, labels, sections = zip(*batch)

    encoding = _tokenizer(
        list(seqs),
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids      = encoding["input_ids"]       # [B, L]
    attention_mask = encoding["attention_mask"]  # [B, L]

    B, L = input_ids.shape
    pos_features = torch.zeros(B, L, dtype=torch.float)
    for i, pos in enumerate(positions):
        token_idx = pos + _CLS_OFFSET  # +1 for [CLS]
        if token_idx < L:
            pos_features[i, token_idx] = 1.0
        else:
            pos_features[i, int(attention_mask[i].sum()) - 1] = 1.0

    lengths = attention_mask.sum(dim=1)
    labels_t = torch.stack(list(labels))
    sections_t = torch.tensor(sections, dtype=torch.float)
    return input_ids, attention_mask, pos_features, lengths, labels_t, sections_t


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class RiNALMoClassifier(nn.Module):
    """
    RiNALMo backbone (DoRA fine-tuned) + marker-attention classification head.

    pooling_mode:
      "marker"    – only the marker token hidden state
      "attention" – marker token + marker-attended global context (concatenated)
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pooling_mode: str = "attention",
    ):
        super().__init__()

        valid_modes = {"marker", "attention"}
        assert pooling_mode in valid_modes, f"pooling_mode must be one of {valid_modes}"

        self.backbone     = backbone
        self.d_model      = d_model
        self.pooling_mode = pooling_mode

        if pooling_mode == "attention":
            self.marker_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.dropout = nn.Dropout(dropout)

        classifier_in = d_model if pooling_mode == "marker" else d_model * 2
        self.classifier = nn.Linear(classifier_in, 1)

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
        pos_features:   torch.Tensor,   # [B, L]  one-hot token-space marker
        lengths:        torch.Tensor,   # [B]  (unused, kept for API compat)
    ) -> torch.Tensor:                  # [B]  raw logits

        # ── backbone ──────────────────────────────────────────────────────────
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state   # [B, L, D]

        # ── extract marker token ───────────────────────────────────────────────
        B = hidden.size(0)
        batch_idx  = torch.arange(B, device=hidden.device)
        marker_idx = pos_features.argmax(dim=1)        # [B]
        marker_h   = hidden[batch_idx, marker_idx]     # [B, D]

        # ── pooling ───────────────────────────────────────────────────────────
        if self.pooling_mode == "marker":
            h = marker_h                               # [B, D]

        elif self.pooling_mode == "attention":
            padding_mask = (attention_mask == 0)       # [B, L]  True = pad
            global_h, _ = self.marker_attention(
                query=marker_h.unsqueeze(1),           # [B, 1, D]
                key=hidden,
                value=hidden,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            global_h = global_h.squeeze(1)            # [B, D]
            h = torch.cat([marker_h, global_h], dim=-1)  # [B, 2D]

        logits = self.classifier(self.dropout(h))      # [B, 1]
        return logits.squeeze(-1)                      # [B]


# ══════════════════════════════════════════════════════════════════════════════
# Build backbone with DoRA
# ══════════════════════════════════════════════════════════════════════════════

def build_dora_backbone(lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Load RiNALMo and wrap with DoRA adapters via PEFT.
    RiNALMo is a standard HuggingFace PreTrainedModel → fully PEFT-compatible.
    """
    print(f"Loading backbone: {RINALMO_MODEL_ID}", flush=True)
    base_model = RiNALMoModel.from_pretrained(RINALMO_MODEL_ID)

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=True,                           # DoRA
        bias="none",
        # RiNALMo uses standard BERT-style names; adjust if needed
        target_modules=["query", "key", "value", "dense"],
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    # peft_model.gradient_checkpointing_enable()  # 
    peft_model.print_trainable_parameters()
    return peft_model


def build_model(hp: dict, device: torch.device) -> RiNALMoClassifier:
    backbone = build_dora_backbone(
        lora_r=hp.get("lora_r", 16),
        lora_alpha=hp.get("lora_alpha", 32),
        lora_dropout=hp.get("lora_dropout", 0.05),
    )

    # RiNALMo hidden sizes: micro=480, mega=640, giga=1280
    d_model = backbone.config.hidden_size
    # d_model = hp.get('d_model', 64)

    model = RiNALMoClassifier(
        backbone=backbone,
        d_model=d_model,
        num_heads=hp.get("num_heads", 8),
        dropout=hp.get("dropout", 0.1),
        pooling_mode=hp.get("pooling_mode", "attention"),
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  |  Trainable: {trainable:,}", flush=True)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Shared metrics computation
# ══════════════════════════════════════════════════════════════════════════════
 
def _compute_metrics_from_logits(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    total_loss: float,
    total_samples: int,
    threshold: float = 0.5,
):
    """
    Compute classification metrics from accumulated logits and labels.
    Shared by both train_one_epoch and evaluate.
    """
    probs = torch.sigmoid(all_logits)
    preds = (probs >= threshold).float()
 
    tp = ((preds == 1) & (all_labels == 1)).sum().item()
    tn = ((preds == 0) & (all_labels == 0)).sum().item()
    fp = ((preds == 1) & (all_labels == 0)).sum().item()
    fn = ((preds == 0) & (all_labels == 1)).sum().item()
 
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
 
    mcc_d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc   = (tp * tn - fp * fn) / np.sqrt(mcc_d) if mcc_d else 0.0
 
    labels_np = all_labels.numpy()
    probs_np  = probs.numpy()
    if len(np.unique(labels_np)) < 2:
        auc = aupr = float("nan")
    else:
        auc  = roc_auc_score(labels_np, probs_np)
        aupr = average_precision_score(labels_np, probs_np)
 
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": spec, "f1": f1, "mcc": mcc,
        "auc": auc, "aupr": aupr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation loops
# ══════════════════════════════════════════════════════════════════════════════
 
def train_one_epoch(model, dataloader, optimizer, criterion_fn, device, threshold=0.5):
    model.train()
    total_loss = total_samples = 0
    all_logits, all_labels = [], []
 
    for input_ids, attention_mask, pos_features, lengths, labels, sections in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_features   = pos_features.to(device)
        lengths        = lengths.to(device)
        labels         = labels.to(device).float()
        sections = sections.to(device).float()
 
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, pos_features, lengths)
        loss   = criterion_fn(logits, labels, sections)
        loss.backward()
        optimizer.step()
 
        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)
 
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
 
    return _compute_metrics_from_logits(
        torch.cat(all_logits), torch.cat(all_labels),
        total_loss, total_samples, threshold,
    )
 
 
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=0.5, ignore_sec=-1):
    model.eval()
    total_loss = total_samples = 0
    all_logits, all_labels = [], []
 
    for input_ids, attention_mask, pos_features, lengths, labels, sections in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_features   = pos_features.to(device)
        lengths        = lengths.to(device)
        labels         = labels.to(device).float()
        sections       = sections.to(device).float()
 
        logits = model(input_ids, attention_mask, pos_features, lengths)
        
        # filter out non canoical tis
        keep = (sections != ignore_sec)
        logits = logits[keep]
        labels = labels[keep]
        if labels.numel() == 0:
            continue
        
        loss   = criterion(logits, labels)
 
        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)
 
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
 
    return _compute_metrics_from_logits(
        torch.cat(all_logits), torch.cat(all_labels),
        total_loss, total_samples, threshold,
    )
 
# def get_criterion_getter(noncanonical_weight_fold, pos_ratio, pos_weight_fold):
#     pw = pos_ratio * pos_weight_fold
#     def criterion_fn(logits, labels, sections):
#         sample_w = sections * noncanonical_weight_fold + (1 - sections)
#         return F.binary_cross_entropy_with_logits(
#             logits, labels, weight=sample_w, pos_weight=pw,
#         )
#     return criterion_fn


def get_criterion_fn(noncanonical_weight_fold, pos_weight, pos_weight_fold):
    """
    Returns a per-batch loss function.
    sections == 1 → non-annotated → weighted by noncanonical_weight_fold
    sections == 0 → annotated     → weight 1
    """
    pw = pos_weight * pos_weight_fold   # built once

    def criterion_fn(logits, labels, sections):
        sample_w = sections * noncanonical_weight_fold + (1 - sections)
        return F.binary_cross_entropy_with_logits(
            logits, labels,
            weight=sample_w,
            pos_weight=pw,
        )
    return criterion_fn

# ══════════════════════════════════════════════════════════════════════════════
# Train loop
# ══════════════════════════════════════════════════════════════════════════════

def train_model(train_loader, val_loader, hp: dict, pos_weight,
                naming_prefix="model", patience=15):

    print("hyperparameters:", hp)
    model = build_model(hp, DEVICE)

    # Train-side
    criterion_fn = get_criterion_fn(
        noncanonical_weight_fold=hp.get("noncanonical_weight_fold", 1.0),
        pos_weight=pos_weight,
        pos_weight_fold=hp.get("pos_weight_fold", 1.0),
    )
    # Eval-side: plain BCE (no sample weight; sections are filtered out anyway)
    eval_criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight * hp.get("pos_weight_fold", 1.0)
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.get("lr", 1e-4),
    )
    # scheduler patience < early stopping patience to give lr reduction a chance
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    best_score = -1
    no_improve  = 0
    run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path   = f"model/{naming_prefix}_{run_id}.pt"
    os.makedirs("model", exist_ok=True)

    for epoch in range(1, hp.get("max_epoch", 50) + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion_fn, DEVICE)
        va = evaluate(model, val_loader, eval_criterion, DEVICE, ignore_sec=0)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | lr={lr_now:.2e} | "
            f"train loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} "
            f"prec={tr['precision']:.4f} rec={tr['recall']:.4f} f1={tr['f1']:.4f} | "
            f"val   loss={va['loss']:.4f} acc={va['accuracy']:.4f} "
            f"prec={va['precision']:.4f} rec={va['recall']:.4f} f1={va['f1']:.4f} "
            f"aupr={va['aupr']:.4f}",
            flush=True,
        )
        scheduler.step(va["aupr"])

        if va["aupr"] > best_score:
            best_score = va["aupr"]
            no_improve  = 0
            print(f"  -> saved best model to {save_path}", flush=True)
            sd = model.state_dict()
            torch.save(sd, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  -> early stopping at epoch {epoch}", flush=True)
                break

    print(f"Best val AUPR: {best_score:.4f}")
    return model, save_path


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_batch(model, dataloader, device, threshold=0.5, ignore_sec=-1):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for input_ids, attention_mask, pos_features, lengths, labels, sections in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_features   = pos_features.to(device)
        lengths        = lengths.to(device)

        logits = model(input_ids, attention_mask, pos_features, lengths)
        probs  = torch.sigmoid(logits).cpu()
        preds  = (probs >= threshold).float()

        keep = (sections != ignore_sec)
        all_probs.append(probs[keep])
        all_preds.append(preds[keep])
        all_labels.append(labels[keep])

    return torch.cat(all_probs), torch.cat(all_preds), torch.cat(all_labels)


def load_model(path: str, hp: dict, device: torch.device = DEVICE) -> RiNALMoClassifier:
    model = build_model(hp, device)
    sd    = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Dataset factory
# ══════════════════════════════════════════════════════════════════════════════

def make_dataset(sequences, positions, labels, groups, sections, batch_size=8):
    assert len(sequences) == len(positions) == len(labels) == len(groups) == len(sections)
    print("labels:", dict(Counter(labels)))

    dataset  = RNADataset(sequences, positions, labels, sections)
    indices  = np.arange(len(dataset))
    groups_a = np.array(groups)
    labels_a = np.array(labels)
    sections_a = np.array(sections)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_idx, test_idx = next(gss_test.split(indices, labels_a, groups=groups_a))
    # test_idx = test_idx[sections_a[test_idx] == 1]   # only keep section label as 1 on test set

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=SEED)
    train_rel, val_rel = next(
        gss_val.split(indices[train_val_idx], labels_a[train_val_idx],
                      groups=groups_a[train_val_idx])
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]
    # val_idx = val_idx[sections_a[val_idx] == 1]  # only keep section label as 1 on dev set
    
    # sanity check: no group overlap across splits
    assert set(groups_a[train_idx]).isdisjoint(set(groups_a[val_idx]))
    assert set(groups_a[train_idx]).isdisjoint(set(groups_a[test_idx]))
    assert set(groups_a[val_idx]).isdisjoint(set(groups_a[test_idx]))

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds   = Subset(dataset, val_idx.tolist())
    test_ds  = Subset(dataset, test_idx.tolist())

    train_labels = labels_a[train_idx]
    num_pos  = (train_labels == 1).sum()
    num_neg  = (train_labels == 0).sum()
    pos_weight = torch.tensor(
        [num_neg / max(num_pos, 1)], dtype=torch.float32, device=DEVICE
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray):
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc   = (tp*tn - fp*fn) / np.sqrt(denom) if denom else 0.0

    if len(np.unique(y_true)) < 2:
        auc = aupr = float("nan")
    else:
        auc  = roc_auc_score(y_true, y_pred_prob)
        aupr = average_precision_score(y_true, y_pred_prob)

    return {
        "accuracy": acc, "precision": precision, "recall": recall,
        "specificity": specificity, "auc": auc, "aupr": aupr,
        "f1": f1, "mcc": mcc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, 'total': tp + tn + fp + fn,
    }


def compute_binary_metrics__alt(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    cutoff=0.5,
    n_downsample_runs: int = 100,
    random_state: int = 42,
):
    """
    Compute binary classification metrics.

    This function returns:
    1. Metrics on the original data distribution.
    2. Repeated random downsampling metrics, summarized by mean and standard deviation.

    Downsampling is applied only for auxiliary balanced evaluation.
    It should not replace metrics computed on the original validation/test distribution.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob)

    if y_true.shape[0] != y_pred_prob.shape[0]:
        raise ValueError("y_true and y_pred_prob must have the same length.")

    # Metrics on the original distribution
    metrics = _compute_metrics(
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        cutoff=cutoff,
    )

    metric_keys = list(metrics.keys())

    # If only one class exists, downsampling is impossible
    if len(np.unique(y_true)) < 2:
        metrics.update(_nan_downsample_summary(metric_keys))
        return metrics

    # Repeated random downsampling
    downsampled_results = []

    for i in range(n_downsample_runs):
        try:
            X = y_pred_prob.reshape(-1, 1)
            y = y_true

            rus = RandomUnderSampler(
                random_state=random_state + i,
            )

            X_resampled, y_resampled = rus.fit_resample(X, y)

            y_pred_prob_resampled = X_resampled.ravel()
            y_true_resampled = y_resampled

            run_metrics = _compute_metrics(
                y_true=y_true_resampled,
                y_pred_prob=y_pred_prob_resampled,
                cutoff=cutoff,
            )

            downsampled_results.append(run_metrics)

        except Exception:
            metrics.update(_nan_downsample_summary(metric_keys))
            return metrics

    # Summarize downsample metrics by mean and std
    for key in metric_keys:
        values = np.array(
            [run[key] for run in downsampled_results],
            dtype=float,
        )

        metrics[f"balanced_downsample_{key}_mean"] = np.nanmean(values)
        metrics[f"balanced_downsample_{key}_std"] = np.nanstd(values, ddof=1)

    return metrics

# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

def hyperparameters_queryer(*args):
    hp = {
        "lora_r":       4, #2, #16,
        "lora_alpha":   32,
        "lora_dropout": 0.1, #0.05,
        "num_heads":    8,      # must divide d_model (giga=1280: ok; micro=480: use 8 or 6)
        "dropout":      0.1,
        "pooling_mode": "attention",
        "batch_size":   16, #8,
        "max_epoch":    50,
        "pos_weight_fold":  0.5,
        "noncanonical_weight_fold":  10,
        "lr":           1e-4,
        "max_len":      512,    # RiNALMo trained up to ~1024; safe upper limit
    }
    if not args:
        return hp
    if len(args) == 1:
        return hp[args[0]]
    return {k: hp[k] for k in args}


# ══════════════════════════════════════════════════════════════════════════════
# Sequence length control
# ══════════════════════════════════════════════════════════════════════════════

def cut_transcript(length_limit: int, seq: str, position: int, buffer: int = 100):
    assert buffer >= 1

    if len(seq) <= length_limit:
        return seq, position

    if position < length_limit - buffer:          # keep 5' end
        return seq[:length_limit], position

    if len(seq) - position < buffer:              # too close to 3' end
        start = len(seq) - length_limit
        return seq[-length_limit:], position - start

    start = position - (length_limit - buffer)    # centre around marker
    return seq[start: position + buffer], position - start


def get_dataset(hp):

    df = pd.read_csv("input_data/df_data.csv")
    print(df.shape, flush=True)

    length_limit = hp["max_len"]
    buffer_size  = 100

    df[["seq", "position"]] = df.apply(
        lambda r: cut_transcript(length_limit, r["seq"], r["position"], buffer_size),
        axis=1, result_type="expand",
    )

    sequences, positions, labels, genes, sections = (
        df.seq.tolist(), df.position.tolist(),
        df.label.tolist(), df.gene_id.tolist(),
        list(map(lambda x: int(x != 'Annotated'), df.section)),
    )

    train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx = make_dataset(
        sequences, positions, labels, genes, sections,
        batch_size=hp["batch_size"],
    )
    
    return train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx
    

def evaluate_group(df, cutoff=0.5, sec_cufoff=dict()):
    rows = []
    for sec in df.section.unique():
        df_temp = df.loc[df.section == sec]
        cutoff = sec_cufoff.get(sec, cutoff)  # select specified cutoff by section
        perf_result = compute_binary_metrics__alt(
            df_temp.label,
            df_temp.yhat_prob,
            cutoff
        )
        # df_temp.label, df_temp.yhat, df_temp.yhat_prob)
        # pprint(perf_result)
        perf_result = {'section': sec} | perf_result
        rows.append(perf_result)
        # print('----')
    return pd.DataFrame.from_dict(rows, ).sort_values(by='section')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("run at:", datetime.now().strftime("%Y%m%d_%H%M%S"))

    hp = hyperparameters_queryer()

    train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx = get_dataset(hp)

    _, best_model_path = train_model(
        train_loader, val_loader, hp, pos_weight,
        naming_prefix="rinalmo_dora", patience=5,
    )

    model = load_model(best_model_path, hp, DEVICE)
    print('-- ignore annotated --')
    yhat_prob, yhat, labels_t = predict_batch(model, test_loader, DEVICE, ignore_sec=0)
    pprint(compute_binary_metrics(labels_t.numpy(), yhat.numpy(), yhat_prob.numpy()))
    print('-- all sites  --')
    yhat_prob, yhat, labels_t = predict_batch(model, test_loader, DEVICE, ignore_sec=-1)
    pprint(compute_binary_metrics(labels_t.numpy(), yhat.numpy(), yhat_prob.numpy()))

    df = pd.read_csv("input_data/df_data.csv")
    df_test = df.iloc[test_idx] 
    df_test['yhat_prob'] = yhat_prob.numpy().tolist()
    df_test['yhat'] = yhat.numpy().tolist()
    print(evaluate_group(df_test).to_string())

if __name__ == "__main__":
    main()