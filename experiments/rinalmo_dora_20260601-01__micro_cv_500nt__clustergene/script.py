import random
import math
import shutil
from typing import List, Tuple
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef
)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from pprint import pprint

# ── RiNALMo (multimolecule) ───────────────────────────────────────────────────
from multimolecule import RnaTokenizer, RiNALMoModel

# ── PEFT / DoRA ───────────────────────────────────────────────────────────────
from peft import get_peft_model, LoraConfig, TaskType

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# ── Model variant: rinalmo-micro (30M) | rinalmo-mega (150M) | rinalmo-giga (650M) ─
RINALMO_MODEL_ID = "multimolecule/rinalmo-micro"
# RINALMO_MODEL_ID = "multimolecule/rinalmo-mega"
# RINALMO_MODEL_ID = "multimolecule/rinalmo-giga"

# ── load tokenizer once ──────────────────────────────────────────────────
print(f"Loading tokenizer: {RINALMO_MODEL_ID}", flush=True)
_tokenizer = RnaTokenizer.from_pretrained(RINALMO_MODEL_ID)

# # RiNALMo tokenizer adds [CLS] at front and [EOS] at the end, so nucleotide
# # position should add 1 offset.
# _CLS_OFFSET = 1

# Pad token: RnaTokenizer recognises the literal string "<pad>" (lowercase!) and
# tokenises it to a single pad token. We use this for left/right padding around
# the marker so the marker nucleotide always lands at a known token index.
_PAD_STR = "<pad>"
_PAD_ID  = _tokenizer.pad_token_id
assert _PAD_ID is not None, "Tokenizer has no pad_token_id"
print(f"pad token id = {_PAD_ID}", flush=True)



def _compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, cutoff=0.5):
    """
    Compute binary classification metrics from true labels and predicted probabilities.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob)

    y_pred = (y_pred_prob > cutoff).astype(int)

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        aupr = float("nan")
    else:
        auc = roc_auc_score(y_true, y_pred_prob)
        aupr = average_precision_score(y_true, y_pred_prob)

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "auc": auc,
        "aupr": aupr,
        "f1": f1,
        "mcc": mcc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(tp + tn + fp + fn),
    }


def _nan_downsample_summary(metric_keys):
    """
    Return NaN mean/std for balanced downsample metrics.
    """

    out = {}

    for key in metric_keys:
        out[f"balanced_downsample_{key}_mean"] = float("nan")
        out[f"balanced_downsample_{key}_std"] = float("nan")

    return out

def compute_downsample_binary_metrics(
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


def evaluate_group(df, cutoff_default=0.5, sec_cutoff=None):
    if sec_cutoff is None:
        sec_cutoff = {}
    rows = [
        {'section': 'all'} | compute_downsample_binary_metrics(
            df.label.values,
            df.yhat_prob.values,
            cutoff_default),
        {'section': 'non-annotated'} | compute_downsample_binary_metrics(
            df.label[df.section != 'Annotated'].values,
            df.yhat_prob[df.section != 'Annotated'].values,
            cutoff_default),
        ]
    for sec in df.section.unique():
        df_temp = df.loc[df.section == sec]
        cutoff = sec_cutoff.get(sec, cutoff_default)  # select specified cutoff by section
        perf_result = compute_downsample_binary_metrics(
            df_temp.label.values,
            df_temp.yhat_prob.values,
            cutoff
        )
        # df_temp.label, df_temp.yhat, df_temp.yhat_prob)
        # pprint(perf_result)
        perf_result = {'section': sec} | perf_result
        rows.append(perf_result)
        # print('----')
    return pd.DataFrame.from_dict(rows, ).sort_values(by='section')


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class RNADataset(Dataset):
    """
    Fixed-window sequences (already padded with `<pad>` literals around the
    marker by `centralize_transcript`) + binary labels.
    Tokenisation is deferred to collate_fn.
    """

    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        sections: List[int],
    ):
        assert len(sequences) == len(labels) == len(sections)
        self.sequences = sequences
        self.labels = labels
        self.sections = sections

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            torch.tensor(self.labels[idx], dtype=torch.float),
            self.sections[idx],
        )


def collate_fn(batch):
    seqs, labels, sections = zip(*batch)

    encoding = _tokenizer(
        list(seqs),
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids      = encoding["input_ids"]       # [B, L]
    attention_mask = encoding["attention_mask"]  # [B, L]

    # The tokenizer treats embedded <pad> tokens as real tokens (mask = 1).
    # We want the model to ignore them, so zero out attention on every pad id.
    attention_mask = attention_mask.masked_fill(input_ids == _PAD_ID, 0)

    lengths    = attention_mask.sum(dim=1)
    labels_t   = torch.stack(list(labels))
    sections_t = torch.tensor(sections, dtype=torch.float)
    return input_ids, attention_mask, lengths, labels_t, sections_t


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class RiNALMoClassifier(nn.Module):
    """
    RiNALMo backbone (DoRA fine-tuned) + classification head.

    pooling_mode:
      "marker"    – only the [CLS] token hidden state
      "attention" – [CLS] + cls-attended global context (concatenated)
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
        lengths:        torch.Tensor,   # [B]  (unused, kept for API compat)
    ) -> torch.Tensor:                  # [B]  raw logits

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state   # [B, L, D]

        # [CLS] token sits at position 0
        B = hidden.size(0)
        batch_idx = torch.arange(B, device=hidden.device)
        cls_h = hidden[batch_idx, 0]  # [B, D]

        if self.pooling_mode == "marker":
            h = cls_h                                   # [B, D]
        elif self.pooling_mode == "attention":
            padding_mask = (attention_mask == 0)        # [B, L]  True = pad
            global_h, _ = self.marker_attention(
                query=cls_h.unsqueeze(1),               # [B, 1, D]
                key=hidden,
                value=hidden,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            global_h = global_h.squeeze(1)              # [B, D]
            h = torch.cat([cls_h, global_h], dim=-1)    # [B, 2D]

        logits = self.classifier(self.dropout(h))       # [B, 1]
        return logits.squeeze(-1)                       # [B]


# ══════════════════════════════════════════════════════════════════════════════
# Build backbone with DoRA
# ══════════════════════════════════════════════════════════════════════════════

def build_dora_backbone(lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Load RiNALMo and wrap with DoRA adapters via PEFT.
    """
    print(f"Loading backbone: {RINALMO_MODEL_ID}", flush=True)
    print(
        f"  tokenizer vocab_size={_tokenizer.vocab_size}, "
        f"len(tokenizer)={len(_tokenizer)}",
        flush=True,
    )
    base_model = RiNALMoModel.from_pretrained(RINALMO_MODEL_ID)
    print(
        f"  base_model vocab_size={base_model.config.vocab_size}, "
        f"emb.shape={base_model.embeddings.word_embeddings.weight.shape}",
        flush=True,
    )

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
    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable()
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

def train_one_epoch(model, dataloader, optimizer, criterion, device, threshold=0.5):
    model.train()
    total_loss = total_samples = 0
    all_logits, all_labels = [], []

    for input_ids, attention_mask, lengths, labels, sections in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        lengths        = lengths.to(device)
        labels         = labels.to(device).float()
        sections       = sections.to(device).float()

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, lengths)
        loss   = criterion(logits, labels)
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

    for input_ids, attention_mask, lengths, labels, sections in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        lengths        = lengths.to(device)
        labels         = labels.to(device).float()
        sections       = sections.to(device).float()

        logits = model(input_ids, attention_mask, lengths)

        # filter out sections we don't want in eval (e.g. section=0)
        keep = (sections != ignore_sec)
        logits = logits[keep]
        labels = labels[keep]
        if labels.numel() == 0:
            continue

        loss = criterion(logits, labels)

        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    return _compute_metrics_from_logits(
        torch.cat(all_logits), torch.cat(all_labels),
        total_loss, total_samples, threshold,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Train loop
# ══════════════════════════════════════════════════════════════════════════════

def train_model(train_loader, val_loader, hp: dict,
                # section_ratio, section_neg_pos_ratio,
                pos_weight,
                patience=15,
                save_path: str = "model.pt"):

    print("hyperparameters:", hp)
    model = build_model(hp, DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight * hp.get("weight_fold", 1.0)
    )
    # eval_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.get("lr", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    best_score = -1
    no_improve = 0

    for epoch in range(1, hp.get("max_epoch", 50) + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va = evaluate(model, val_loader, criterion, DEVICE, ignore_sec=hp.get('ignore_sec', -1))
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | lr={lr_now:.2e} | "
            f"train loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} "
            f"prec={tr['precision']:.4f} rec={tr['recall']:.4f} f1={tr['f1']:.4f} "
            f"aupr={tr['aupr']:.4f} | "
            f"val   loss={va['loss']:.4f} acc={va['accuracy']:.4f} "
            f"prec={va['precision']:.4f} rec={va['recall']:.4f} f1={va['f1']:.4f} "
            f"aupr={va['aupr']:.4f}",
            flush=True,
        )
        scheduler.step(va["aupr"])

        if va["aupr"] > best_score:
            best_score = va["aupr"]
            no_improve = 0
            print(f"  -> saved best model to {save_path}", flush=True)
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  -> early stopping at epoch {epoch}", flush=True)
                break

    print(f"Best val AUPR: {best_score:.4f}")

    # Reload best checkpoint into the SAME model object to avoid a second
    # build_model() (which can hit vocab-size drift in multimolecule + PEFT).
    if best_score > -1 and os.path.exists(save_path):
        print(f"Reloading best checkpoint from {save_path} into trained model",
              flush=True)
        sd = torch.load(save_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(sd)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_batch(model, dataloader, device, threshold=0.5, ignore_sec=-1):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for input_ids, attention_mask, lengths, labels, sections in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        lengths        = lengths.to(device)

        logits = model(input_ids, attention_mask, lengths).cpu()
        probs  = torch.sigmoid(logits)
        preds  = (probs >= threshold).float()

        keep = (sections != ignore_sec)
        all_probs.append(probs[keep])
        all_preds.append(preds[keep])
        all_labels.append(labels[keep])

    return torch.cat(all_probs), torch.cat(all_preds), torch.cat(all_labels)

def load_model(path: str, hp: dict, device: torch.device = DEVICE) -> RiNALMoClassifier:
    model = build_model(hp, device)
    sd    = torch.load(path, map_location=device, weights_only=True)

    # Defensive: backbone word_embeddings size in the checkpoint may differ
    # from a freshly-built backbone (multimolecule / PEFT vocab drift).
    # Resize the current model's embedding to match the checkpoint before load.
    emb_key = "backbone.base_model.model.embeddings.word_embeddings.weight"
    if emb_key in sd:
        ckpt_vocab = sd[emb_key].shape[0]
        try:
            cur_emb = model.backbone.base_model.model.embeddings.word_embeddings
            cur_vocab = cur_emb.weight.shape[0]
            if ckpt_vocab != cur_vocab:
                print(
                    f"[load_model] backbone vocab mismatch "
                    f"(current={cur_vocab}, checkpoint={ckpt_vocab}); "
                    f"resizing to {ckpt_vocab}",
                    flush=True,
                )
                model.backbone.base_model.model.resize_token_embeddings(ckpt_vocab)
        except AttributeError:
            # different PEFT wrapping structure; fall through to strict load
            pass

    model.load_state_dict(sd)
    model.eval()
    
    try:
        model.backbone.gradient_checkpointing_disable()
    except AttributeError:
        pass
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Repeated CV dataset factory
# ══════════════════════════════════════════════════════════════════════════════

def _build_loaders_from_indices(dataset, train_idx, val_idx, batch_size, training):
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds   = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=training, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
    )
    return train_loader, val_loader


def make_cv_folds(
    sequences, labels, groups, sections,
    n_splits: int = 5,
    n_repeats: int = 5,
    batch_size: int = 8,
    training: bool = True,
):
    """
    Generator over repeated StratifiedGroupKFold splits.

    Yields per fold:
        repeat_idx, fold_idx,
        train_loader, val_loader,
        pos_weight,
        train_idx, val_idx
    """
    assert len(sequences) == len(labels) == len(groups) == len(sections)
    print("labels (full):", dict(Counter(labels)), flush=True)

    dataset    = RNADataset(sequences, labels, sections)
    indices    = np.arange(len(dataset))
    labels_a   = np.array(labels)
    groups_a   = np.array(groups)

    for repeat_idx in range(n_repeats):
        seed = SEED + repeat_idx
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed,
        )

        for fold_idx, (train_rel, val_rel) in enumerate(
            sgkf.split(indices, labels_a, groups=groups_a)
        ):
            train_idx = indices[train_rel]
            val_idx   = indices[val_rel]

            # sanity: group-disjoint
            assert set(groups_a[train_idx]).isdisjoint(set(groups_a[val_idx])), \
                f"group leak at repeat={repeat_idx} fold={fold_idx}"

            train_loader, val_loader = _build_loaders_from_indices(
                dataset, train_idx, val_idx, batch_size, training,
            )

            train_labels = labels_a[train_idx]
            num_pos = (train_labels == 1).sum()
            num_neg = (train_labels == 0).sum()
            pos_weight = torch.tensor(
                [num_neg / max(num_pos, 1)],
                dtype=torch.float32, device=DEVICE,
            )

            print(
                f"\n=== repeat {repeat_idx+1}/{n_repeats} | "
                f"fold {fold_idx+1}/{n_splits} ===", flush=True,
            )
            print("train labels:", dict(Counter(train_labels)), flush=True)
            print("val   labels:", dict(Counter(labels_a[val_idx])), flush=True)
            print("pos_weight:", pos_weight, flush=True)

            yield (
                repeat_idx, fold_idx,
                train_loader, val_loader,
                pos_weight,
                train_idx, val_idx,
            )


def get_cv_dataset(df, hp, n_splits=5, n_repeats=5, training=True):
    """
    Returns a generator over (repeat, fold) splits.
    """
    df = df.copy()
    print(df.shape, flush=True)

    left_window  = hp["left_window"]
    right_window = hp["right_window"]

    df["seq"] = [
        centralize_transcript(left_window, right_window, s, p)
        for s, p in zip(df["seq"].values, df["position"].values)
    ]

    section_idx = {
        'Annotated':   0,
        '5UTRnonATG':  1,
        '5UTRATG':     2,
        'CDSATG':      3,
        'CDSnonATG':   4,
        '3UTRATG':     0,  # too small, merge to annotated
        '3UTRnonATG':  0,  # too small, merge to annotated
    }

    sequences = df.seq.tolist()
    labels    = df.label.tolist()
    cluster     = df.gene_id.tolist()
    # cluster     = df.cluster.tolist()
    sections  = list(map(lambda x: section_idx[x], df.section))

    return make_cv_folds(
        sequences, labels, cluster, sections,
        n_splits=n_splits,
        n_repeats=n_repeats,
        batch_size=hp["batch_size"],
        training=training,
    )


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


def summarize_cv_results(df_all: pd.DataFrame, metric_cols=None):
    """
    df_all columns: repeat, fold, section, and metrics
    return (mean_df, std_df), section as columns
    """
    if metric_cols is None:
        skip = {"repeat", "fold", "section"}
        metric_cols = [c for c in df_all.columns if c not in skip]

    grouped = df_all.groupby("section", as_index=True)[metric_cols]
    mean_df = grouped.mean(numeric_only=True)
    std_df  = grouped.std(numeric_only=True, ddof=1)
    return mean_df, std_df


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

def hyperparameters_queryer(*args):
    hp = {
        "lora_r":       4,
        "lora_alpha":   32,
        "lora_dropout": 0.1,
        "num_heads":    8,      # must divide d_model (giga=1280: ok; micro=480: use 8 or 6)
        "dropout":      0.1,
        "pooling_mode": "attention",
        "batch_size":   16,
        "max_epoch":    50,
        'weight_fold': 0.5,
        "max_weight": 300,
        "lr":           1e-4,
        "left_window":  300,
        "right_window": 200,
        'ignore_sec': -1
    }
    if not args:
        return hp
    if len(args) == 1:
        return hp[args[0]]
    return {k: hp[k] for k in args}


# ══════════════════════════════════════════════════════════════════════════════
# Sequence length control
# ══════════════════════════════════════════════════════════════════════════════
def centralize_transcript(left_window: int, right_window: int, seq: str, position: int):
    # left side: `left_window` nt before the marker
    nt_before = position
    if nt_before >= left_window:
        left_padding = ""
        lb = position - left_window
    else:
        left_padding = _PAD_STR * (left_window - nt_before)
        lb = 0

    # right side: `right_window` nt after the marker
    nt_after = len(seq) - 1 - position
    if nt_after >= right_window:
        right_padding = ""
        rb = position + right_window + 1   # +1 because slice is exclusive
    else:
        right_padding = _PAD_STR * (right_window - nt_after)
        rb = len(seq)

    return left_padding + seq[lb:rb] + right_padding


# ══════════════════════════════════════════════════════════════════════════════
# Main (CV version)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    print("run at:", datetime.now().strftime("%Y%m%d_%H%M%S"), flush=True)

    # CLI: python script.py [train|eval] [n_splits] [n_repeats]
    mode      = sys.argv[1] if len(sys.argv) > 1 else "eval"
    n_splits  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    print(f"mode={mode} | n_splits={n_splits} | n_repeats={n_repeats}", flush=True)

    hp = hyperparameters_queryer()
    df = pd.read_csv("../../input_data/df_data.csv").reset_index(drop=True)

    all_val_rows   = []   # val set evaluate_group result of each fold
    fold_summaries = []   # summary of each fold

    cv_iter = get_cv_dataset(
        df, hp,
        n_splits=n_splits,
        n_repeats=n_repeats,
        training=(mode == "train"),
    )

    for (
        repeat_idx, fold_idx,
        train_loader, val_loader,
        pos_weight,
        train_idx, val_idx,
    ) in cv_iter:

        save_path = f"model_r{repeat_idx}_f{fold_idx}.pt"

        if mode == "train":
            model = train_model(
                train_loader, val_loader, hp, pos_weight,
                patience=4, save_path=save_path,
            )
        else:
            if not Path(save_path).is_file():
                print("model not exist: ", save_path)
                continue
            model = load_model(save_path, hp, DEVICE)

        # ── val set evaluation ────────────────────────────────────────────
        yhat_prob, yhat, labels_t = predict_batch(
            model, val_loader, DEVICE, ignore_sec=-1,
        )
        df_val = df.iloc[val_idx].copy()
        df_val["yhat_prob"] = yhat_prob.numpy()
        df_val["yhat"]      = yhat.numpy()

        df_grp = evaluate_group(df_val)
        df_grp["repeat"] = repeat_idx
        df_grp["fold"]   = fold_idx
        all_val_rows.append(df_grp)

        # quick per-fold log
        all_row = df_grp[df_grp.section == "all"].iloc[0]
        fold_summaries.append({
            "repeat": repeat_idx, "fold": fold_idx,
            "aupr": all_row.get("aupr", float("nan")),
            "auc":  all_row.get("auc",  float("nan")),
            "f1":   all_row.get("f1",   float("nan")),
            "mcc":  all_row.get("mcc",  float("nan")),
        })
        print(
            f"[r{repeat_idx} f{fold_idx}] val all-section: "
            f"aupr={all_row.get('aupr', float('nan')):.4f} "
            f"auc={all_row.get('auc',  float('nan')):.4f} "
            f"f1={all_row.get('f1',    float('nan')):.4f} "
            f"mcc={all_row.get('mcc',  float('nan')):.4f}",
            flush=True,
        )

        # release GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── aggregate across folds & repeats ─────────────────────────────────────
    df_all = pd.concat(all_val_rows, ignore_index=True)

    out_csv = f"cv_results_{n_splits}fold_{n_repeats}repeat.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nper-fold detail saved to: {out_csv}", flush=True)

    print("\n========== per-fold summary (all-section) ==========", flush=True)
    print(pd.DataFrame(fold_summaries).to_string(index=False), flush=True)

    mean_df, std_df = summarize_cv_results(df_all)

    print(f"\n========== mean across {n_splits}x{n_repeats} folds ==========",
          flush=True)
    print(mean_df.to_string(), flush=True)

    print(f"\n========== std across {n_splits}x{n_repeats} folds ==========",
          flush=True)
    print(std_df.to_string(), flush=True)


if __name__ == "__main__":
    main()