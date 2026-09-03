import random
import math
from typing import List, Tuple
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, fbeta_score
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
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# ── Model variant: rinalmo-micro (30M) | rinalmo-mega (150M) | rinalmo-giga (650M) ─
RINALMO_MODEL_ID = "multimolecule/rinalmo-micro"
# RINALMO_MODEL_ID = "multimolecule/rinalmo-mega"
# RINALMO_MODEL_ID = "multimolecule/rinalmo-giga"

# ── 全域只載入一次 tokenizer ──────────────────────────────────────────────────
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


# def get_criterion_fn(hyperparameters: dict, section_ratio: dict[int, float], section_neg_pos_ratio: dict[int, float]):
#     section_scaling = float(hyperparameters.get("section_scaling", 0.0))
#     section_label_scaling = float(hyperparameters.get("section_label_scaling", 1.5))
#     max_weight = hyperparameters.get("max_weight", 100)

#     def criterion_fn(logits, labels, sections):
#         sample_w = torch.ones_like(labels, dtype=torch.float32, device=logits.device)
#         for section in torch.unique(sections):
#             key = int(section.item())
#             mask = sections == section
#             sec_w = float(section_ratio.get(key, 1.0)) ** section_scaling * hyperparameters.get("section_weight_fold", {}).get(str(key), 1)
#             pos_w = float(section_neg_pos_ratio.get(key, 1.0)) ** section_label_scaling * hyperparameters.get("section_pos_weight_fold", {}).get(str(key), 1)
#             sample_w[mask] *= sec_w
#             sample_w[mask & (labels == 1)] *= pos_w
#         if max_weight is not None:
#             sample_w = torch.clamp(sample_w, max=float(max_weight))
#         return F.binary_cross_entropy_with_logits(logits, labels.float(), weight=sample_w)

#     return criterion_fn


# ══════════════════════════════════════════════════════════════════════════════
# Train loop
# ══════════════════════════════════════════════════════════════════════════════

def train_model(train_loader, val_loader, hp: dict,
                # section_ratio, section_neg_pos_ratio,
                pos_weight,
                patience=15):

    print("hyperparameters:", hp)
    model = build_model(hp, DEVICE)

    # criterion_fn = get_criterion_fn(
    #     hp,
    #     section_ratio,
    #     section_neg_pos_ratio,
    # )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight * hp.get("weight_fold", 1.0)
    )
    # eval_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.get("lr", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    best_score = -1
    no_improve = 0
    # run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_path  = f"model/{naming_prefix}_{run_id}.pt"
    # os.makedirs("model", exist_ok=True)
    save_path = 'model.pt'

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
    sd    = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Dataset factory
# ══════════════════════════════════════════════════════════════════════════════

def _stratified_group_split(indices, labels_a, groups_a, test_frac, seed):
    """
    StratifiedGroupKFold has no `test_size` arg — it's controlled via n_splits.
    We pick the closest n_splits to the requested fraction and take the first fold.
    """
    n_splits = max(2, round(1.0 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(indices, labels_a, groups=groups_a))
    return train_idx, test_idx


def make_dataset(sequences, labels, groups, sections, batch_size=8, training=True):
    assert len(sequences) == len(labels) == len(groups) == len(sections)
    print("labels:", dict(Counter(labels)))

    dataset = RNADataset(sequences, labels, sections)

    indices    = np.arange(len(dataset))
    groups_a   = np.array(groups)
    labels_a   = np.array(labels)
    sections_a = np.array(sections)

    # outer split: ~20% test, group-disjoint, label-stratified
    train_val_idx, test_idx = _stratified_group_split(
        indices, labels_a, groups_a, test_frac=0.2, seed=SEED,
    )

    # inner split: take ~10 of the inner pool as val
    inner_indices = indices[train_val_idx]
    inner_labels  = labels_a[train_val_idx]
    inner_groups  = groups_a[train_val_idx]
    train_rel, val_rel = _stratified_group_split(
        inner_indices, inner_labels, inner_groups,
        test_frac=0.1 / 0.8, seed=SEED,
    )
    train_idx = train_val_idx[train_rel]
    val_idx   = train_val_idx[val_rel]

    # sanity: no group overlap across splits
    assert set(groups_a[train_idx]).isdisjoint(set(groups_a[val_idx]))
    assert set(groups_a[train_idx]).isdisjoint(set(groups_a[test_idx]))
    assert set(groups_a[val_idx]).isdisjoint(set(groups_a[test_idx]))

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds   = Subset(dataset, val_idx.tolist())
    test_ds  = Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=training,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # # determine section weight (from training data only)
    # train_labels   = labels_a[train_idx]
    # train_sections = sections_a[train_idx]

    # section_counts = Counter(train_sections)
    # total_train    = len(train_idx)

    # section_ratio = {
    #     section: total_train / count
    #     for section, count in section_counts.items()
    # }

    # section_label_counts = defaultdict(lambda: Counter())
    # for section, label in zip(train_sections, train_labels):
    #     section_label_counts[int(section)][int(label)] += 1

    # section_neg_pos_ratio = {}
    # for section in section_counts:
    #     neg_count = section_label_counts[section][0]
    #     pos_count = section_label_counts[section][1]
    #     if pos_count == 0:
    #         section_neg_pos_ratio[int(section)] = 1.0
    #     else:
    #         section_neg_pos_ratio[int(section)] = neg_count / pos_count

    train_labels = labels_a[train_idx]
    num_pos  = (train_labels == 1).sum()
    num_neg  = (train_labels == 0).sum()
    pos_weight = torch.tensor(
        [num_neg / max(num_pos, 1)], dtype=torch.float32, device=DEVICE
    )
    print("train labels:", dict(Counter(train_labels)))
    print("val labels:",   dict(Counter(labels_a[val_idx])))
    print("test labels:",  dict(Counter(labels_a[test_idx])))

    # print("train sections:", dict(section_counts))
    # print("section_ratio:",  section_ratio)
    # print("section_neg_pos_ratio:", section_neg_pos_ratio)
    print("pos_weight:", pos_weight)

    return (
        train_loader, val_loader, test_loader,
        # section_ratio, section_neg_pos_ratio,
        pos_weight,
        train_idx, val_idx, test_idx,
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
        # "section_scaling": 0,
        # "section_weight_fold": {
        #     "0": 1,
        #     "1": 1,
        #     "2": 1,
        #     "3": 1,
        #     "4": 1
        # },
        # "section_label_scaling": 0,
        # "section_pos_weight_fold": {
        #     "0": 1,
        #     "1": 10,
        #     "2": 1,
        #     "3": 1,
        #     "4": 10
        # },
        'weight_fold': 0.5,
        "max_weight": 300,
        "lr":           1e-4,
        "left_window":  500,
        "right_window": 500,
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
    """
    Build a fixed-length window of (left_window + 1 + right_window) nucleotides
    centred on `seq[position]` (the marker). Missing sides are padded with
    literal '<pad>' tokens so the marker always lands at the same token index
    after tokenisation: CLS + left_window pads/nts, then marker.
    """
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


def get_dataset(df, hp, training=True):
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
    genes     = df.gene_id.tolist()
    sections  = list(map(lambda x: section_idx[x], df.section))

    return make_dataset(
        sequences, labels, genes, sections,
        batch_size=hp["batch_size"], training=training
    )

def select_best_cu_on_fscore(y, yhat_prob, beta=2):
    best_cu = 0.5
    best_score = -1
    for cu in yhat_prob.unique():
        yhat = [(1 if x > cu else 0) for x in yhat_prob]
        score = fbeta_score(y, yhat, zero_division=0, beta=beta)
        if score > best_score:
            best_cu = cu
            best_score = score
    return best_cu


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    print("run at:", datetime.now().strftime("%Y%m%d_%H%M%S"))

    hp = hyperparameters_queryer()
    df = pd.read_csv("../../input_data/df_data.csv").reset_index(drop=True)

    if len(sys.argv) > 1 and sys.argv[1] == 'train':  # train mode
        (
            train_loader, val_loader, test_loader,
            # section_ratio, section_neg_pos_ratio,
            pos_weight,
            train_idx, val_idx, test_idx,
        ) = get_dataset(df, hp)
        train_model(
            train_loader, val_loader, hp, pos_weight, patience=5,
        )
        model = load_model('model.pt', hp, DEVICE)
    else:
        (
            train_loader, val_loader, test_loader,
            # section_ratio, section_neg_pos_ratio,
            pos_weight,
            train_idx, val_idx, test_idx,
        ) = get_dataset(df, hp, False)

        model = load_model('model.pt', hp, DEVICE)

        # training set
        print('========== training set: all sites ==========')
        yhat_prob, yhat, labels_t = predict_batch(model, train_loader, DEVICE, ignore_sec=-1)
        df_train = df.iloc[train_idx].copy()
        df_train['yhat_prob'] = yhat_prob.numpy()
        df_train['yhat'] = yhat.numpy()
        print(evaluate_group(df_train).to_string())
    
    # val set
    print('========== val set: all sites ==========')
    yhat_prob, yhat, labels_t = predict_batch(model, val_loader, DEVICE, ignore_sec=-1)
    df_val = df.iloc[val_idx].copy()
    df_val['yhat_prob'] = yhat_prob.numpy()
    df_val['yhat'] = yhat.numpy()
    print(evaluate_group(df_val).to_string())
    sect_best_cut = {
        sec: select_best_cu_on_fscore(
            df_val.label[df_val.section == sec],
            df_val.yhat_prob[df_val.section == sec],
            beta=2)
        for sec in df_val.section.unique()
    }
    print('sect_best_cut:', sect_best_cut)
    print('--- cutoff adj ---')
    print(evaluate_group(df_val, sec_cutoff=sect_best_cut).to_string())
       
    # test set
    print('========== test set: all sites ==========')
    yhat_prob, yhat, labels_t = predict_batch(model, test_loader, DEVICE, ignore_sec=-1)
    pprint(compute_binary_metrics(labels_t.numpy(), yhat.numpy(), yhat_prob.numpy()))

    df_test = df.iloc[test_idx].copy()
    df_test['yhat_prob'] = yhat_prob.numpy()
    df_test['yhat'] = yhat.numpy()
    
    print(evaluate_group(df_test).to_string())
    print('--- cutoff adj ---')
    print(evaluate_group(df_test, sec_cutoff=sect_best_cut).to_string())


if __name__ == "__main__":
    main()