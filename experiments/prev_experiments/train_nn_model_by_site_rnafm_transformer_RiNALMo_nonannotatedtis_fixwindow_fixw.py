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
os.environ["PYTORCH_CUDAsb_ALLOC_CONF"] = "expandable_segments:True"

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

# RiNALMo tokenizer adds [CLS] at front and [EOS] at the end, so nucleotide
# position should add 1 offset.
_CLS_OFFSET = 1

# Pad token: RnaTokenizer recognises the literal string "<pad>" (lowercase!) and
# tokenises it to a single pad token. We use this for left/right padding around
# the marker so the marker nucleotide always lands at a known token index.
_PAD_STR = "<pad>"
_PAD_ID  = _tokenizer.pad_token_id
assert _PAD_ID is not None, "Tokenizer has no pad_token_id"
print(f"pad token id = {_PAD_ID}", flush=True)

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

    # lengths    = attention_mask.sum(dim=1)
    labels_t   = torch.stack(list(labels))
    sections_t = torch.tensor(sections, dtype=torch.float)
    return input_ids, attention_mask, labels_t, sections_t



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
    peft_model.gradient_checkpointing_enable()  # 
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
 
    for input_ids, attention_mask, labels, sections in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device).float()
        sections = sections.to(device).float()
 
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
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
 
    for input_ids, attention_mask, labels, sections in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device).float()
        sections       = sections.to(device).float()
 
        logits = model(input_ids, attention_mask)
        
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

    for input_ids, attention_mask, labels, sections in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = model(input_ids, attention_mask)
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

def make_dataset(sequences, labels, groups, sections, batch_size=8):
    assert len(sequences) == len(labels) == len(groups) == len(sections)
    print("labels:", dict(Counter(labels)))

    dataset = RNADataset(sequences, labels, sections)
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
        "left_window":  411,
        "right_window": 101,
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
    Build a fixed-length window around `position` (0-indexed nucleotide) by
    cropping the sequence and padding the missing sides with the literal
    '<pad>' token string. RnaTokenizer recognises '<pad>' (lowercase) and maps
    it to the pad token id, so the marker nucleotide always lands at a known
    token index after tokenisation.
    """
    if position > left_window:
        left_padding = ""
        lb = position - left_window
    else:
        left_padding = _PAD_STR * (left_window - position)
        lb = 0

    if len(seq) - position > right_window:
        right_padding = ""
        rb = position + right_window
    else:
        rb = len(seq)
        right_padding = _PAD_STR * (right_window - (len(seq) - position))

    return left_padding + seq[lb:rb] + right_padding


def get_dataset(hp):

    df = pd.read_csv("input_data/df_data.csv")
    print(df.shape, flush=True)

    left_window  = hp["left_window"]
    right_window = hp["right_window"]

    df["seq"] = df.apply(
        lambda r: centralize_transcript(left_window, right_window, r["seq"], r["position"]),
        axis=1,
    )

    sequences, labels, genes, sections = (
        df.seq.tolist(),
        df.label.tolist(), df.gene_id.tolist(),
        list(map(lambda x: int(x != 'Annotated'), df.section)),
    )

    train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx = make_dataset(
        sequences, labels, genes, sections,
        batch_size=hp["batch_size"],
    )
    
    return train_loader, val_loader, test_loader, pos_weight, train_idx, val_idx, test_idx
    

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
    print('================= only non-indicated site =================')
    yhat_prob, yhat, labels_t = predict_batch(model, test_loader, DEVICE, ignore_sec=0)
    pprint(compute_binary_metrics(labels_t.numpy(), yhat.numpy(), yhat_prob.numpy()))
    print('================= all site =================')
    yhat_prob, yhat, labels_t = predict_batch(model, test_loader, DEVICE, ignore_sec=-1)
    pprint(compute_binary_metrics(labels_t.numpy(), yhat.numpy(), yhat_prob.numpy()))


if __name__ == "__main__":
    main()