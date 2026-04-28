import random
import math
from typing import List, Tuple
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from pprint import pprint

# ── AIDO.RNA ──────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel

# ── PEFT / DoRA ───────────────────────────────────────────────────────────────
from peft import get_peft_model, LoraConfig, TaskType

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# ── AIDO.RNA HuggingFace model id ─────────────────────────────────────────────
AIDO_RNA_MODEL_ID = "genbio-ai/AIDO.RNA-1.6B"   # adjust if using a different variant  # aido_rna_1b600m

PAD_IDX = 0   # kept for collate_fn interface; actual padding handled by tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════════════════

def encode_position(position: int, n: int) -> List[int]:
    """One-hot position indicator vector of length n."""
    assert position < n, f"position {position} >= seq len {n}"
    vec = [0] * n
    vec[position] = 1
    return vec


def get_sampler(labels_np):
    from torch.utils.data import WeightedRandomSampler
    class_counts = np.bincount(labels_np)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_np]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


class RNADataset(Dataset):
    """
    Stores raw sequences (strings) + position indicators + binary labels.
    Tokenisation is deferred to collate_fn so the HF tokenizer can batch-pad.
    """

    def __init__(
        self,
        sequences: List[str],
        positions: List[int],
        labels: List[int],
    ):
        assert len(sequences) == len(positions) == len(labels)
        for seq, pos in zip(sequences, positions):
            assert len(seq) > pos, f"position {pos} out of range for seq len {len(seq)}"

        self.sequences = sequences                    # keep as raw strings
        self.positions = positions
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            self.sequences[idx],                                          # str
            self.positions[idx],                                          # int
            torch.tensor(self.labels[idx], dtype=torch.float),           # scalar
        )


# def make_collate_fn(tokenizer):
#     """
#     Returns a collate_fn that tokenizes a batch of raw RNA strings with the
#     AIDO.RNA tokenizer and also builds per-sample position indicator tensors
#     that are aligned to the tokenizer token ids.

#     NOTE: AIDO.RNA uses a character-level (or near character-level) BPE tokenizer.
#     For a nucleotide-level tokenizer each character maps to one token, so
#     `position` in nucleotide space == token index.  If you switch to a
#     multi-char BPE tokenizer you will need to adjust the position mapping via
#     the tokenizer's `char_to_token` offset mapping.
#     """

#     def collate_fn(batch):
#         seqs, positions, labels = zip(*batch)

#         # ── tokenise ──────────────────────────────────────────────────────────
#         encoding = tokenizer(
#             list(seqs),
#             return_tensors="pt",
#             padding=True,
#             truncation=False,          # caller is responsible for length control
#             return_offsets_mapping=True,
#         )
#         input_ids = encoding["input_ids"]           # [B, L_tok]
#         attention_mask = encoding["attention_mask"] # [B, L_tok]
#         offsets = encoding["offset_mapping"]        # [B, L_tok, 2]

#         B, L_tok = input_ids.shape

#         # ── map nucleotide position → token index ─────────────────────────────
#         # For each sample find the token whose char-span contains `position`.
#         pos_features = torch.zeros(B, L_tok, dtype=torch.float)
#         for i, pos in enumerate(positions):
#             token_idx = None
#             for t, (start, end) in enumerate(offsets[i].tolist()):
#                 if start <= pos < end:
#                     token_idx = t
#                     break
#             if token_idx is None:
#                 # fallback: use the last non-padding token
#                 token_idx = int(attention_mask[i].sum()) - 1
#             pos_features[i, token_idx] = 1.0

#         lengths = attention_mask.sum(dim=1)          # [B]
#         labels_t = torch.stack(list(labels))         # [B]

#         return input_ids, attention_mask, pos_features, lengths, labels_t

#     return collate_fn
from modelgenerator.tasks import Embed

def get_collate_fn(model_id):
    def collate_fn(batch):
        wrapper = Embed.from_config({"model.backbone": model_id}).eval()
        seqs, positions, labels = zip(*batch)
        
        transformed = wrapper.transform({"sequences": list(seqs)})
        input_ids      = transformed["input_ids"]       # [B, L]
        attention_mask = transformed["attention_mask"]  # [B, L]
        
        B, L = input_ids.shape
        pos_features = torch.zeros(B, L, dtype=torch.float)
        for i, pos in enumerate(positions):
            pos_features[i, pos + 1] = 1.0  # +1 for CLS token
        
        lengths  = attention_mask.sum(dim=1)
        labels_t = torch.stack(list(labels))
        return input_ids, attention_mask, pos_features, lengths, labels_t
    return collate_fn

class AIDORNAClassifier(nn.Module):
    """
    AIDO.RNA backbone (frozen except for DoRA adapters) + lightweight
    classifier head that attends from the target-position token to the
    full sequence context.

    Positional / frame embeddings from the original model are commented out;
    the backbone's own positional encoding is used instead.
    """

    def __init__(
        self,
        backbone: nn.Module,           # PEFT-wrapped AIDO.RNA
        d_model: int,                  # hidden size of the backbone
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling_mode: str = "attention",  # "marker" | "mean" | "attention"
    ):
        super().__init__()

        valid_modes = {"marker", "mean", "attention"}
        assert pooling_mode in valid_modes, f"pooling_mode must be one of {valid_modes}"

        self.backbone = backbone
        self.d_model = d_model
        self.pooling_mode = pooling_mode

        # ── (commented out) position / frame embeddings ──────────────────────
        # self.position_embedding = nn.Embedding(max_len * 2, d_model)
        # self.frame_embedding    = nn.Embedding(math.ceil(2 * max_len / 3) + 1, d_model)

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

    # ── pooling helper ────────────────────────────────────────────────────────

    def _masked_mean_pool(
        self,
        x: torch.Tensor,                  # [B, L, D]
        attention_mask: torch.Tensor,      # [B, L]  1=valid, 0=pad
    ) -> torch.Tensor:                     # [B, D]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)       # [B, 1]
        return (x * mask).sum(dim=1) / denom

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        pos_features: torch.Tensor,    # [B, L]  one-hot token position indicator
        lengths: torch.Tensor,         # [B]  (unused internally, kept for API compat)
    ) -> torch.Tensor:                 # [B]  raw logits

        # ── backbone (AIDO.RNA + DoRA adapters) ───────────────────────────────
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # last hidden state: [B, L, D]
        hidden = outputs.last_hidden_state

        # ── (commented out) add relative positional / frame embeddings ────────
        # marker_pos = pos_features.argmax(dim=1)
        # pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        # relative_pos_id = pos_ids - marker_pos.unsqueeze(1) + self.max_len - 1
        # hidden = hidden + self.position_embedding(relative_pos_id)
        # relative_frame_ids = relative_pos_id // 3
        # hidden = hidden + self.frame_embedding(relative_frame_ids)

        B = hidden.size(0)
        batch_idx = torch.arange(B, device=hidden.device)
        marker_idx = pos_features.argmax(dim=1)   # [B]
        marker_h = hidden[batch_idx, marker_idx]  # [B, D]

        # ── pooling ───────────────────────────────────────────────────────────
        if self.pooling_mode == "marker":
            h = marker_h                                                  # [B, D]

        elif self.pooling_mode == "attention":
            padding_mask = (attention_mask == 0)

            # marker token attend to whole sequence
            global_h, _ = self.marker_attention(
                query=marker_h.unsqueeze(1),   # [B, 1, D]
                key=hidden,
                value=hidden,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            global_h = global_h.squeeze(1)    # [B, D]

            cls_h = hidden[:, 0, :]           # [B, D] CLS Token

            # concat: marker(candidate TIS site) + CLS + marker attend to whole seq
            h = torch.cat([marker_h, global_h, cls_h], dim=-1)  # [B, 3D]

        logits = self.classifier(self.dropout(h))   # [B, 1]
        return logits.squeeze(-1)                   # [B]


def build_dora_backbone(model_id, lora_r=16 , lora_alpha=32, lora_dropout=0.05):
    wrapper = Embed.from_config({"model.backbone": model_id})  # aido_rna_1b600m
    base_model = wrapper.model
    
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=True,
        bias="none",
        target_modules=["query", "key", "value", "dense"],
    )
    return get_peft_model(base_model, lora_cfg), wrapper.transform

def build_model(hp: dict, device: torch.device) -> AIDORNAClassifier:
    backbone, _ = build_dora_backbone(
        model_id=AIDO_RNA_MODEL_ID,
        lora_r=hp.get("lora_r", 16),
        lora_alpha=hp.get("lora_alpha", 32),
        lora_dropout=hp.get("lora_dropout", 0.05),
    )

    # Get hidden dim from backbone config
    d_model = backbone.config.hidden_size

    model = AIDORNAClassifier(
        backbone=backbone,
        d_model=d_model,
        num_heads=hp.get("num_heads", 8),
        dropout=hp.get("dropout", 0.1),
        pooling_mode=hp.get("pooling_mode", "attention"),
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  |  Trainable: {trainable:,}", flush=True)
    return model


def train_one_epoch(model, dataloader, optimizer, criterion, device, threshold=0.5):
    model.train()
    total_loss = total_samples = 0
    tp = tn = fp = fn = 0

    for input_ids, attention_mask, pos_features, lengths, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_features = pos_features.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, pos_features, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"loss": total_loss / max(total_samples, 1),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = total_samples = 0
    tp = tn = fp = fn = 0
    all_labels, all_probs = [], []

    for input_ids, attention_mask, pos_features, lengths, labels in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_features   = pos_features.to(device)
        lengths        = lengths.to(device)
        labels         = labels.to(device).float()

        logits = model(input_ids, attention_mask, pos_features, lengths)
        loss   = criterion(logits, labels)

        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())

    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    spec = tn  / max(tn + fp, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    mcc_d = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc  = (tp*tn - fp*fn) / np.sqrt(mcc_d) if mcc_d else 0.0

    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()
    if len(np.unique(all_labels)) < 2:
        auc = aupr = float("nan")
    else:
        auc  = roc_auc_score(all_labels, all_probs)
        aupr = average_precision_score(all_labels, all_probs)

    return {"loss": total_loss / max(total_samples, 1),
            "accuracy": acc, "precision": prec, "recall": rec,
            "specificity": spec, "f1": f1, "mcc": mcc,
            "auc": auc, "aupr": aupr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def train_model(train_loader, val_loader, hp: dict, pos_weight,
                naming_prefix="model", patience=10):

    print("hyperparameters:", hp)
    model = build_model(hp, DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight * hp.get("weight_fold", 1.0)
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.get("lr", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    best_score = -1
    no_improve = 0
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model/{naming_prefix}_{run_id}.pt"
    os.makedirs("model", exist_ok=True)

    for epoch in range(1, hp.get("max_epoch", 50) + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va = evaluate(model, val_loader, criterion, DEVICE)
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
        scheduler.step(va['aupr'])

        if va['aupr'] > best_score:
            best_score = va['aupr']
            no_improve = 0
            print(f"  -> saved best model to {save_path}", flush=True)
            sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(sd, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  -> early stopping at epoch {epoch}", flush=True)
                break

    print(f"Best val F1: {best_score:.4f}")
    return model, save_path




@torch.no_grad()
def predict_batch(model, dataloader, device, threshold=0.5):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for input_ids, attention_mask, pos_features, lengths, labels in dataloader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_features   = pos_features.to(device)
        lengths        = lengths.to(device)

        logits = model(input_ids, attention_mask, pos_features, lengths)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= threshold).float()

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels)

    return torch.cat(all_probs), torch.cat(all_preds), torch.cat(all_labels)


def load_model(path: str, hp: dict, device: torch.device = DEVICE) -> AIDORNAClassifier:
    model = build_model(hp, device)
    sd = torch.load(path, map_location=device)
    core = model.module if isinstance(model, nn.DataParallel) else model
    core.load_state_dict(sd)
    model.eval()
    return model


def make_dataset(sequences, positions, labels, groups, batch_size=8):
    assert len(sequences) == len(positions) == len(labels) == len(groups)
    print("labels:", dict(Counter(labels)))

    dataset = RNADataset(sequences, positions, labels)
    indices  = np.arange(len(dataset))
    groups_a = np.array(groups)
    labels_a = np.array(labels)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_idx, test_idx = next(gss_test.split(indices, labels_a, groups=groups_a))

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=SEED)
    train_rel, val_rel = next(
        gss_val.split(indices[train_val_idx], labels_a[train_val_idx],
                      groups=groups_a[train_val_idx])
    )
    train_idx = train_val_idx[train_rel]
    val_idx   = train_val_idx[val_rel]

    # sanity check
    assert set(groups_a[train_idx]).isdisjoint(set(groups_a[val_idx]))
    assert set(groups_a[train_idx]).isdisjoint(set(groups_a[test_idx]))
    assert set(groups_a[val_idx]).isdisjoint(set(groups_a[test_idx]))

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds   = Subset(dataset, val_idx.tolist())
    test_ds  = Subset(dataset, test_idx.tolist())

    train_labels = labels_a[train_idx]
    num_pos = (train_labels == 1).sum()
    num_neg = (train_labels == 0).sum()
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32, device=DEVICE)

    # collate = make_collate_fn(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=get_collate_fn(AIDO_RNA_MODEL_ID))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=get_collate_fn(AIDO_RNA_MODEL_ID))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=get_collate_fn(AIDO_RNA_MODEL_ID))

    return train_loader, val_loader, test_loader, pos_weight


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray):
    """
    y_true: [N] 0/1
    y_pred: [N] 0/1
    """

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # confusion matrix: [[tn, fp],
    #                    [fn, tp]]
    # Pass labels=[0, 1] to ensure a 2x2 confusion matrix even if only one class is present
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # MCC
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        mcc = 0.0
    else:
        mcc = (tp * tn - fp * fn) / np.sqrt(denom)

    # AUC / AUPR is not defined when label only single class
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        aupr = float("nan")
    else:
        auc = roc_auc_score(y_true, y_pred_prob)
        aupr = average_precision_score(y_true, y_pred_prob)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "auc": auc,
        "aupr": aupr,
        "f1": f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def hyperparameters_queryer(*args):
    hp = {
        "lora_r":       16,
        "lora_alpha":   32,
        "lora_dropout": 0.05,
        "num_heads":    8,
        "dropout":      0.1,
        "pooling_mode": "attention",
        "batch_size":   4, #8,
        "max_epoch":    50,
        "weight_fold":  0.1,
        "lr":           1e-4,       # lower LR suits fine-tuning
        "max_len":      200, #2048,
    }
    if not args:
        return hp
    if len(args) == 1:
        return hp[args[0]]
    return {k: hp[k] for k in args}



def main():
    print("run at:", datetime.now().strftime("%Y%m%d_%H%M%S"))

    hp = hyperparameters_queryer()

    # tokenizer = AutoTokenizer.from_pretrained(AIDO_RNA_MODEL_ID, trust_remote_code=True)

    df = pd.read_csv("input_data/df_data.csv")
    print(df.shape, flush=True)


    def cut_transcript(length_limit, seq, position, buffer=100):
        assert buffer >= 1

        if len(seq) <= length_limit:
            return seq, position

        if position < length_limit - buffer:
            return seq[:length_limit], position

        if len(seq) - position < buffer:
            start = len(seq) - length_limit
            return seq[-length_limit:], position - start

        start = position - (length_limit - buffer)
        return seq[start: position + buffer], position - start


    buffer_size  = 100
    length_limit = hp['max_len']


    df[["seq", "position"]] = df.apply(
        lambda r: cut_transcript(length_limit, r["seq"], r["position"], buffer_size),
        axis=1, result_type="expand",
    )

    sequences, positions, labels, genes = (
        df.seq.tolist(), df.position.tolist(),
        df.label.tolist(), df.gene_id.tolist(),
    )

    train_loader, val_loader, test_loader, pos_weight = make_dataset(
        sequences, positions, labels, genes, #tokenizer,
        batch_size=hp["batch_size"],
    )

    _, best_model_path = train_model(
        train_loader, val_loader, hp, pos_weight,
        naming_prefix="aido_rna_dora",
    )

    model = load_model(best_model_path, hp, DEVICE)
    yhat_prob, yhat, labels_t = predict_batch(model, test_loader, DEVICE)
    pprint(compute_binary_metrics(labels_t.numpy(), yhat.numpy()))


if __name__ == "__main__":
    main()