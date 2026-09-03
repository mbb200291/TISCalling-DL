import random
from typing import List, Tuple
from collections import Counter
from datetime import datetime
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
from performer_pytorch.performer_pytorch import FixedPositionalEmbedding
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score, 
    average_precision_score
)
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from pprint import pprint

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', DEVICE)
# RNA vocabulary
VOCAB = {
    "A": 1,
    "T": 2,
    "C": 3,
    "G": 4,
    "N": 5,
}
PAD_IDX = 0
def encode_sequence(seq: str) -> List[int]:
    """
    convert seq to int list
    """
    # seq = seq.upper().replace("T", "U")  # convert to u
    return [VOCAB.get(ch, VOCAB["N"]) for ch in seq]

def encode_position(position: int, n: int) -> List[int]:
    assert position < n
    """
    convert position int to as onehot label
    """
    return [0] * position + [1] + [0] * (n - position - 1)

def get_sampler(labels_np):
    class_counts = np.bincount(labels_np)   # [count_0, count_1]
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels_np]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

class RNADataset(Dataset):
    def __init__(self, sequences: List[str], positions: List[int], labels: List[int]):
        assert len(sequences) == len(labels), "different length of sequences and labels"
        assert len(sequences) >= len(positions), "different length of sequences and postions"
        for seq, pos in zip(sequences, positions):
            assert len(seq) > pos, 'position out of sequence %d/%d' %(len(seq), pos)
        self.sequences = [encode_sequence(seq) for seq in sequences]
        self.positions = [encode_position(pos, len(seq)) for pos, seq in zip(positions, sequences)]
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_ids = self.sequences[idx]
        positions = self.positions[idx]
        label = self.labels[idx]

        return (
            torch.tensor(seq_ids, dtype=torch.long),      # [L]
            torch.tensor(positions, dtype=torch.float),        # [L]
            torch.tensor(label, dtype=torch.float),        # [L]
        )

def collate_fn(batch):
    """
    return:
      padded_seqs:   [B, L]
      padded_position_encode:  [B, L]
      lengths:       [B]
      labels:        [B]
    """
    seqs, feats, labels = zip(*batch)

    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    max_len = max(lengths).item()
    batch_size = len(seqs)

    padded_seqs = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long)
    padded_feats = torch.zeros((batch_size, max_len), dtype=torch.float)

    for i, (seq, feat) in enumerate(zip(seqs, feats)):
        seq_len = len(seq)
        padded_seqs[i, :seq_len] = seq
        padded_feats[i, :seq_len] = feat

    labels = torch.stack(labels)
    return padded_seqs, padded_feats, lengths, labels


class RNASequenceBinaryClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.2,
        max_len: int = 4096,
        pad_idx: int = PAD_IDX,
        pooling_mode: str = "attention",  # "marker" | "mean" | "attention"
    ):
        super().__init__()

        valid_pooling_modes = {"marker", "mean", "attention"}
        if pooling_mode not in valid_pooling_modes:
            raise ValueError(
                f"pooling_mode must be one of {valid_pooling_modes}, got {pooling_mode}"
            )

        self.pad_idx = pad_idx
        self.d_model = d_model
        self.max_len = max_len
        self.pooling_mode = pooling_mode

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # input = token embedding + 1 scalar feature
        self.input_proj = nn.Linear(embed_dim + 1, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.position_embedding = nn.Embedding(max_len * 2, d_model)   # embed by simply postion number
        self.frame_embedding = nn.Embedding(math.ceil(2 * max_len / 3) + 1, d_model)  # 3 reading frames 
        # self.pos_emb = FixedPositionalEmbedding(d_model, max_len)  # Sinusoidal

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        if self.pooling_mode == "attention":
            self.marker_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.dropout = nn.Dropout(dropout)

        if self.pooling_mode == "marker":
            classifier_in_dim = d_model
        else:  # mean pooling
            classifier_in_dim = d_model * 2

        self.classifier = nn.Linear(classifier_in_dim, 1)

    def _masked_mean_pool(
        self,
        x: torch.Tensor,              # [B, L, D]
        padding_mask: torch.Tensor,   # [B, L], True = padding
    ) -> torch.Tensor:
        valid_mask = (~padding_mask).unsqueeze(-1).float()    # [B, L, 1]
        x_masked = x * valid_mask                             # [B, L, D]
        denom = valid_mask.sum(dim=1).clamp(min=1.0)         # [B, 1]
        mean_pooled = x_masked.sum(dim=1) / denom            # [B, D]
        return mean_pooled

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, L]
        pos_features: torch.Tensor,   # [B, L]
        lengths: torch.Tensor,        # [B], kept for interface compatibility
    ) -> torch.Tensor:
        del lengths  # no needed

        B, L = input_ids.shape

        if L > self.max_len:
            raise ValueError(
                f"Sequence length {L} exceeds max_len={self.max_len}"
            )

        # [B, L, E]
        seq_emb = self.embedding(input_ids)

        # [B, L, 1]
        ind = pos_features.unsqueeze(-1).float()

        # [B, L, E+1] -> [B, L, D]
        x = torch.cat([seq_emb, ind], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)

        # positional embedding
        marker_pos = pos_features.argmax(dim=1)        # (B,)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
        relative_pos_id = pos_ids - marker_pos.unsqueeze(1) + self.max_len - 1  # (B, L)  # add max len to ensure index all postion and map zero to fix number
        x += self.position_embedding(relative_pos_id)

        # ## Sinusoidal
        # x += self.pos_emb(x)

        # frame embedding
        # frame_offset = (marker_pos % 3).unsqueeze(1)   # (B, 1)
        # pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
        # frame_ids = (pos_ids + 3 - frame_offset) // 3      # (B, L)
        relative_frame_ids = relative_pos_id // 3
        x += self.frame_embedding(relative_frame_ids)

        # label the padding token
        padding_mask = input_ids.eq(self.pad_idx)  # [B, L]

        # [B, L, D]
        out = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # candidate start site indicators
        marker_idx = pos_features.argmax(dim=1)  # [B]
        batch_idx = torch.arange(B, device=input_ids.device)

        # [B, D]
        marker_h = out[batch_idx, marker_idx, :]

        if self.pooling_mode == "marker":
            h = marker_h

        elif self.pooling_mode == "mean":
            global_h = self._masked_mean_pool(out, padding_mask)  # [B, D]
            h = torch.cat([marker_h, global_h], dim=-1)           # [B, 2D]

        elif self.pooling_mode == "attention":
            query = marker_h.unsqueeze(1)  # [B, 1, D]

            global_h, _ = self.marker_attention(
                query=query,
                key=out,
                value=out,
                key_padding_mask=padding_mask,
                need_weights=False,
            )                               # [B, 1, D]

            global_h = global_h.squeeze(1)  # [B, D]
            h = torch.cat([marker_h, global_h], dim=-1)  # [B, 2D]

        else:
            raise RuntimeError(f"Unexpected pooling_mode: {self.pooling_mode}")

        logits = self.classifier(self.dropout(h))  # [B, 1]
        return logits.squeeze(-1)                  # [B]


def train_one_epoch(model, dataloader, optimizer, criterion, device, threshold: float = 0.5):
    model.train()

    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0

    for input_ids, pos_features, lengths, labels in dataloader:
        input_ids = input_ids.to(device)
        pos_features = pos_features.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        logits = model(input_ids, pos_features, lengths)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold: float = 0.5):
    
    model.eval()

    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0

    all_labels = []
    all_probs = []

    for input_ids, pos_features, lengths, labels in dataloader:
        input_ids = input_ids.to(device)
        pos_features = pos_features.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device).float()

        logits = model(input_ids, pos_features, lengths)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # MCC
    mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if mcc_denom == 0:
        mcc = 0.0
    else:
        mcc = (tp * tn - fp * fn) / np.sqrt(mcc_denom)

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # AUC / AUPR is not defined when label only single class
    if len(np.unique(all_labels)) < 2:
        auc = float("nan")
        aupr = float("nan")
    else:
        auc = roc_auc_score(all_labels, all_probs)
        aupr = average_precision_score(all_labels, all_probs)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "auc": auc,
        "aupr": aupr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_model(
    train_loader,
    val_loader,
    hp_queryer,
    # embed_dim=16,
    # d_model=64,
    # num_layers=1,
    # dropout=0.2,
    # num_heads=4,
    # ff_dim=128,
    # max_epoch=1,
    pos_weight,
    naming_prefix='model',
    patience=10,
    # pooling_mode='marker',
):
    print('hyperparameters:', hp_queryer())
    model = RNASequenceBinaryClassifier(
        vocab_size=max(VOCAB.values()) + 1,
        **hp_queryer(
            "embed_dim", 'num_layers', 'dropout', 'd_model', 'num_heads',
            'ff_dim', 'max_len', 'pooling_mode'),
        # embed_dim=embed_dim,
        # num_layers=num_layers,
        # dropout=dropout,
        # d_model=d_model,
        # num_heads=num_heads,
        # ff_dim=ff_dim,
        # max_len=4096,
        pad_idx=PAD_IDX,
        # pooling_mode=pooling_mode,  # "marker" | "mean" | "attention"
    ).to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    print("total parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    print("total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight * hp_queryer('weight_fold')['weight_fold'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp_queryer('lr')['lr'])

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,       # lr shrink factor
        patience=10,       # number of epoch to tolenent no loss improving
        min_lr=1e-6,
    )

    best_val_f1 = -1
    # best_val_loss = float('inf')
    epochs_no_improve = 0   # early stopping counter
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model/{naming_prefix}_{run_id}.pt"

    for epoch in range(1, hp_queryer('max_epoch')['max_epoch'] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:02d} | lr={current_lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_accuracy={train_metrics['accuracy']:.4f} "
            f"train_precision={train_metrics['precision']:.4f} "
            f"train_recall={train_metrics['recall']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} | "
            # f"train_aupr={train_metrics['aupr']:.4f}",
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}",
            f"val_aupr={val_metrics['aupr']:.4f}",
            flush=True
        )

        scheduler.step(val_metrics['f1'])

        # if val_metrics["f1"] > best_val_loss:
        if val_metrics["f1"] > best_val_f1:
            # best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            print(f"  -> saved best model to {save_path}", flush=True)
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  -> early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
                break

    print(f"Best validation F1: {best_val_f1:.4f}")
    return model, save_path


@torch.no_grad()
def predict_prob(model, seq: str, feat: List[int], device: torch.device) -> float:
    model.eval()

    seq_ids = encode_sequence(seq)
    assert len(seq_ids) == len(feat), "positions != sequence"

    input_ids = torch.tensor([seq_ids], dtype=torch.long).to(device)      # [1, L]
    pos_features = torch.tensor([feat], dtype=torch.float).to(device)     # [1, L]
    lengths = torch.tensor([len(seq_ids)], dtype=torch.long).to(device)   # [1]

    logits = model(input_ids, pos_features, lengths)                      # [1]
    prob = torch.sigmoid(logits).item()
    return prob


@torch.no_grad()
def predict_batch(model, dataloader, device: torch.device, threshold: float = 0.5):
    model.eval()

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    for input_ids, pos_features, lengths, labels in dataloader:
        input_ids = input_ids.to(device)
        pos_features = pos_features.to(device)
        lengths = lengths.to(device)

        logits = model(input_ids, pos_features, lengths)   # [B]
        probs = torch.sigmoid(logits)                      # [B]
        preds = (probs >= threshold).float()               # [B]

        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels)

    all_probs = torch.cat(all_probs)   # [N]
    all_preds = torch.cat(all_preds)   # [N]
    all_labels = torch.cat(all_labels) # [N]

    return all_logits, all_probs, all_preds, all_labels

def make_dataset(sequences, pos_features, labels, groups: List[str], batch_size=2):
    assert len(sequences) == len(pos_features) == len(labels) == len(groups), \
        "sequences, pos_features, labels, groups length not the same"

    print('labels:', dict(Counter(labels)))

    dataset = RNADataset(sequences, pos_features, labels)

    indices = np.arange(len(dataset))
    groups = np.array(groups)
    labels = np.array(labels)

    # split train-val/test：15%
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEED,
    )
    train_val_idx, test_idx = next(
        gss_test.split(indices, labels, groups=groups)
    )

    # split val from train-val (85%) -> adjust ratio as based on total
    # 0.15 / 0.85 ≈ 0.17647
    val_ratio_in_trainval = 0.15 / 0.85

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio_in_trainval,
        random_state=SEED,
    )
    train_rel_idx, val_rel_idx = next(
        gss_val.split(
            indices[train_val_idx],
            labels[train_val_idx],
            groups=groups[train_val_idx],
        )
    )

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    # sanity check group not overlap
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    assert train_groups.isdisjoint(val_groups), "train / val groups overlaped"
    assert train_groups.isdisjoint(test_groups), "train / test groups overlaped"
    assert val_groups.isdisjoint(test_groups), "val / test groups overlaped"

    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())
    test_dataset = Subset(dataset, test_idx.tolist())

    # Get labels for the training subset to create the sampler
    train_labels_for_sampler = labels[train_idx]
    # sampler = get_sampler(train_labels_for_sampler)  # switch to loss weight
    num_pos = (train_labels_for_sampler == 1).sum()
    num_neg = (train_labels_for_sampler == 0).sum()
    pos_weight = torch.tensor(
        [num_neg / max(num_pos, 1)],
        dtype=torch.float32,
        device=DEVICE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # sampler=sampler,
        # shuffle=False,  # disable when sampler provided
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

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

def get_test_dataset(batch_size):
    # test run
    sequences = [
        "ATGCTTACG",
        "GGGAAATTTCC",
        "ACGTACGT",
        "TTTTGGCA",
        "CGATATACGGA",
        "ACGTACGT",
        "TTTTGGCA",
        "CGATATACGGA",
        "TTTTGGCA",
        "CGATATACGGA",
    ]
    
    positions = [
        2, 5, 6, 3, 9, 2, 5, 6, 3, 9
    ]
    
    labels = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]

    genes = ['a', 'a', 'b', 'c', 'd', 'd', 'e', 'f', 'f', 'g']

    train_loader, val_loader, test_loader, pos_weight = make_dataset(
        sequences, positions, labels, genes, batch_size)
    return train_loader, val_loader, test_loader, pos_weight


def load_model(path: str, hp_queryer, device: torch.device = DEVICE) -> RNASequenceBinaryClassifier:
    model = RNASequenceBinaryClassifier(
        vocab_size=max(VOCAB.values()) + 1,
        **hp_queryer("embed_dim", 'num_layers', 'dropout', 'd_model', 'num_heads', 'ff_dim', 'max_len', 'pooling_mode'),
        # embed_dim=16,
        # d_model=32,
        # num_layers=1,
        # num_heads=2,
        # ff_dim=64,
        # dropout=0.1,
        # max_len=4096,
        pad_idx=PAD_IDX,
        # pooling_mode="attention",
        # pooling_mode="marker",
    ).to(device)

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:  # seperate to multiple gpu
        model = torch.nn.DataParallel(model)

    model.eval()
    return model

def hyperparameters_queryer(*args):
    return {k: v for k, v in {
        'embed_dim': 32,
        'd_model': 64, # 32 ~ 64
        'num_layers': 3, # 1 ~ 2
        'num_heads': 8, # 2 ~ 4 
        'ff_dim': 128, #64 ~ 128 (convension = d_model * 2)
        'batch_size': 32, #16, # 32 ~ 64
        'max_epoch': 50,
        'dropout': 0.3, #0.2, 
        'weight_fold': 0.1 , #0.5, #1,
        'max_len': 2048, #4096,
        'pooling_mode': 'attention',
        'lr': 1e-3,
    }.items() if k in args or len(args)==0}


def main():
    print('run at:', datetime.now().strftime("%Y%m%d_%H%M%S"))
    train_loader, val_loader, test_loader, pos_weight = get_test_dataset(10)

    model, best_model_path = train_model(
            train_loader, val_loader,
            hyperparameters_queryer,
            pos_weight, 
            naming_prefix='test-run'
    )

    # real data
    df_data = pd.read_csv("input_data/df_data.csv")
    print(df_data.shape, flush=True)
    
    # # exclude size more than length limit
    # ## exclude strategy 1: remove position label only
    # # df_data = df_data.loc[df_data.position < 4096 - 50]
    # ## exclude strategy 2: revmove whole transcript when candidate start site exceed 4k
    # gene_to_exclude = set(df_data.gene_id[(df_data.position > 4096 - 50) & (df_data.label == 1)].tolist())
    # df_data = df_data.loc[~df_data.gene_id.isin(gene_to_exclude)]
    # df_data = df_data.loc[df_data.position < 4096 - 50]
    # df_data['seq'] = df_data.seq.str[:4096]  
    # print(df_data.shape, flush=True)
    ## exclude strategy 3: centralize by candidate atg
    def cut_transcript(length_limit: int, seq: str, position: int, buffer=100):
        assert buffer >= 1

        if len(seq) <= length_limit:
            return seq, position

        if position < length_limit - buffer:  # keep 5'end
            return seq[:length_limit], position

        if len(seq) - position < buffer:  # 3' end buffer is not enough -> cut from 3'end to centralize target
            start = len(seq) - length_limit
            return seq[-length_limit:], position - start

        left_truncate_size = length_limit - buffer
        start = position - left_truncate_size
        end = position + buffer
        return seq[start:end], position - start

    length_limit = hyperparameters_queryer("max_len")["max_len"]
    buffer_size = 100

    df_data[["seq", "position"]] = df_data.apply(
        lambda row: cut_transcript(length_limit, row["seq"], row["position"], buffer_size),
        axis=1,
        result_type="expand"
    )

    sequences, positions, labels, genes = (
        df_data.seq.tolist(), df_data.position.tolist(),
        df_data.label.tolist(), df_data.gene_id.tolist())

    # # debug -> label as 1 when codon is ATG
    # labels = []
    # for seq, pos in zip(sequences, positions):
    #     if seq[pos:pos+3] == 'ATG':
    #         labels.append(1)
    #     else:
    #         labels.append(0)

    # # debug -> truncate seq before atg -> examine relative postion
    # for i in range(len(sequences)):
    #     sequences[i] = sequences[i][positions[i]:]
    #     positions[i] = 0

    # # debug -> select window range arround target
    # for i in range(len(sequences)):
    #     sequences[i] = sequences[i][max(positions[i]-100, 0):positions[i]+101]
    #     positions[i] = min(100, positions[i])

    train_loader, val_loader, test_loader, pos_weight = make_dataset(sequences, positions, labels, genes, **hyperparameters_queryer('batch_size'))

    _, best_model_path = train_model(
        train_loader, val_loader,
        hyperparameters_queryer,
        pos_weight,
        naming_prefix='transformer_clf',
    )
    
    model = load_model(best_model_path, hyperparameters_queryer, DEVICE)

    # evaluate test set
    _, yhat_prob, yhat, labels = predict_batch(model, test_loader, DEVICE)
    pprint(compute_binary_metrics(labels.numpy(), yhat.numpy(), yhat_prob.numpy()))


if __name__ == '__main__':
    main()
    print()
