import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import pprint
from datetime import datetime


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return [0] * position + [1] * (n - position)


class RNADataset(Dataset):
    def __init__(self, sequences: List[str], positions: List[int], labels: List[int]):
        assert len(sequences) == len(labels), "different length of sequences and labels"
        assert len(sequences) >= len(positions), "different length of sequences and postions"
        for seq, pos in zip(sequences, positions):
            assert len(seq) > pos, 'position out of sequence'
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
        hidden_dim: int = 64,
        # feature_dim: int = 1,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.encoder = nn.GRU(
            input_size=embed_dim + 1,  # 1 for the position indicator
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)  # binary -> 1 logit
        # self.classifier = nn.Linear(hidden_dim * 4, 1)  # [another version concat position output and all postion polling]

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, L]
        pos_features: torch.Tensor,   # [B, L]
        lengths: torch.Tensor,        # [B]
    ) -> torch.Tensor:
        seq_emb = self.embedding(input_ids)                    # [B, L, E]
        feat = pos_features.unsqueeze(-1).float()             # [B, L, 1]
        x = torch.cat([seq_emb, feat], dim=-1)                # [B, L, E+1]

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
        )                                                     # [B, L, 2H]

        marker_idx = pos_features.argmax(dim=1)               # [B]

        batch_size = input_ids.size(0)
        batch_idx = torch.arange(batch_size, device=input_ids.device)

        h = out[batch_idx, marker_idx, :]              # [B, 2H]

        # # [concat to global mean pooling]
        # mask = (input_ids != self.pad_idx).unsqueeze(-1)      # [B, L, 1]
        # out_masked = out * mask
        # summed = out_masked.sum(dim=1)                        # [B, 2H]
        # denom = mask.sum(dim=1).clamp(min=1)                  # [B, 1]
        # h_global = summed / denom                             # [B, 2H]
        # h = torch.cat([h, h_global], dim=-1)           # [B, 4H]
        
        logits = self.classifier(self.dropout(h))      # [B, 1]
        return logits.squeeze(-1)   


def compute_binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for input_ids, pos_features, lengths, labels in dataloader:
        input_ids = input_ids.to(device)
        pos_features = pos_features.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids, pos_features, lengths)   # [B]
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += compute_binary_accuracy(logits, labels) * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for input_ids, pos_features, lengths, labels in dataloader:
        input_ids = input_ids.to(device)
        pos_features = pos_features.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        logits = model(input_ids, pos_features, lengths)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += compute_binary_accuracy(logits, labels) * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples


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

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels)   # labels 通常已在 CPU

    all_probs = torch.cat(all_probs)   # [N]
    all_preds = torch.cat(all_preds)   # [N]
    all_labels = torch.cat(all_labels) # [N]

    return all_probs, all_preds, all_labels
    # return {
    #     "probs": all_probs,
    #     "preds": all_preds,
    #     "labels": all_labels,
    # }


def train_model(train_loader, val_loader, 
                embed_dim=16, hidden_dim=64, num_layers=1, dropout=0.2,
                EPOCH=1,
               ):
    model = RNASequenceBinaryClassifier(
        vocab_size=max(VOCAB.values()) + 1,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)
    print("total parameters:", sum(p.numel() for p in model.parameters()))
    print("total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = EPOCH
    best_val_acc = 0.0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )
    
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"biGRU_clf_{run_id}.pt")
            print("  -> saved best model")
    return model

def make_dataset(sequences, pos_features, labels, batch_size=2):
    dataset = RNADataset(sequences, pos_features, labels)


    train_size = int(0.7 * len(sequences))
    val_size   = int(0.15 * len(sequences))
    test_size  = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
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
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

def main():
    # hyper parameters
    h_bs = 2
    h_emb_dim = 16
    h_hidden_dim = 32 # 64
    h_num_layers = 1
    h_dropout = 0.2
    h_epoch = 5
    
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
    
    train_loader, val_loader, test_loader = make_dataset(sequences, positions, labels, h_bs)
    model = train_model(
        train_loader, val_loader,
        h_emb_dim, h_hidden_dim, h_num_layers,
        h_dropout, h_epoch)
    
    # real data 
    df_data = pd.read_csv("input_data/df_data.csv")
    sequences, positions, labels = df_data.seq.tolist(), df_data.position.tolist(), df_data.label.tolist()
    
    train_loader, val_loader, test_loader = make_dataset(sequences, positions, labels, h_bs)
    # model = train_model(
    #     train_loader, val_loader,
    #     h_emb_dim, h_hidden_dim, h_num_layers,
    #     h_dropout, h_epoch)
    
    yhat_prob, yhat, labels = predict_batch(model, test_loader, DEVICE)
    pprint.pprint(compute_binary_metrics(yhat.numpy(), labels.numpy()))


if __name__ == '__main__':
    main()
