#!/usr/bin/env python3
"""
Train a simple MLP baseline with concatenated utter-level audio & text embeddings.
This serves as a naive multimodal fusion baseline to compare against β-Gating Fusion.
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


# ---------------------------------------------------------------
# Utility
# ---------------------------------------------------------------

def set_seed(seed=1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class ConcatDataset(Dataset):
    """Loads utter-level audio & text embeddings, concatenates along last dim."""

    def __init__(self, df, audio_dir, text_dir, uid_col, label_col, label2id):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)
        self.uid_col = uid_col
        self.label_col = label_col
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def _load_feat(self, path: Path):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            for k in ["hidden", "embedding", "feat"]:
                if k in obj:
                    return obj[k]
        raise ValueError(f"Cannot read feature: {path}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row[self.uid_col]
        label_str = row[self.label_col]
        a_feat = self._load_feat(self.audio_dir / f"{uid}.pt").float()
        t_feat = self._load_feat(self.text_dir / f"{uid}.pt").float()
        x = torch.cat([a_feat, t_feat], dim=-1)  # [d_a + d_t]
        label = self.label2id[label_str]
        return x, label


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

class ConcatClassifier(nn.Module):
    """Simple MLP classifier after concatenating multimodal embeddings."""

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------
# Training & Eval
# ---------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/iemocap_index_splits.csv")
    ap.add_argument("--audio_dir", default="features/utter_level/audio")
    ap.add_argument("--text_dir", default="features/utter_level/text")
    ap.add_argument("--uid_col", default="utter_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--train_split_name", default="train")
    ap.add_argument("--val_split_name", default="val")
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--out_dir", default="runs/concat_baseline")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    label2id = {l: i for i, l in enumerate(sorted(df[args.label_col].unique()))}
    print("[Labels]", label2id)

    train_df = df[df[args.split_col] == args.train_split_name]
    val_df = df[df[args.split_col] == args.val_split_name]

    train_ds = ConcatDataset(train_df, args.audio_dir, args.text_dir, args.uid_col, args.label_col, label2id)
    val_ds = ConcatDataset(val_df, args.audio_dir, args.text_dir, args.uid_col, args.label_col, label2id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Detect feature dimension automatically
    sample_x, _ = train_ds[0]
    in_dim = sample_x.numel()
    print(f"[Model] Input dim = {in_dim}")

    model = ConcatClassifier(in_dim=in_dim, hidden_dim=args.hidden_dim,
                             num_classes=len(label2id), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | Val: loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_concat.pt")

    print(f"[Saved] Best val acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
