#!/usr/bin/env python3
"""
Train FusionClassifier on sequence-level features with Beta-Gated Cross-Modal Fusion.

This script:
    - Loads sequence-level audio & text features:
          audio: hidden[T_a, d], attention_mask[T_a]
          text:  hidden[L_t, d], attention_mask[L_t]
    - Pads sequences in each batch and constructs key padding masks
    - Feeds them into:
          CrossModalTransformer + BetaGate + Classifier
    - Evaluates accuracy and monitors the mean beta value

This is the key step to verify:
    - Our architecture works properly on [B, L, d] inputs with masks
    - Beta-gating behaves reasonably at sequence-level
Before adding the Emotion-Level Decoder.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.fusion_classifier import FusionClassifier


# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------

def set_seed(seed: int = 1234):
    """Fix random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()

    # Data paths
    ap.add_argument("--csv", type=str, default="data/iemocap_index_splits.csv",
                    help="CSV file with utterance metadata, labels, and splits.")
    ap.add_argument("--audio_dir", type=str, default="features/seq_level/audio",
                    help="Directory with seq-level audio features (.pt).")
    ap.add_argument("--text_dir", type=str, default="features/seq_level/text",
                    help="Directory with seq-level text features (.pt).")

    # CSV column names
    ap.add_argument("--uid_col", type=str, default="utter_id",
                    help="Column name for unique utterance ID.")
    ap.add_argument("--label_col", type=str, default="label",
                    help="Column name for emotion label.")
    ap.add_argument("--split_col", type=str, default="split",
                    help="Column name indicating split (e.g., train/val/test).")
    ap.add_argument("--train_split_name", type=str, default="train")
    ap.add_argument("--val_split_name", type=str, default="val")

    # Model hyperparameters
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--num_classes", type=int, default=6,
                    help="Number of emotion classes. Will be overridden by CSV if mismatched.")
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--beta_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)

    # Training hyperparameters
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Smaller batch size recommended for seq-level due to memory.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    # Logging / saving
    ap.add_argument("--out_dir", type=str, default="runs/fusion_seq_level_beta")
    ap.add_argument("--seed", type=int, default=1234)

    return ap.parse_args()


# -------------------------------------------------------------------------
# Dataset for sequence-level features
# -------------------------------------------------------------------------

class SeqLevelFusionDataset(Dataset):
    """
    Sequence-level multimodal dataset.

    Each item:
        - Loads audio seq features from {audio_dir}/{uid}.pt
        - Loads text  seq features from {text_dir}/{uid}.pt

    Expected .pt format for both modalities:
        {
            "hidden": [L, d],
            "attention_mask": [L]
        }

    Returns (un-padded):
        h_a: [L_a, d]
        m_a: [L_a]
        h_t: [L_t, d]
        m_t: [L_t]
        label_id: int
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        text_dir: Path,
        uid_col: str,
        label_col: str,
        label2id: Dict[str, int],
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.uid_col = uid_col
        self.label_col = label_col
        self.label2id = label2id

        # Keep only rows where both features exist
        keep_indices: List[int] = []
        for i, row in self.df.iterrows():
            uid = str(row[self.uid_col])
            if (audio_dir / f"{uid}.pt").is_file() and (text_dir / f"{uid}.pt").is_file():
                keep_indices.append(i)

        if len(keep_indices) < len(self.df):
            print(f"[Dataset] Filtered {len(self.df) - len(keep_indices)} missing feature rows.")

        self.df = self.df.iloc[keep_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_seq_feat(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load sequence-level features:
            returns (hidden[L,d], mask[L])
        """
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict with 'hidden' and 'attention_mask' in {path}")

        h = obj.get("hidden", None)
        m = obj.get("attention_mask", None)

        if h is None or m is None:
            raise ValueError(f"Missing keys in {path}, found keys: {list(obj.keys())}")

        h = h.float()                        # [L, d]
        m = m.to(torch.bool)                 # [L], 1/0 -> bool
        # We expect mask == 1 for valid, 0 for pad from extractor.
        # Our model expects True = PAD, so invert:
        m = ~m                               # now: True = pad, False = valid

        return h, m

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uid = str(row[self.uid_col])
        label_str = row[self.label_col]

        a_path = self.audio_dir / f"{uid}.pt"
        t_path = self.text_dir / f"{uid}.pt"

        h_a, m_a = self._load_seq_feat(a_path)
        h_t, m_t = self._load_seq_feat(t_path)

        label_id = self.label2id[label_str]

        return h_a, m_a, h_t, m_t, label_id


# -------------------------------------------------------------------------
# Collate function for variable-length sequences
# -------------------------------------------------------------------------

def collate_seq_batch(batch):
    """
    Collate function to:
        - pad audio/text sequences in batch to max length
        - stack into tensors
        - keep boolean key_padding_mask (True = pad)

    Input (list of):
        (h_a[L_a,d], m_a[L_a], h_t[L_t,d], m_t[L_t], label)

    Output:
        h_a: [B, L_a_max, d]
        mask_a: [B, L_a_max]  (bool, True=pad)
        h_t: [B, L_t_max, d]
        mask_t: [B, L_t_max]
        labels: [B]
    """
    hs_a, ms_a, hs_t, ms_t, labels = zip(*batch)

    B = len(batch)
    d = hs_a[0].size(-1)

    L_a_max = max(x.size(0) for x in hs_a)
    L_t_max = max(x.size(0) for x in hs_t)

    padded_h_a = torch.zeros(B, L_a_max, d)
    padded_m_a = torch.ones(B, L_a_max, dtype=torch.bool)  # default pad=True
    padded_h_t = torch.zeros(B, L_t_max, d)
    padded_m_t = torch.ones(B, L_t_max, dtype=torch.bool)

    for i in range(B):
        La = hs_a[i].size(0)
        Lt = hs_t[i].size(0)

        padded_h_a[i, :La] = hs_a[i]
        padded_m_a[i, :La] = ms_a[i]

        padded_h_t[i, :Lt] = hs_t[i]
        padded_m_t[i, :Lt] = ms_t[i]

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_h_a, padded_m_a, padded_h_t, padded_m_t, labels


# -------------------------------------------------------------------------
# Dataloaders & label mapping
# -------------------------------------------------------------------------

def create_label_mapping(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    labels = sorted(df[label_col].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    print("[Labels]", label2id)
    return label2id


def get_dataloaders(args):
    df = pd.read_csv(args.csv)

    label2id = create_label_mapping(df, args.label_col)

    audio_dir = Path(args.audio_dir)
    text_dir = Path(args.text_dir)

    train_df = df[df[args.split_col] == args.train_split_name]
    val_df = df[df[args.split_col] == args.val_split_name]

    train_ds = SeqLevelFusionDataset(
        train_df, audio_dir, text_dir,
        uid_col=args.uid_col,
        label_col=args.label_col,
        label2id=label2id,
    )
    val_ds = SeqLevelFusionDataset(
        val_df, audio_dir, text_dir,
        uid_col=args.uid_col,
        label_col=args.label_col,
        label2id=label2id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_seq_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_seq_batch,
    )

    return train_loader, val_loader, label2id


# -------------------------------------------------------------------------
# Training / Evaluation
# -------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for h_a, m_a, h_t, m_t, labels in tqdm(loader, desc="Train", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, beta, _ = model(h_a, h_t, mask_a=m_a, mask_t=m_t)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    beta_values: List[float] = []

    for h_a, m_a, h_t, m_t, labels in tqdm(loader, desc="Val", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        labels = labels.to(device)

        logits, beta, _ = model(h_a, h_t, mask_a=m_a, mask_t=m_t)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        beta_values.extend(beta.squeeze(-1).detach().cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    mean_beta = float(np.mean(beta_values)) if beta_values else 0.0
    return avg_loss, acc, mean_beta


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataloaders
    train_loader, val_loader, label2id = get_dataloaders(args)

    # Adjust num_classes based on labels actually present
    num_classes = len(label2id)
    if num_classes != args.num_classes:
        print(
            f"[Warning] num_classes ({args.num_classes}) != labels in CSV ({num_classes}), using {num_classes}."
        )
        args.num_classes = num_classes

    # Initialize model
    model = FusionClassifier(
        d_model=args.d_model,
        num_classes=args.num_classes,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        beta_hidden=args.beta_hidden,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, mean_beta = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}  "
            f"||  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Mean β: {mean_beta:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
                "args": vars(args),
                "label2id": label2id,
            }

    if best_state is not None:
        ckpt_path = out_dir / "best_fusion_seq.pt"
        torch.save(best_state, ckpt_path)
        print(f"\n[Saved] Best model to {ckpt_path} (Val Acc = {best_val_acc:.4f})")
        meta = {
            "best_val_acc": best_val_acc,
            "label2id": label2id,
            "args": vars(args),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    else:
        print("[Warning] No best model saved (empty val set?).")


if __name__ == "__main__":
    main()
