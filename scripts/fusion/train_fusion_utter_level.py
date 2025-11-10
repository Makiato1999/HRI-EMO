#!/usr/bin/env python3
"""
Train FusionClassifier on utterance-level features with Beta-Gated Cross-Modal Fusion.

This script:
    - Loads pre-extracted utterance-level audio & text features
    - Aligns them by utter_id using a CSV index file
    - Builds a simple dataset (audio_feat, text_feat, label)
    - Trains FusionClassifier:
          CrossModalTransformer + BetaGate + Classifier
    - Logs train/val performance and saves the best checkpoint

Designed as a "quick, clean" experiment to validate:
    - Cross-modal interaction is useful
    - The learned beta gate improves over naive concatenation
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.fusion_classifier import FusionClassifier

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def set_seed(seed: int = 1234):
    """Fix random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data / feature paths
    parser.add_argument(
        "--csv",
        type=str,
        default="data/iemocap_index_splits.csv",
        help="CSV file with utterance metadata, labels, and splits.",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="features/utter_level/audio",
        help="Directory of utterance-level audio features (.pt files).",
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        default="features/utter_level/text",
        help="Directory of utterance-level text features (.pt files).",
    )

    # Column names (adapt to your CSV if needed)
    parser.add_argument(
        "--uid_col",
        type=str,
        default="utter_id",
        help="Column name for unique utterance ID.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Column name for emotion label.",
    )
    parser.add_argument(
        "--split_col",
        type=str,
        default="split",
        help="Column name indicating data split (e.g., train/val/test).",
    )
    parser.add_argument(
        "--train_split_name",
        type=str,
        default="train",
        help="Value in split_col used for the training set.",
    )
    parser.add_argument(
        "--val_split_name",
        type=str,
        default="val",
        help="Value in split_col used for the validation set.",
    )

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--beta_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # Logging / saving
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/fusion_utter_level",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class UtterLevelFusionDataset(Dataset):
    """
    Dataset for utterance-level multimodal fusion.

    For each row in the CSV:
        - Loads audio feature from {audio_dir}/{utter_id}.pt
        - Loads text feature  from {text_dir}/{utter_id}.pt
        - Returns (h_audio, h_text, label_id)

    Assumes each .pt file contains either:
        - a single tensor, or
        - a dict with one of the keys: ["feat", "embedding", "hidden"]
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

        # Optional: filter rows with missing feature files
        valid_indices: List[int] = []
        for idx, row in self.df.iterrows():
            uid = str(row[self.uid_col])
            a_path = self.audio_dir / f"{uid}.pt"
            t_path = self.text_dir / f"{uid}.pt"
            if a_path.is_file() and t_path.is_file():
                valid_indices.append(idx)

        if len(valid_indices) < len(self.df):
            print(
                f"[Dataset] Filtered {len(self.df) - len(valid_indices)} rows without both features."
            )

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_feat(self, path: Path) -> torch.Tensor:
        """Load feature tensor from .pt file in a robust way."""
        obj = torch.load(path, map_location="cpu")

        # Case 1: directly a tensor
        if isinstance(obj, torch.Tensor):
            return obj

        # Case 2: dict-like, try common keys
        if isinstance(obj, dict):
            for key in ["feat", "feats", "embedding", "hidden", "repr"]:
                if key in obj:
                    t = obj[key]
                    if isinstance(t, torch.Tensor):
                        return t
        raise ValueError(f"Unsupported feature format in {path}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uid = str(row[self.uid_col])
        label_str = row[self.label_col]

        a_path = self.audio_dir / f"{uid}.pt"
        t_path = self.text_dir / f"{uid}.pt"

        h_a = self._load_feat(a_path).float()  # [d] or [L,d]; model handles both
        h_t = self._load_feat(t_path).float()

        label_id = self.label2id[label_str]

        return h_a, h_t, label_id


# -----------------------------------------------------------------------------
# Training / Evaluation loops
# -----------------------------------------------------------------------------


def create_label_mapping(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    """
    Build a label → id mapping from the labels present in the dataframe.
    Keeps stable ordering by sorting label names.
    """
    labels = sorted(df[label_col].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    print("[Labels]", label2id)
    return label2id


def get_dataloaders(
    args,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    df = pd.read_csv(args.csv)

    # Build label mapping using all rows (or restrict to train only if preferred)
    label2id = create_label_mapping(df, args.label_col)

    audio_dir = Path(args.audio_dir)
    text_dir = Path(args.text_dir)

    # Train subset
    train_df = df[df[args.split_col] == args.train_split_name]
    val_df = df[df[args.split_col] == args.val_split_name]

    train_ds = UtterLevelFusionDataset(
        train_df,
        audio_dir,
        text_dir,
        uid_col=args.uid_col,
        label_col=args.label_col,
        label2id=label2id,
    )
    val_ds = UtterLevelFusionDataset(
        val_df,
        audio_dir,
        text_dir,
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
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, label2id


def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for h_a, h_t, labels in tqdm(loader, desc="Train", leave=False):
        h_a = h_a.to(device)
        h_t = h_t.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, beta, _ = model(h_a, h_t)  # masks None for utter-level
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


def evaluate(model, loader, criterion, device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    beta_values: List[float] = []

    with torch.no_grad():
        for h_a, h_t, labels in tqdm(loader, desc="Val", leave=False):
            h_a = h_a.to(device)
            h_t = h_t.to(device)
            labels = labels.to(device)

            logits, beta, _ = model(h_a, h_t)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Collect β for interpretability monitoring
            beta_values.extend(beta.squeeze(-1).cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    mean_beta = float(np.mean(beta_values)) if beta_values else 0.0
    return avg_loss, acc, mean_beta


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, label2id = get_dataloaders(args)
    num_classes = len(label2id)
    if num_classes != args.num_classes:
        print(
            f"[Warning] num_classes ({args.num_classes}) != labels in CSV ({num_classes}), "
            f"using {num_classes}."
        )
        args.num_classes = num_classes

    # Model
    model = FusionClassifier(
        d_model=args.d_model,
        num_classes=args.num_classes,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        beta_hidden=args.beta_hidden,
        dropout=args.dropout,
    ).to(device)

    # Loss & Optimizer
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

        # Track best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "label2id": label2id,
                "val_acc": best_val_acc,
                "epoch": epoch,
            }

    # Save best checkpoint
    if best_state is not None:
        ckpt_path = out_dir / "best_fusion_utt.pt"
        torch.save(best_state, ckpt_path)
        print(f"\n[Saved] Best model to {ckpt_path} (Val Acc = {best_val_acc:.4f})")

        # Also dump label mapping & config as JSON for convenience
        meta = {
            "label2id": label2id,
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    else:
        print("[Warning] No model was saved (no validation data or training failed).")


if __name__ == "__main__":
    main()
