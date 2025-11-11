#!/usr/bin/env python3
"""
Train FusionWithEmotionDecoder on sequence-level multimodal features.

Pipeline:
    seq-level audio/text features (.pt)
        → TACFN-style CrossModalTransformer
        → vector-wise BetaGate fusion
        → EmotionDecoder with learnable emotion queries
        → per-emotion logits

Supports:
    - single-label (softmax + CrossEntropyLoss)
    - multi-label (sigmoid + BCEWithLogitsLoss)

This script is the "full model" version:
    fusion backbone + emotion-level decoder head.
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

from models.fusion_with_emotion_decoder import FusionWithEmotionDecoder


# ---------------------------------------------------------------
# Utils
# ---------------------------------------------------------------

def set_seed(seed: int = 1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()

    # Data paths
    ap.add_argument("--csv", type=str, default="data/iemocap_index_splits.csv",
                    help="CSV with utterance-level metadata, labels, and splits.")
    ap.add_argument("--audio_dir", type=str, default="features/seq_level/audio",
                    help="Directory with seq-level audio features (.pt).")
    ap.add_argument("--text_dir", type=str, default="features/seq_level/text",
                    help="Directory with seq-level text features (.pt).")

    # Column names
    ap.add_argument("--uid_col", type=str, default="utter_id")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--split_col", type=str, default="split")
    ap.add_argument("--train_split_name", type=str, default="train")
    ap.add_argument("--val_split_name", type=str, default="val")

    # Model hyperparameters
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--num_layers_fusion", type=int, default=2)
    ap.add_argument("--num_layers_decoder", type=int, default=2)
    ap.add_argument("--beta_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Emotion / label settings
    ap.add_argument("--loss_type", type=str, default="single_label",
                    choices=["single_label", "multi_label"],
                    help="single_label: CrossEntropy over emotions; "
                         "multi_label: BCEWithLogits over per-emotion scores.")

    # Training hyperparameters
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    # Logging / saving
    ap.add_argument("--out_dir", type=str, default="runs/fusion_seq_level_decoder")
    ap.add_argument("--seed", type=int, default=1234)

    return ap.parse_args()


# ---------------------------------------------------------------
# Dataset / Collate (reuse逻辑自 seq-level fusion)
# ---------------------------------------------------------------

class SeqLevelFusionDataset(Dataset):
    """
    Sequence-level multimodal dataset for:
        FusionWithEmotionDecoder.

    Each item:
        - audio seq-level feature: hidden[L_a, d], attention_mask[L_a]
        - text  seq-level feature: hidden[L_t, d], attention_mask[L_t]
        - label (single-label index or multi-hot vector)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        text_dir: Path,
        uid_col: str,
        label_col: str,
        label2id: Dict[str, int],
        loss_type: str = "single_label",
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.uid_col = uid_col
        self.label_col = label_col
        self.label2id = label2id
        self.loss_type = loss_type

        keep_indices: List[int] = []
        for i, row in self.df.iterrows():
            uid = str(row[self.uid_col])
            if (audio_dir / f"{uid}.pt").is_file() and (text_dir / f"{uid}.pt").is_file():
                keep_indices.append(i)

        if len(keep_indices) < len(self.df):
            print(f"[Dataset] Filtered {len(self.df) - len(keep_indices)} rows without features.")

        self.df = self.df.iloc[keep_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_seq_feat(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load sequence-level features:
            {
                "hidden": [L, d],
                "attention_mask": [L] (1 = valid, 0 = pad)
            }
        Return:
            hidden: [L, d] float
            mask:   [L] bool (True = PAD)
        """
        obj = torch.load(path, map_location="cpu")
        h = obj["hidden"].float()              # [L, d]
        m = obj["attention_mask"].long()       # [L]
        # Convert to bool key_padding_mask convention: True = PAD
        m = (m == 0)
        return h, m

    def _encode_label_single(self, label_str: str) -> int:
        return self.label2id[label_str]

    def _encode_label_multi(self, label_str: str) -> torch.Tensor:
        """
        For multi-label, assume CSV's label_col is a single primary label.
        We convert it into a one-hot over all emotions.
        If you have real multi-label annotations, adapt this part accordingly.
        """
        num_classes = len(self.label2id)
        y = torch.zeros(num_classes, dtype=torch.float32)
        if label_str in self.label2id:
            y[self.label2id[label_str]] = 1.0
        return y

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uid = str(row[self.uid_col])
        label_str = row[self.label_col]

        a_path = self.audio_dir / f"{uid}.pt"
        t_path = self.text_dir / f"{uid}.pt"

        h_a, m_a = self._load_seq_feat(a_path)
        h_t, m_t = self._load_seq_feat(t_path)

        if self.loss_type == "single_label":
            label = self._encode_label_single(label_str)
        else:
            label = self._encode_label_multi(label_str)

        return h_a, m_a, h_t, m_t, label


def collate_seq_batch(batch, loss_type: str):
    """
    Collate for variable-length seq-level multimodal features.

    Input: list of (h_a[L_a,d], m_a[L_a], h_t[L_t,d], m_t[L_t], label)

    Output:
        h_a: [B, L_a_max, d]
        mask_a: [B, L_a_max] bool (True = PAD)
        h_t: [B, L_t_max, d]
        mask_t: [B, L_t_max] bool (True = PAD)
        labels:
            - single_label: [B] long
            - multi_label:  [B, C] float
    """
    hs_a, ms_a, hs_t, ms_t, labels = zip(*batch)

    B = len(batch)
    d = hs_a[0].size(-1)

    L_a_max = max(x.size(0) for x in hs_a)
    L_t_max = max(x.size(0) for x in hs_t)

    padded_h_a = torch.zeros(B, L_a_max, d)
    padded_m_a = torch.ones(B, L_a_max, dtype=torch.bool)  # default PAD = True
    padded_h_t = torch.zeros(B, L_t_max, d)
    padded_m_t = torch.ones(B, L_t_max, dtype=torch.bool)

    for i in range(B):
        La = hs_a[i].size(0)
        Lt = hs_t[i].size(0)
        padded_h_a[i, :La] = hs_a[i]
        padded_m_a[i, :La] = ms_a[i]
        padded_h_t[i, :Lt] = hs_t[i]
        padded_m_t[i, :Lt] = ms_t[i]

    if loss_type == "single_label":
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = torch.stack(labels, dim=0)  # [B, C] float

    return padded_h_a, padded_m_a, padded_h_t, padded_m_t, labels


# ---------------------------------------------------------------
# Label mapping / Dataloaders
# ---------------------------------------------------------------

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
        train_df,
        audio_dir,
        text_dir,
        uid_col=args.uid_col,
        label_col=args.label_col,
        label2id=label2id,
        loss_type=args.loss_type,
    )
    val_ds = SeqLevelFusionDataset(
        val_df,
        audio_dir,
        text_dir,
        uid_col=args.uid_col,
        label_col=args.label_col,
        label2id=label2id,
        loss_type=args.loss_type,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_seq_batch(b, args.loss_type),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_seq_batch(b, args.loss_type),
    )

    return train_loader, val_loader, label2id


# ---------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------

def train_one_epoch(model, loader, criterion, device, loss_type: str):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_beta = []

    for h_a, m_a, h_t, m_t, labels in tqdm(loader, desc="Train", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        labels = labels.to(device)

        logits, beta, _ = model(h_a, h_t, m_a, m_t)

        if loss_type == "single_label":
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        else:
            loss = criterion(logits, labels)
            preds = (torch.sigmoid(logits) > 0.5).float()
            # simple multi-label accuracy: all labels match
            correct += ((preds == labels).all(dim=-1)).sum().item()
            total += labels.size(0)

        # (optional) encourage non-degenerate beta: small regularizer
        beta_reg = (beta * (1 - beta)).mean()
        loss = loss - 0.01 * beta_reg

        total_loss += loss.item() * labels.size(0)
        all_beta.extend(beta.detach().cpu().view(-1).tolist())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        model.optimizer.step()
        model.optimizer.zero_grad()

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    mean_beta = float(np.mean(all_beta)) if all_beta else 0.0
    return avg_loss, acc, mean_beta


@torch.no_grad()
def evaluate(model, loader, criterion, device, loss_type: str):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_beta = []

    for h_a, m_a, h_t, m_t, labels in tqdm(loader, desc="Val", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        labels = labels.to(device)

        logits, beta, _ = model(h_a, h_t, m_a, m_t)

        if loss_type == "single_label":
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        else:
            loss = criterion(logits, labels)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += ((preds == labels).all(dim=-1)).sum().item()
            total += labels.size(0)

        total_loss += loss.item() * labels.size(0)
        all_beta.extend(beta.detach().cpu().view(-1).tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    mean_beta = float(np.mean(all_beta)) if all_beta else 0.0
    return avg_loss, acc, mean_beta


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, label2id = get_dataloaders(args)
    num_emotions = len(label2id)

    print(f"[Info] num_emotions = {num_emotions}")

    # Build model
    model = FusionWithEmotionDecoder(
        d_model=args.d_model,
        num_emotions=num_emotions,
        n_heads=args.n_heads,
        num_layers_fusion=args.num_layers_fusion,
        num_layers_decoder=args.num_layers_decoder,
        beta_hidden=args.beta_hidden,
        dropout=args.dropout,
    ).to(device)

    # Attach optimizer to model (used inside train_one_epoch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model.optimizer = optimizer

    # Loss
    if args.loss_type == "single_label":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss, train_acc, train_beta = train_one_epoch(
            model, train_loader, criterion, device, args.loss_type
        )
        val_loss, val_acc, val_beta = evaluate(
            model, val_loader, criterion, device, args.loss_type
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Mean β: {train_beta:.3f}  "
            f"||  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Mean β: {val_beta:.3f}"
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
        ckpt_path = out_dir / "best_fusion_seq_decoder.pt"
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
        print("[Warning] No best model saved (empty val set or training issue).")


if __name__ == "__main__":
    main()
