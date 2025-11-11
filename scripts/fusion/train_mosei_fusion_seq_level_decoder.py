#!/usr/bin/env python3
"""
Train MOSEI multi-label emotion model:

    seq-level audio (COVAREP) + text (TimestampedWordVectors)
      -> projection to shared d_model
      -> TACFN-style CrossModal + BetaGate fusion
      -> Emotion-level decoder (per-emotion queries)
      -> 6-dim logits (happy, sad, anger, fear, disgust, surprise)

Loss: multi-label BCEWithLogitsLoss with soft targets from MOSEI annotations.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder


# ---------------- Utils ----------------

def set_seed(seed: int = 1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index_csv", type=str, default="data/mosei_index_splits.csv")
    ap.add_argument("--audio_dir", type=str, default="features/mosei/seq_level/audio")
    ap.add_argument("--text_dir", type=str, default="features/mosei/seq_level/text")

    ap.add_argument("--uid_col", type=str, default="uid")
    ap.add_argument("--video_col", type=str, default="video_id")
    ap.add_argument("--split_col", type=str, default="split")

    # emotion columns in CSV
    ap.add_argument(
        "--emo_cols",
        nargs="+",
        default=[
            "emo_happy",
            "emo_sad",
            "emo_anger",
            "emo_fear",
            "emo_disgust",
            "emo_surprise",
        ],
    )

    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--num_layers_fusion", type=int, default=2)
    ap.add_argument("--num_layers_decoder", type=int, default=2)
    ap.add_argument("--beta_hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)

    # train
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    # misc
    ap.add_argument("--out_dir", type=str, default="runs/mosei_fusion_decoder")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_train_samples", type=int, default=None)

    return ap.parse_args()


# ---------------- Dataset ----------------

class MoseiSeqDataset(Dataset):
    """
    Each item:
        h_a: [L_a, d_audio], m_a: [L_a] (all ones)
        h_t: [L_t, d_text],  m_t: [L_t] (all ones)
        y:   [C] float in [0,1]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        text_dir: Path,
        uid_col: str,
        emo_cols: List[str],
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.uid_col = uid_col
        self.emo_cols = emo_cols

        keep = []
        for i, row in self.df.iterrows():
            uid = str(row[uid_col])
            if (audio_dir / f"{uid}.pt").is_file() and (text_dir / f"{uid}.pt").is_file():
                keep.append(i)
        if len(keep) < len(self.df):
            print(f"[Dataset] Filtered {len(self.df) - len(keep)} rows without features.")
        self.df = self.df.iloc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_feat(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = torch.load(path, map_location="cpu")
        h = obj["hidden"].float()
        m = obj["attention_mask"].long()
        # convention: here 1 = valid, 0 = pad -> convert to bool True=PAD
        mask = (m == 0)
        # if no padding (all ones), mask will be all False
        return h, mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uid = str(row[self.uid_col])

        a_path = self.audio_dir / f"{uid}.pt"
        t_path = self.text_dir / f"{uid}.pt"

        h_a, m_a = self._load_feat(a_path)
        h_t, m_t = self._load_feat(t_path)

        # multi-label soft targets
        y = torch.tensor([float(row[c]) for c in self.emo_cols], dtype=torch.float32)

        return h_a, m_a, h_t, m_t, y


def collate_fn(batch):
    hs_a, ms_a, hs_t, ms_t, ys = zip(*batch)
    B = len(batch)

    d_a = hs_a[0].size(-1)
    d_t = hs_t[0].size(-1)

    L_a_max = max(x.size(0) for x in hs_a)
    L_t_max = max(x.size(0) for x in hs_t)

    pad_h_a = torch.zeros(B, L_a_max, d_a)
    pad_m_a = torch.ones(B, L_a_max, dtype=torch.bool)  # True = PAD
    pad_h_t = torch.zeros(B, L_t_max, d_t)
    pad_m_t = torch.ones(B, L_t_max, dtype=torch.bool)

    for i in range(B):
        La = hs_a[i].size(0)
        Lt = hs_t[i].size(0)
        pad_h_a[i, :La] = hs_a[i]
        pad_m_a[i, :La] = ms_a[i]
        pad_h_t[i, :Lt] = hs_t[i]
        pad_m_t[i, :Lt] = ms_t[i]

    y = torch.stack(ys, dim=0)  # [B, C]
    return pad_h_a, pad_m_a, pad_h_t, pad_m_t, y, d_a, d_t


# ---------------- Train / Eval ----------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    beta_list = []

    for h_a, m_a, h_t, m_t, y, d_a, d_t in tqdm(loader, desc="Train", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, beta, _ = model(h_a, h_t, m_a, m_t)

        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

        # simple multi-label accuracy: compare argmax of logits vs argmax of y
        pred_idx = logits.sigmoid().argmax(dim=-1)
        gold_idx = y.argmax(dim=-1)
        correct += (pred_idx == gold_idx).sum().item()

        if beta is not None:
            beta_list.extend(beta.detach().cpu().view(-1).tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    mean_beta = float(np.mean(beta_list)) if beta_list else 0.0
    return avg_loss, acc, mean_beta


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    beta_list = []

    for h_a, m_a, h_t, m_t, y, d_a, d_t in tqdm(loader, desc="Val", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        y = y.to(device)

        logits, beta, _ = model(h_a, h_t, m_a, m_t)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

        pred_idx = logits.sigmoid().argmax(dim=-1)
        gold_idx = y.argmax(dim=-1)
        correct += (pred_idx == gold_idx).sum().item()

        if beta is not None:
            beta_list.extend(beta.detach().cpu().view(-1).tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    mean_beta = float(np.mean(beta_list)) if beta_list else 0.0
    return avg_loss, acc, mean_beta


# ---------------- Main ----------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index_csv)

    train_df = df[df[args.split_col] == "train"]
    val_df = df[df[args.split_col] == "val"]

    if args.max_train_samples is not None and args.max_train_samples < len(train_df):
        train_df = train_df.sample(args.max_train_samples, random_state=42).reset_index(drop=True)
        print(f"[Info] Train subset: {len(train_df)} samples")

    audio_dir = Path(args.audio_dir)
    text_dir = Path(args.text_dir)

    train_ds = MoseiSeqDataset(train_df, audio_dir, text_dir, args.uid_col, args.emo_cols)
    val_ds = MoseiSeqDataset(val_df, audio_dir, text_dir, args.uid_col, args.emo_cols)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # 读取 meta 确定原始维度
    text_meta = json.loads((text_dir / "meta.json").read_text())
    audio_meta = json.loads((audio_dir / "meta.json").read_text())
    d_text = int(text_meta["hidden_dim"])
    d_audio = int(audio_meta["hidden_dim"])

    num_emotions = len(args.emo_cols)

    model = MoseiFusionWithEmotionDecoder(
        d_audio=d_audio,
        d_text=d_text,
        d_model=args.d_model,
        num_emotions=num_emotions,
        n_heads=args.n_heads,
        num_layers_fusion=args.num_layers_fusion,
        num_layers_decoder=args.num_layers_decoder,
        beta_hidden=args.beta_hidden,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()  # soft multi-label targets

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc, train_beta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_beta = evaluate(model, val_loader, criterion, device)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc(argmax): {train_acc:.4f} | Mean β: {train_beta:.3f}  "
            f"||  Val Loss: {val_loss:.4f} | Val Acc(argmax): {val_acc:.4f} | Mean β: {val_beta:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
                "args": vars(args),
                "emo_cols": args.emo_cols,
            }

    if best_state is not None:
        ckpt_path = out_dir / "best_mosei_fusion_decoder.pt"
        torch.save(best_state, ckpt_path)
        print(f"\n[Saved] Best model to {ckpt_path} (Val Acc(argmax) = {best_val_acc:.4f})")
    else:
        print("[Warning] No best model saved.")


if __name__ == "__main__":
    import json
    main()
