#!/usr/bin/env python3
"""
Train MOSEI multi-label emotion model (Simplified Stable Version)

    seq-level audio (COVAREP) + text (TimestampedWordVectors)
      -> projection to shared d_model
      -> TACFN-style CrossModal + BetaGate fusion
      -> Emotion-level decoder (per-emotion queries)
      -> 6-dim logits (happy, sad, anger, fear, disgust, surprise)

Loss: BCEWithLogitsLoss with soft targets normalized to [0,1]
      (negative emotion values are clipped to 0, then divided by 3)

Implements:
- Cosine LR with warmup + gradient accumulation
- Optional β entropy regularizer
- AMP via torch.amp.*
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, roc_auc_score

from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder


# ========================
# Utils
# ========================

def set_seed(seed: int = 1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()

    # Data paths
    ap.add_argument("--index_csv", type=str, default="data/mosei_index_splits.csv")
    ap.add_argument("--audio_dir", type=str, default="features/mosei/seq_level/audio")
    ap.add_argument("--text_dir", type=str, default="features/mosei/seq_level/text")

    # CSV column names
    ap.add_argument("--uid_col", type=str, default="uid")
    ap.add_argument("--video_col", type=str, default="video_id")
    ap.add_argument("--split_col", type=str, default="split")

    # Emotion columns (MOSEI 6 emotion dimensions)
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

    # Model config
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--num_layers_fusion", type=int, default=2)
    ap.add_argument("--num_layers_decoder", type=int, default=2)
    ap.add_argument("--beta_hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training config
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--warmup_ratio", type=float, default=0.1, help="fraction of total steps for warmup")
    ap.add_argument("--beta_entropy", type=float, default=1e-3, help="λ for beta entropy regularizer (0 to disable)")

    # Optional sequence caps (for stability)
    ap.add_argument("--max_len_audio", type=int, default=300)
    ap.add_argument("--max_len_text", type=int, default=128)

    # Misc
    ap.add_argument("--out_dir", type=str, default="runs/mosei_fusion_decoder")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_train_samples", type=int, default=None)

    return ap.parse_args()


def multilabel_metrics_from_logits(logits, targets, threshold=0.5):
    """
    logits: [N, C] raw logits
    targets: [N, C] labels in original scale (e.g., [0,3] or [0,~1.7])
    评估按“是否表达该情绪”：targets>0 视为正类
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true_cont = targets.detach().cpu().numpy()
    y_true_bin = (y_true_cont > 0.0).astype(int)
    y_pred_bin = (probs >= threshold).astype(int)

    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)

    aucs = []
    for c in range(probs.shape[1]):
        col = y_true_bin[:, c]
        if (col.max() > 0) and (col.min() < 1):
            aucs.append(roc_auc_score(col, probs[:, c]))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    return micro_f1, macro_f1, macro_auc


# ========================
# Dataset
# ========================

def crop_center(x: torch.Tensor, max_len: int) -> torch.Tensor:
    """Center-crop sequence to max_len if too long."""
    if x.size(0) <= max_len:
        return x
    start = (x.size(0) - max_len) // 2
    return x[start:start + max_len]


class MoseiSeqDataset(Dataset):
    """
    One item returns:

        h_a: [L_a, d_audio]
        m_a: [L_a] bool mask, True = PAD, False = valid

        h_t: [L_t, d_text]
        m_t: [L_t] bool mask, True = PAD, False = valid

        y:   [C] float in original label scale (e.g., [0,3])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        text_dir: Path,
        uid_col: str,
        emo_cols: List[str],
        max_len_audio: int,
        max_len_text: int,
    ):
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.uid_col = uid_col
        self.emo_cols = emo_cols
        self.max_len_audio = max_len_audio
        self.max_len_text = max_len_text

        df = df.reset_index(drop=True)

        # Keep only rows with both audio & text features
        keep_indices = []
        missing = 0
        for i, row in df.iterrows():
            uid = str(row[uid_col])
            if (audio_dir / f"{uid}.pt").is_file() and (text_dir / f"{uid}.pt").is_file():
                keep_indices.append(i)
            else:
                missing += 1

        if missing > 0:
            print(f"[Dataset] Filtered out {missing} rows without both modalities.")

        self.df = df.iloc[keep_indices].reset_index(drop=True)
        print(f"[Dataset] Final size: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def _load_feat(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load feature dict:
            {
                "hidden": [L, D] float,
                "attention_mask": [L] (1 = valid, 0 = pad)  [optional]
            }

        Returns:
            h: [L, D] float32, cleaned (no NaN/Inf)
            mask: [L] bool, True = PAD, False = valid
        """
        obj = torch.load(path, map_location="cpu")
        h = obj["hidden"].float()

        # Safety: ensure no NaN/Inf
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        if "attention_mask" in obj:
            # convention in extracted features: 1 = valid, 0 = pad
            m = obj["attention_mask"].long()
        else:
            m = torch.ones(h.size(0), dtype=torch.long)

        mask = (m == 0)  # True = PAD

        if torch.isnan(h).any() or torch.isinf(h).any():
            raise ValueError(f"Found NaN/Inf in features at {path}")
        return h, mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uid = str(row[self.uid_col])

        a_path = self.audio_dir / f"{uid}.pt"
        t_path = self.text_dir / f"{uid}.pt"

        h_a, m_a = self._load_feat(a_path)
        h_t, m_t = self._load_feat(t_path)

        # Center-crop for stability (optional, controlled by args)
        if self.max_len_audio > 0:
            h_a = crop_center(h_a, self.max_len_audio)
            m_a = torch.zeros(h_a.size(0), dtype=torch.bool)
        if self.max_len_text > 0:
            h_t = crop_center(h_t, self.max_len_text)
            m_t = torch.zeros(h_t.size(0), dtype=torch.bool)

        # Build multi-label targets from MOSEI emotions (original scale)
        y = torch.tensor([float(row[c]) for c in self.emo_cols], dtype=torch.float32)
        y = torch.nan_to_num(y, nan=0.0)

        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError(f"Found NaN/Inf in labels for uid={uid}")

        return h_a, m_a, h_t, m_t, y


def collate_fn(batch):
    """
    Pad variable-length sequences in the batch.

    Input:
        list of (h_a[L_a,d_a], m_a[L_a], h_t[L_t,d_t], m_t[L_t], y[C])

    Output:
        pad_h_a: [B, L_a_max, d_a]
        pad_m_a: [B, L_a_max] bool, True = PAD
        pad_h_t: [B, L_t_max, d_t]
        pad_m_t: [B, L_t_max] bool, True = PAD
        y:       [B, C] (original scale, e.g., [0,3])
        d_a: int
        d_t: int
    """
    hs_a, ms_a, hs_t, ms_t, ys = zip(*batch)
    B = len(batch)

    d_a = hs_a[0].size(-1)
    d_t = hs_t[0].size(-1)

    L_a_max = max(x.size(0) for x in hs_a)
    L_t_max = max(x.size(0) for x in hs_t)

    pad_h_a = torch.zeros(B, L_a_max, d_a)
    pad_m_a = torch.ones(B, L_a_max, dtype=torch.bool)  # default PAD=True
    pad_h_t = torch.zeros(B, L_t_max, d_t)
    pad_m_t = torch.ones(B, L_t_max, dtype=torch.bool)

    for i in range(B):
        La = hs_a[i].size(0)
        Lt = hs_t[i].size(0)
        pad_h_a[i, :La] = hs_a[i]
        pad_m_a[i, :La] = ms_a[i]
        pad_h_t[i, :Lt] = hs_t[i]
        pad_m_t[i, :Lt] = ms_t[i]

    y = torch.stack(ys, dim=0)
    return pad_h_a, pad_m_a, pad_h_t, pad_m_t, y, d_a, d_t


# ========================
# Aux: β entropy regularizer
# ========================

def beta_entropy_loss(beta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    beta: [...], in [0,1] (if your model returns modality weight for audio or similar)
    Encourage decisive gates (lower entropy).
    """
    b = torch.clamp(beta, eps, 1.0 - eps)
    ent = -(b * torch.log(b) + (1 - b) * torch.log(1 - b))
    return ent.mean()

# ========================
# Label normalization
# ========================

def normalize_mosei_emotions(y: torch.Tensor) -> torch.Tensor:
    """
    Mainstream MOSEI normalization:
    - Negative values → 0
    - Divide by 3 to map [0,3] → [0,1]
    """
    y = torch.clamp(y, min=0.0, max=3.0)
    return y / 3.0

# ========================
# Train / Eval
# ========================

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, args):
    model.train()
    total_loss, total_samples = 0.0, 0
    beta_list = []
    all_logits, all_targets = [], []

    optimizer.zero_grad(set_to_none=True)

    for step, (h_a, m_a, h_t, m_t, y, _, _) in enumerate(tqdm(loader, desc="Train", leave=False)):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        y = y.to(device)

        with autocast('cuda', enabled=(device.type == "cuda")):
            logits, beta, _ = model(h_a, h_t, m_a, m_t)

            y_loss = normalize_mosei_emotions(y)

            raw_loss = criterion(logits, y_loss)
            if beta is not None and args.beta_entropy > 0:
                raw_loss = raw_loss + args.beta_entropy * beta_entropy_loss(beta)
            loss = raw_loss / args.grad_accum

        if torch.isnan(loss) or torch.isinf(loss):
            print("[Warn] NaN/Inf loss in train batch, skipping.")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # logging (sample-weighted)
        bs = y.size(0)
        total_loss += raw_loss.detach().float().cpu().item() * bs
        total_samples += bs

        if beta is not None:
            beta_list.extend(beta.detach().float().cpu().view(-1).tolist())

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu()) # keep original scale for metrics

    avg_loss = total_loss / max(1, total_samples)
    if avg_loss < 0:
        print(f"[Warn] avg_loss < 0 due to tiny fp error: {avg_loss:.6e}")
        avg_loss = float(abs(avg_loss))

    mean_beta = float(np.mean(beta_list)) if beta_list else 0.0

    if all_logits:
        all_logits = torch.cat(all_logits, 0)
        all_targets = torch.cat(all_targets, 0)
        micro_f1, macro_f1, macro_auc = multilabel_metrics_from_logits(all_logits, all_targets)
    else:
        micro_f1 = macro_f1 = macro_auc = 0.0

    return avg_loss, micro_f1, macro_f1, macro_auc, mean_beta


@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    model.eval()
    total_loss, total_samples = 0.0, 0
    beta_list = []
    all_logits, all_targets = [], []

    for h_a, m_a, h_t, m_t, y, _, _ in tqdm(loader, desc="Val", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        y = y.to(device)

        with autocast('cuda', enabled=(device.type == "cuda")):
            logits, beta, _ = model(h_a, h_t, m_a, m_t)

            y_loss = normalize_mosei_emotions(y)
            raw_loss = criterion(logits, y_loss)
            if beta is not None and args.beta_entropy > 0:
                raw_loss = raw_loss + args.beta_entropy * beta_entropy_loss(beta)

        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            print("[Warn] NaN/Inf loss in val batch, skipping.")
            continue

        bs = y.size(0)
        total_loss += raw_loss.item() * bs
        total_samples += bs

        if beta is not None:
            beta_list.extend(beta.detach().float().cpu().view(-1).tolist())

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())  # keep original scale for metrics

    avg_loss = total_loss / max(1, total_samples)
    if avg_loss < 0:
        print(f"[Warn] avg_loss < 0 due to tiny fp error: {avg_loss:.6e}")
        avg_loss = float(abs(avg_loss))

    mean_beta = float(np.mean(beta_list)) if beta_list else 0.0

    if all_logits:
        all_logits = torch.cat(all_logits, 0)
        all_targets = torch.cat(all_targets, 0)
        micro_f1, macro_f1, macro_auc = multilabel_metrics_from_logits(all_logits, all_targets)
    else:
        micro_f1 = macro_f1 = macro_auc = 0.0

    print(f"[Val Metrics] Loss={avg_loss:.4f} | micro-F1={micro_f1:.3f} | macro-F1={macro_f1:.3f} | macro-AUC={macro_auc:.3f}")
    return avg_loss, micro_f1, macro_f1, macro_auc, mean_beta


# ========================
# Main
# ========================

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load split index
    df = pd.read_csv(args.index_csv)
    train_df = df[df[args.split_col] == "train"].reset_index(drop=True)
    val_df = df[df[args.split_col] == "val"].reset_index(drop=True)

    if args.max_train_samples is not None and args.max_train_samples < len(train_df):
        train_df = train_df.sample(args.max_train_samples, random_state=42).reset_index(drop=True)
        print(f"[Info] Using train subset: {len(train_df)} samples")

    audio_dir = Path(args.audio_dir)
    text_dir = Path(args.text_dir)

    # Datasets
    train_ds = MoseiSeqDataset(
        train_df, audio_dir, text_dir, args.uid_col, args.emo_cols,
        max_len_audio=args.max_len_audio, max_len_text=args.max_len_text
    )
    val_ds = MoseiSeqDataset(
        val_df, audio_dir, text_dir, args.uid_col, args.emo_cols,
        max_len_audio=args.max_len_audio, max_len_text=args.max_len_text
    )

    # Dataloaders
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=pin, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=pin, collate_fn=collate_fn,
    )

    # Read feature dims from meta.json
    text_meta = json.loads((text_dir / "meta.json").read_text())
    audio_meta = json.loads((audio_dir / "meta.json").read_text())
    d_text = int(text_meta["hidden_dim"])
    d_audio = int(audio_meta["hidden_dim"])
    num_emotions = len(args.emo_cols)

    # Model
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

    # Optimizer / Loss / Scheduler / AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))

    # Cosine + warmup schedule over total train steps (accounting grad_accum)
    total_train_steps = (len(train_loader) * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_train_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))  # cosine in [0,1]

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        tr_loss, tr_f1_micro, tr_f1_macro, tr_auc_macro, tr_beta = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, args)
        va_loss, va_f1_micro, va_f1_macro, va_auc_macro, va_beta = evaluate(
            model, val_loader, criterion, device, args
        )

        print(
            f"Train Loss: {tr_loss:.4f} | "
            f"F1 micro/macro: {tr_f1_micro:.3f}/{tr_f1_macro:.3f} | "
            f"AUC macro: {tr_auc_macro:.3f} | Mean β: {tr_beta:.3f}  ||  "
            f"Val Loss: {va_loss:.4f} | "
            f"F1 micro/macro: {va_f1_micro:.3f}/{va_f1_macro:.3f} | "
            f"AUC macro: {va_auc_macro:.3f} | Mean β: {va_beta:.3f}"
        )

        # Save by macro-F1
        if va_f1_macro > best_val_macro_f1:
            best_val_macro_f1 = va_f1_macro
            best_state = {
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "epoch": epoch,
                "args": vars(args),
                "emo_cols": args.emo_cols
            }

    if best_state is not None:
        ckpt_path = Path(args.out_dir) / "best_mosei_fusion_decoder.pt"
        torch.save(best_state, ckpt_path)
        print(f"\n[Saved] Best model to {ckpt_path} (Val macro-F1 = {best_val_macro_f1:.4f})")
    else:
        print("[Warning] No valid model checkpoint saved.")


if __name__ == "__main__":
    main()