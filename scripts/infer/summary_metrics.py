#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summary_metrics.py
  用于推理后指标汇总：
    - 从 test_y_prob.npy / test_y_true.npy 计算 micro/macro F1、macro AUC
    - 输出每类 emotion 的阈值、F1、AP
    - 自动从 ckpt 读取保存的阈值（val_calibrated_thresholds）
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# ---------- 常量 ----------
EMO = ["emo_happy","emo_sad","emo_anger","emo_fear","emo_disgust","emo_surprise"]

# ---------- 主函数 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_small/infer_outputs",
                    help="存放 y_prob.npy 和 y_true.npy 的路径（可换成本地路径）")
    ap.add_argument("--ckpt", type=str, default="/content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_small/best_mosei_fusion_decoder.pt",
                    help="模型 checkpoint 路径，用于读取 val_calibrated_thresholds")
    ap.add_argument("--out_csv", type=str, default="summary_metrics.csv")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_csv = Path(args.out_csv)

    # ---------- 加载 ----------
    y_prob = np.load(base_dir / "test_y_prob.npy")
    y_true_cont = np.load(base_dir / "test_y_true.npy")
    y_true = (y_true_cont > 0).astype(int)

    # ---------- 阈值 ----------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ths = np.array(ckpt.get("val_calibrated_thresholds", [0.5]*len(EMO)), dtype=float)
    print(f"[✓] Loaded thresholds from ckpt ({len(ths)})")

    y_pred = (y_prob >= ths[None, :]).astype(int)

    # ---------- 整体指标 ----------
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    aucs = []
    for i in range(len(EMO)):
        col = y_true[:, i]
        if col.max() > 0 and col.min() < 1:
            aucs.append(roc_auc_score(col, y_prob[:, i]))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0

    print("\n=== Overall Metrics ===")
    print(f"micro-F1 : {micro_f1:.4f}")
    print(f"macro-F1 : {macro_f1:.4f}")
    print(f"macro-AUC: {macro_auc:.4f}")

    # ---------- 各类指标 ----------
    rows = []
    for i, emo in enumerate(EMO):
        y = y_true[:, i]
        p = y_prob[:, i]
        yhat = y_pred[:, i]
        ap = average_precision_score(y, p) if y.max() > 0 else 0.0
        rows.append({
            "emotion": emo,
            "threshold": ths[i],
            "f1": f1_score(y, yhat, zero_division=0),
            "AP": ap,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n[Saved] per-class results ->", out_csv)
    print(df.round(4))

if __name__ == "__main__":
    main()
