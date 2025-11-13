#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_metrics.py
  从推理结果绘制各类性能图表（F1/AP/阈值/PR/ROC）。
  自动适配本地与 Colab，只需修改 BASE_PATH_MODE。
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

# ----------- 全局路径配置 ----------
BASE_PATH_MODE = "colab"   # 🔁 修改为 "local" 或 "colab"

if BASE_PATH_MODE == "colab":
    BASE_DIR = "/content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_small"
else:  # local
    BASE_DIR = "./HRI-EMO-results/mosei_fusion_decoder_small"

# 默认路径（会被命令行参数覆盖）
DEFAULT_BASE_DIR = f"{BASE_DIR}/infer_outputs"
DEFAULT_CKPT = f"{BASE_DIR}/best_mosei_fusion_decoder.pt"
DEFAULT_OUT_DIR = f"{BASE_DIR}/plots"

# ----------- 其余逻辑 ----------
EMO = ["emo_happy","emo_sad","emo_anger","emo_fear","emo_disgust","emo_surprise"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def overall_metrics(y_true_bin, y_prob, y_pred_bin):
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)

    aucs = []
    for i in range(y_prob.shape[1]):
        col = y_true_bin[:, i]
        if col.max() > 0 and col.min() < 1:
            aucs.append(roc_auc_score(col, y_prob[:, i]))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    return micro_f1, macro_f1, macro_auc

def save_text_metrics(out_dir: Path, micro_f1, macro_f1, macro_auc):
    txt = (f"micro-F1: {micro_f1:.4f}\n"
           f"macro-F1: {macro_f1:.4f}\n"
           f"macro-AUC: {macro_auc:.4f}\n")
    (out_dir / "overall_metrics.txt").write_text(txt)
    plt.figure(figsize=(4,2.2))
    plt.axis("off")
    plt.text(0.0, 0.7, f"micro-F1 : {micro_f1:.4f}")
    plt.text(0.0, 0.4, f"macro-F1 : {macro_f1:.4f}")
    plt.text(0.0, 0.1, f"macro-AUC: {macro_auc:.4f}")
    plt.tight_layout()
    plt.savefig(out_dir / "overall_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()

def bar_plot(values, labels, ylabel, title, out_path: Path):
    plt.figure(figsize=(8,4))
    x = np.arange(len(labels))
    plt.bar(x, values, color="#5DADE2")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()

def plot_pr_curve(y_true, y_prob, emo_name, out_path: Path):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if y_true.max() > 0 else 0.0
    plt.figure(figsize=(4.2,3.6))
    plt.plot(r, p, color="#F39C12")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{emo_name} PR (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()

def plot_roc_curve(y_true, y_prob, emo_name, out_path: Path):
    if y_true.max() == 0 or y_true.min() == 1:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    au = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(4.2,3.6))
    plt.plot(fpr, tpr, color="#45B39D")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{emo_name} ROC (AUC={au:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap.add_argument("--plot_roc", action="store_true", help="额外保存每类 ROC 图")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir  = ensure_dir(Path(args.out_dir))
    split = args.split

    # 数据
    y_prob = np.load(base_dir / f"{split}_y_prob.npy")
    y_true_cont = np.load(base_dir / f"{split}_y_true.npy")
    y_true = (y_true_cont > 0).astype(int)

    # 阈值
    ckpt = torch.load(args.ckpt, map_location="cpu")
    thresholds = np.array(ckpt.get("val_calibrated_thresholds", [0.5]*len(EMO)), dtype=float)

    # 预测
    y_pred = (y_prob >= thresholds[None, :]).astype(int)

    # 整体指标
    micro_f1, macro_f1, macro_auc = overall_metrics(y_true, y_prob, y_pred)
    save_text_metrics(out_dir, micro_f1, macro_f1, macro_auc)

    # 每类统计
    f1_list, ap_list = [], []
    for i in range(len(EMO)):
        f1_list.append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        ap_list.append(average_precision_score(y_true[:, i], y_prob[:, i]) if y_true[:, i].max() > 0 else 0.0)

    bar_plot(f1_list, EMO, "F1", f"{split.upper()} per-class F1", out_dir / f"{split}_bar_f1.png")
    bar_plot(ap_list, EMO, "AP", f"{split.upper()} per-class AP", out_dir / f"{split}_bar_ap.png")
    bar_plot(thresholds, EMO, "threshold", f"{split.upper()} thresholds", out_dir / f"{split}_bar_thresholds.png")

    pr_dir = ensure_dir(out_dir / f"{split}_PR_curves")
    roc_dir = ensure_dir(out_dir / f"{split}_ROC_curves") if args.plot_roc else None

    for i, emo in enumerate(EMO):
        plot_pr_curve(y_true[:, i], y_prob[:, i], emo, pr_dir / f"{emo}.png")
        if args.plot_roc and roc_dir is not None:
            plot_roc_curve(y_true[:, i], y_prob[:, i], emo, roc_dir / f"{emo}.png")

    print(f"[✓] Figures saved to {out_dir}")

if __name__ == "__main__":
    main()
