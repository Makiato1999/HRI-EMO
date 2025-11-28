import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

# ----------- Default Configuration ----------
# Default values are written here, but command-line arguments are recommended
DEFAULT_BASE_DIR = "/content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs"
DEFAULT_CKPT = "/content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/best_mosei_fusion_decoder.pt"
DEFAULT_OUT_DIR = "/content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/plots"

EMO = ["Happy", "Sad", "Anger", "Fear", "Disgust", "Surprise"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] # Dedicated color palette

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

# 1. Save text metrics
def save_text_metrics(out_dir: Path, micro_f1, macro_f1, macro_auc):
    txt = (f"micro-F1: {micro_f1:.4f}\n"
           f"macro-F1: {macro_f1:.4f}\n"
           f"macro-AUC: {macro_auc:.4f}\n")
    (out_dir / "overall_metrics.txt").write_text(txt)
    print("--- Overall Metrics Saved ---")
    print(txt)

# 2. Plot bar chart (F1, AP, etc.)
def bar_plot(values, labels, ylabel, title, out_path: Path):
    plt.figure(figsize=(8,4))
    x = np.arange(len(labels))
    plt.bar(x, values, color="#5DADE2", alpha=0.8, edgecolor='black')
    plt.xticks(x, labels, rotation=0) # No rotation needed for short English labels
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# 3. Plot single PR curve
def plot_single_pr(y_true, y_prob, emo_name, out_path: Path):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    # Calculate AP, handling cases with no positive samples
    ap = average_precision_score(y_true, y_prob) if y_true.max() > 0 else 0.0 
    plt.figure(figsize=(5, 4))
    plt.plot(r, p, color="#F39C12", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{emo_name} (AP={ap:.2f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# 4. [NEW] Plot combined PR curve (Figure for publications)
def plot_combined_pr(y_true, y_prob, out_path: Path):
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12}) # Adjust font size

    for i, name in enumerate(EMO):
        # Only plot if the class has positive samples
        if y_true[:, i].sum() > 0:
            prec, rec, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
            plt.plot(rec, prec, lw=2.5, color=COLORS[i], label=f'{name} (AP={ap:.2f})')
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Multi-label Emotion Recognition)")
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.grid(alpha=0.4, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Combined PR curve for publication saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    infer_dir = Path(args.infer_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    
    print(f"--- Plotting metrics for {args.split.upper()} set ---")

    # Load data
    probs = np.load(infer_dir / f"{args.split}_y_prob.npy")
    y_true_cont = np.load(infer_dir / f"{args.split}_y_true.npy")
    # Convert continuous/ratio labels (e.g., MOSEI) to binary labels
    y_true = (y_true_cont > 0).astype(int) 

    # Load thresholds
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        # Use calibrated thresholds if available, otherwise default to 0.5
        ths = np.array(ckpt.get("val_calibrated_thresholds", [0.5]*len(EMO)), dtype=float) 
        print(f"Using thresholds: {np.round(ths, 2)}")
    except Exception:
        # Fallback if checkpoint cannot be loaded
        ths = np.full(len(EMO), 0.5) 
        print("Warning: Could not load checkpoint. Using default threshold of 0.5 for all classes.")

    # Apply thresholds to get binary predictions
    y_pred = (probs >= ths[None, :]).astype(int)

    # 1. Calculate overall metrics
    from sklearn.metrics import f1_score, roc_auc_score
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    aucs = []
    for i in range(len(EMO)):
        col = y_true[:, i]
        # Only calculate AUC if the class has both positive and negative examples
        if col.max() > 0 and col.min() < 1: 
            aucs.append(roc_auc_score(col, probs[:, i]))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0

    save_text_metrics(out_dir, micro_f1, macro_f1, macro_auc)

    # 2. Calculate per-class AP and F1 for bar plots
    f1_list, ap_list = [], []
    for i in range(len(EMO)):
        f1_list.append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        # Handle case where a class has no positive samples
        ap_list.append(average_precision_score(y_true[:, i], probs[:, i]) if y_true[:, i].max() > 0 else 0.0)

    # 3. Plotting
    # A. Bar Charts
    bar_plot(f1_list, EMO, "F1 Score", f"{args.split.upper()} F1 per Emotion", out_dir / f"{args.split}_bar_f1.png")
    bar_plot(ap_list, EMO, "AP Score", f"{args.split.upper()} AP per Emotion", out_dir / f"{args.split}_bar_ap.png")
    
    # B. Combined PR Curve (The most important figure!)
    plot_combined_pr(y_true, probs, out_dir / f"{args.split}_combined_PR_curve.png")

    # C. Individual PR Curves (For archive/reference)
    pr_subdir = ensure_dir(out_dir / f"{args.split}_individual_PR")
    for i, name in enumerate(EMO):
        plot_single_pr(y_true[:, i], probs[:, i], name, pr_subdir / f"{name}.png")

    print(f"\n✅ All plots saved to: {out_dir}")

if __name__ == "__main__":
    main()