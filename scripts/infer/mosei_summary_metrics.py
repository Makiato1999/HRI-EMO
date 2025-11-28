import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_dir", type=str, required=True, help="Directory containing .npy outputs")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to best checkpoint (to load calibrated thresholds)")
    parser.add_argument("--split", type=str, default="test", help="Which split to evaluate (val or test)")
    args = parser.parse_args()

    infer_dir = Path(args.infer_dir)
    print(f"--- Loading results from: {infer_dir} ---")

    # 1. Load prediction results (.npy)
    prob_path = infer_dir / f"{args.split}_y_prob.npy"
    true_path = infer_dir / f"{args.split}_y_true.npy"

    if not prob_path.exists() or not true_path.exists():
        print(f"Error: Could not find .npy files for split '{args.split}' in {infer_dir}")
        return

    probs = np.load(prob_path)
    y_true_cont = np.load(true_path)
    # MOSEI Standard: >0 is positive
    # Convert continuous/ratio labels (e.g., MOSEI) to binary labels
    y_true = (y_true_cont > 0).astype(int) 

    EMO = ["Happy", "Sad", "Anger", "Fear", "Disgust", "Surprise"]

    # 2. Determine Thresholds
    if args.ckpt:
        print(f"--- Loading thresholds from: {args.ckpt} ---")
        try:
            ckpt = torch.load(args.ckpt, map_location="cpu")
            # Attempt to get saved thresholds; use 0.5 if not found
            ths = np.array(ckpt.get("val_calibrated_thresholds", [0.5]*len(EMO)), dtype=float) 
            print(f"Using Calibrated Thresholds: {np.round(ths, 3)}")
        except Exception as e:
            print(f"Warning: Failed to load ckpt ({e}), utilizing default 0.5")
            ths = np.full(len(EMO), 0.5)
    else:
        print("--- No checkpoint provided, using default threshold 0.5 ---")
        ths = np.full(len(EMO), 0.5)

    # 3. Calculate Overall Metrics
    # Apply thresholds to get binary predictions
    y_pred = (probs >= ths[None, :]).astype(int) 

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    aucs = []
    for i in range(len(EMO)):
        col = y_true[:, i]
        # Only calculate AUC if the class has both positive and negative examples
        if col.max() > 0 and col.min() < 1: 
            aucs.append(roc_auc_score(col, probs[:, i]))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0

    print("\n" + "="*40)
    print(f"📢 Overall Results ({args.split.upper()})")
    print("="*40)
    print(f"Micro-F1 : {micro_f1:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"Macro-AUC: {macro_auc:.4f}")
    print("="*40)

    # 4. Calculate Per-Class Detailed Metrics
    per_cls = []
    for i, name in enumerate(EMO):
        y = y_true[:, i]
        p = probs[:, i]
        # Use the specific threshold for this class
        yhat = (p >= ths[i]).astype(int) 
        
        score = f1_score(y, yhat, zero_division=0)
        # Count the number of positive instances (Support)
        support = int(y.sum()) 
        
        per_cls.append({
            "Emotion": name,
            "Threshold": ths[i],
            "F1": score,
            "Support": support
        })

    df = pd.DataFrame(per_cls)
    print("\n--- Per-Class Breakdown ---")
    # Display the breakdown table, rounding to 3 decimal places
    print(df.round(3).to_string(index=False)) 
    
    # Optional: Save to CSV
    csv_path = infer_dir / f"{args.split}_summary_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Metrics table to: {csv_path}")

if __name__ == "__main__":
    main()