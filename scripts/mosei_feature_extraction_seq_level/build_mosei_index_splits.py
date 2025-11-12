#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from mmsdk import mmdatasdk

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/MOSEI")
    ap.add_argument("--out_csv", type=str, default="data/mosei_index_splits.csv")
    return ap.parse_args()

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csd_path = data_root / "CMU_MOSEI_Labels.csd"
    if not csd_path.exists():
        raise FileNotFoundError(f"{csd_path} not found.")

    mosei = mmdatasdk.cmu_mosei
    label_key = next(iter(mosei.labels.keys()))

    print(f"[Info] Loading labels from local file: {csd_path}")
    label_ds = mmdatasdk.mmdataset({label_key: str(csd_path)})
    data = label_ds.computational_sequences[label_key].data

    folds = mosei.standard_folds
    train_vids = set(folds.standard_train_fold)
    val_vids   = set(folds.standard_valid_fold)
    test_vids  = set(folds.standard_test_fold)

    rows = []
    neg_counter = 0

    for vid, obj in data.items():
        feats = obj["features"]  # [num_segs,7] = [sentiment, 6 emos]
        num_segs = feats.shape[0]

        if vid in train_vids:
            split = "train"
        elif vid in val_vids:
            split = "val"
        elif vid in test_vids:
            split = "test"
        else:
            continue

        for seg_idx in range(num_segs):
            uid = f"{vid}_{seg_idx}"
            l = feats[seg_idx].astype(float)
            sent = float(l[0])
            emos = l[1:7].astype(float)

            if (emos < 0).any():
                neg_counter += int((emos < 0).sum())

            rows.append({
                "uid": uid,
                "video_id": vid,
                "seg_idx": seg_idx,
                "sentiment": sent,
                "emo_happy":   emos[0],
                "emo_sad":     emos[1],
                "emo_anger":   emos[2],
                "emo_fear":    emos[3],
                "emo_disgust": emos[4],
                "emo_surprise":emos[5],
                "split": split,
            })

    df = pd.DataFrame(rows)

    emo_cols = ["emo_happy","emo_sad","emo_anger","emo_fear","emo_disgust","emo_surprise"]
    print("[Sanity] per-emotion min/max (raw, unchanged):")
    for c in emo_cols:
        print(f"  {c}: min={df[c].min():.4f}, max={df[c].max():.4f}")
    if neg_counter > 0:
        print(f"[Sanity] Observed {neg_counter} negative emotion values in raw CSD (kept as-is).")

    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
