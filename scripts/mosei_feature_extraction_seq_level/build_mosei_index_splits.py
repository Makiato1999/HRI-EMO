#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mmsdk import mmdatasdk


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        default="data/MOSEI",
        help="Folder where CMU_MOSEI_Labels.csd is stored",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="data/mosei_index_splits.csv",
        help="Path to save generated index splits CSV",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- 只用本地 label CSD，不从网上拉 -----
    csd_path = data_root / "CMU_MOSEI_Labels.csd"
    if not csd_path.exists():
        raise FileNotFoundError(
            f"{csd_path} not found.\n"
            f"请确认 CMU_MOSEI_Labels.csd 在 {data_root} 下，"
            f"或用 --data_root 指到包含该文件的目录。"
        )

    mosei = mmdatasdk.cmu_mosei
    # mosei.labels: 例如 {"All Labels": "CMU_MOSEI_Labels.csd"}
    label_key = next(iter(mosei.labels.keys()))

    print(f"[Info] Loading labels from local file: {csd_path}")
    label_ds = mmdatasdk.mmdataset({label_key: str(csd_path)})

    data = label_ds.computational_sequences[label_key].data

    # 官方给的是 video-level folds
    folds = mosei.standard_folds
    train_vids = set(folds.standard_train_fold)
    val_vids = set(folds.standard_valid_fold)
    test_vids = set(folds.standard_test_fold)

    rows = []

    # 每个 vid:
    #   obj["features"]: [num_segs, 7]
    #   7 dims: [sentiment, happy, sad, anger, fear, disgust, surprise]
    for vid, obj in data.items():
        feats = obj["features"]
        num_segs = feats.shape[0]

        if vid in train_vids:
            split = "train"
        elif vid in val_vids:
            split = "val"
        elif vid in test_vids:
            split = "test"
        else:
            # 不在标准划分里的就丢掉
            continue

        for seg_idx in range(num_segs):
            uid = f"{vid}_{seg_idx}"
            l = feats[seg_idx].astype(float)

            rows.append(
                {
                    "uid": uid,
                    "video_id": vid,
                    "seg_idx": seg_idx,
                    "sentiment": l[0],
                    "emo_happy": l[1],
                    "emo_sad": l[2],
                    "emo_anger": l[3],
                    "emo_fear": l[4],
                    "emo_disgust": l[5],
                    "emo_surprise": l[6],
                    "split": split,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
