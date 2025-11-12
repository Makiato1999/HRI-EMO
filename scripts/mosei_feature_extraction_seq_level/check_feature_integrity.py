import pandas as pd
from pathlib import Path
import os
import torch

# df = pd.read_csv("data/mosei_index_splits.csv")
# uids = set(df["uid"])

# text_dir = Path("features/mosei/seq_level/text")
# audio_dir = Path("features/mosei/seq_level/audio")

# text_uids = {p.stem for p in text_dir.glob("*.pt")}
# audio_uids = {p.stem for p in audio_dir.glob("*.pt")}

# print("csv rows:", len(uids))
# print("text feats:", len(text_uids))
# print("audio feats:", len(audio_uids))
# print("only_in_csv_not_text:", len(uids - text_uids))
# print("only_in_csv_not_audio:", len(uids - audio_uids))
# print("only_text_not_audio:", len(text_uids - audio_uids))
# print("only_audio_not_text:", len(audio_uids - text_uids))


# df = pd.read_csv("data/mosei_index_splits.csv")
# print(df.isna().sum())

def scan(dir_path, name):
    dir_path = Path(dir_path)
    n = nan_files = inf_files = 0
    for p in dir_path.glob("*.pt"):
        n += 1
        obj = torch.load(p, map_location="cpu")

        if isinstance(obj, dict):
            x = obj.get("hidden", None)
        else:
            x = obj

        if x is None:
            continue

        if torch.isnan(x).any():
            nan_files += 1
        if torch.isinf(x).any():
            inf_files += 1

    print(f"{name}: total={n}, nan_files={nan_files}, inf_files={inf_files}")

if __name__ == "__main__":
    scan("features/mosei/seq_level/audio", "audio")
    scan("features/mosei/seq_level/text", "text")