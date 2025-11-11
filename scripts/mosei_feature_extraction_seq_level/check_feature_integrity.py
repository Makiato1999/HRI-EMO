import pandas as pd
from pathlib import Path
import os

df = pd.read_csv("data/mosei_index_splits.csv")
uids = set(df["uid"])

text_dir = Path("features/mosei/seq_level/text")
audio_dir = Path("features/mosei/seq_level/audio")

text_uids = {p.stem for p in text_dir.glob("*.pt")}
audio_uids = {p.stem for p in audio_dir.glob("*.pt")}

print("csv rows:", len(uids))
print("text feats:", len(text_uids))
print("audio feats:", len(audio_uids))
print("only_in_csv_not_text:", len(uids - text_uids))
print("only_in_csv_not_audio:", len(uids - audio_uids))
print("only_text_not_audio:", len(text_uids - audio_uids))
print("only_audio_not_text:", len(audio_uids - text_uids))
