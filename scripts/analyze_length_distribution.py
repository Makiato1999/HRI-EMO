# scripts/analyze_length_distribution.py
from pathlib import Path
import numpy as np, torch

A_DIR = Path("../features/audio")
T_DIR = Path("../features/text")

def length_of(p):
    x = torch.load(p, map_location="cpu")
    x = x["feat"] if isinstance(x, dict) else x
    return x.shape[0] if x.dim() == 2 else 1

a_lens = [length_of(p) for p in A_DIR.glob("*.pt")]
t_lens = [length_of(p) for p in T_DIR.glob("*.pt")]

print("Audio: median =", np.median(a_lens), "95th =", np.percentile(a_lens, 95))
print("Text : median =", np.median(t_lens), "95th =", np.percentile(t_lens, 95))

import pandas as pd
pd.DataFrame({"audio_len": a_lens, "text_len": t_lens}).to_csv("../data/length_stats.csv", index=False)
print("Saved to data/length_stats.csv")
