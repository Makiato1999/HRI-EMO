# scripts/check_feature_integrity.py
from pathlib import Path
import random, torch, pandas as pd

A_DIR = Path("../features/audio")   # directory for audio features (.pt)
T_DIR = Path("../features/text")    # directory for text features (.pt)
INDEX_PATH = Path("../data/iemocap_index.csv")  # index file containing utterance IDs

# Check number of files and duplicates
a_ids = sorted([p.stem for p in A_DIR.glob("*.pt")])
t_ids = sorted([p.stem for p in T_DIR.glob("*.pt")])
print(f"Audio count: {len(a_ids)}, Text count: {len(t_ids)}")

assert len(a_ids) == len(t_ids), "Error: audio and text counts do not match"
assert len(set(a_ids)) == len(a_ids), "Error: duplicate IDs found in audio"
assert len(set(t_ids)) == len(t_ids), "Error: duplicate IDs found in text"

# Check ID consistency between audio and text
print("audio - text:", set(a_ids) - set(t_ids))
print("text  - audio:", set(t_ids) - set(a_ids))

# Check consistency with index.csv
df = pd.read_csv(INDEX_PATH)
csv_ids = set(df["utter_id"].astype(str))

assert csv_ids == set(a_ids) == set(t_ids), "Error: mismatch between index.csv and feature IDs"
print("File counts and index file are fully consistent!")

# Sample check for NaN/Inf values and tensor shape
random.seed(42)
sample_ids = random.sample(a_ids, 10)

def check_tensor(pt_path):
    """Load a .pt file, check for invalid values, and print shape + basic stats."""
    obj = torch.load(pt_path, map_location="cpu")
    x = obj["feat"] if isinstance(obj, dict) else obj
    assert torch.isfinite(x).all(), f"{pt_path} contains NaN or Inf"
    print(f"{pt_path.name:25s} shape={tuple(x.shape)} mean={x.mean():.3f} std={x.std():.3f}")

for uid in sample_ids:
    check_tensor(A_DIR / f"{uid}.pt")
    check_tensor(T_DIR / f"{uid}.pt")

print("Random sample check passed — all features are healthy.")
