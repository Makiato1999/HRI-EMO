# scripts/linear_probe_baseline.py
from pathlib import Path
import pandas as pd
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

A_DIR = Path("../features/audio")
T_DIR = Path("../features/text")
INDEX_PATH = Path("../data/iemocap_index_splits.csv")  # your CSV with split column

def mean_pool(x: torch.Tensor) -> torch.Tensor:
    """Average along time dimension if needed."""
    x = x.float()
    if x.dim() == 1:
        return x                    # [D]
    elif x.dim() == 2:
        return x.mean(0)            # [T,D] -> [D]
    else:
        x = x.view(x.size(0), -1)
        return x.mean(0)

def load_vec(pt_path: Path) -> torch.Tensor:
    """Robust loader: supports Tensor or {'feat': Tensor}."""
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        x = obj.get("feat", None)
        if x is None:
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    x = v; break
        if x is None:
            raise ValueError(f"No tensor found in {pt_path}")
    else:
        raise TypeError(f"Unsupported object type in {pt_path}: {type(obj)}")
    return mean_pool(x)

df = pd.read_csv(INDEX_PATH)

id_col    = "utter_id"
label_col = "label"   
split_col = "split"

# Normalize ID (remove extensions/paths)
from os.path import basename, splitext
df["_id_"] = df[id_col].astype(str).map(lambda v: splitext(basename(v.strip()))[0])

# ncode labels (supports string categories) 
labels_raw = df[label_col].astype(str).str.strip()
# Try int; if fails, fall back to category codes
try:
    df["_label_"] = labels_raw.astype(int)
    label_names = sorted(df["_label_"].unique().tolist())
    label_map_print = {int(k): str(k) for k in label_names}
except Exception:
    # Use pandas categorical codes to get stable 0..C-1 mapping
    cat = pd.Categorical(labels_raw)
    df["_label_"] = cat.codes.astype(int)   # -1 means NaN originally; guard below
    if (df["_label_"] < 0).any():
        raise ValueError("Label column contains missing values; please clean them.")
    # Build name->id and id->name mapping for printing
    label_names = list(cat.categories)
    label_map_print = {i: name for i, name in enumerate(label_names)}

print("Label mapping (id -> name):", label_map_print)

# Normalize split names to train/val/test
df["_split_"] = df[split_col].astype(str).str.lower().map(
    lambda s: "train" if "train" in s
    else ("val" if ("val" in s or "dev" in s)
    else ("test" if "test" in s else s))
)


a_ids = {p.stem for p in A_DIR.glob("*.pt")}
t_ids = {p.stem for p in T_DIR.glob("*.pt")}
both  = a_ids & t_ids
missing = set(df["_id_"]) - both
if missing:
    print(f"⚠️  Skipping {len(missing)} rows without both features. Example: {list(missing)[:5]}")
df = df[df["_id_"].isin(both)].reset_index(drop=True)


class FeatDS(Dataset):
    def __init__(self, frame: pd.DataFrame, split: str):
        self.df = frame[frame["_split_"] == split].reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        uid = r["_id_"]
        a = load_vec(A_DIR / f"{uid}.pt")
        t = load_vec(T_DIR / f"{uid}.pt")
        x = torch.cat([a, t], dim=-1)
        y = torch.tensor(int(r["_label_"]), dtype=torch.long)
        return x, y

splits = ["train", "val", "test"]
sizes = {s: int((df["_split_"] == s).sum()) for s in splits}
print("Split sizes:", sizes)

train_ds = FeatDS(df, "train")
val_ds   = FeatDS(df, "val")
test_ds  = FeatDS(df, "test") if sizes.get("test", 0) > 0 else None

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False) if test_ds else None

# Model (Linear Probe)
in_dim  = train_ds[0][0].numel()
num_cls = int(df["_label_"].max()) + 1
model   = nn.Linear(in_dim, num_cls)
opt     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
crit    = nn.CrossEntropyLoss()

def eval_loader(loader):
    model.eval()
    total = correct = 0
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            logit = model(x)
            pred  = logit.argmax(-1)
            preds.append(pred.cpu().numpy())
            gts.append(y.cpu().numpy())
            correct += (pred == y).sum().item()
            total   += y.numel()
    acc = correct / total if total else 0.0
    try:
        from sklearn.metrics import f1_score
        f1 = f1_score(np.concatenate(gts), np.concatenate(preds), average="macro")
    except Exception:
        f1 = float("nan")
    return acc, f1

# Training
best = 0.0
for epoch in range(10):
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    val_acc, val_f1 = eval_loader(val_loader)
    best = max(best, val_acc)
    print(f"Epoch {epoch:02d} | Val Acc={val_acc:.3f} | Val F1={val_f1:.3f} | Best Acc={best:.3f}")


if test_loader:
    test_acc, test_f1 = eval_loader(test_loader)
    print(f"[TEST] Acc={test_acc:.3f} | Macro-F1={test_f1:.3f}")
