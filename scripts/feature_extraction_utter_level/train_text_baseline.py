#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../data/iemocap_index_splits.csv")
    ap.add_argument("--feat_dir", default="../features/text", help="directory with .pt embeddings")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mlp_hidden", type=int, default=0, help="0 = linear head; >0 = 1-layer MLP")
    ap.add_argument("--out_dir", default="runs/train_textonly_baseline")
    return ap.parse_args()

class FeatDS(Dataset):
    def __init__(self, df, feat_dir, label2id):
        self.df = df.reset_index(drop=True)
        self.root = Path(feat_dir)
        self.label2id = label2id
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = torch.load(self.root / f"{r['utter_id']}.pt")  # [H]
        y = self.label2id[r["label"]]
        return x.float(), torch.tensor(y, dtype=torch.long)

# Model Builder
def build_model(in_dim, n_cls, mlp_hidden=0):
    if mlp_hidden and mlp_hidden > 0:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.1),
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, n_cls)
        )
    else:
        return nn.Linear(in_dim, n_cls)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        preds.extend(pred.cpu().tolist())
        gts.extend(y.cpu().tolist())
    acc = accuracy_score(gts, preds)
    f1m = f1_score(gts, preds, average="macro") # Equal weight for all classes
    f1w = f1_score(gts, preds, average="weighted") # Weighted by class frequency
    return {"acc": acc, "f1_macro": f1m, "f1_weighted": f1w}

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    exist = df["utter_id"].map(lambda u: Path(args.feat_dir, f"{u}.pt").exists())
    df = df[exist].copy()

    labels = sorted(df["label"].unique())
    label2id = {lab:i for i,lab in enumerate(labels)}
    n_cls = len(labels)

    sample_vec = torch.load(Path(args.feat_dir, f"{df.iloc[0]['utter_id']}.pt"))
    in_dim = int(sample_vec.numel())

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]

    ds_tr = FeatDS(tr, args.feat_dir, label2id)
    ds_va = FeatDS(va, args.feat_dir, label2id)
    ds_te = FeatDS(te, args.feat_dir, label2id)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(in_dim, n_cls, args.mlp_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_va = -1.0
    ckpt = out_dir / "best.pt"

    # Training
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dl_tr, desc=f"Epoch {epoch:02d}", leave=False)
        
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        va_metrics = evaluate(model, dl_va, device)
        va_score = va_metrics["f1_macro"]

        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(ds_tr):.4f} "
              f"| val_acc={va_metrics['acc']:.4f} | val_f1m={va_score:.4f}")
        
        # Save model if validation F1 improves
        if va_score > best_va:
            best_va = va_score
            torch.save({"model": model.state_dict(),
                        "in_dim": in_dim, "n_cls": n_cls,
                        "mlp_hidden": args.mlp_hidden,
                        "label2id": label2id},
                       ckpt)

    # Test
    state = torch.load(ckpt, map_location="cpu")
    model = build_model(state["in_dim"], state["n_cls"], state["mlp_hidden"]).to(device)
    model.load_state_dict(state["model"])
    te_metrics = evaluate(model, dl_te, device)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"val_best_f1_macro": best_va,
                   "test_metrics": te_metrics,
                   "labels": labels}, f, indent=2)

    print("Test:", te_metrics)

if __name__ == "__main__":
    main()
