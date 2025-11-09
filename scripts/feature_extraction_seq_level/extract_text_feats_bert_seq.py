#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/iemocap_index_splits.csv")
    ap.add_argument("--uid_col", default="utter_id",
                    help="Column name for unique utterance ID")
    ap.add_argument("--text_col", default="text",
                    help="Column name for transcript text")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out_dir", default="features/seq_level/text")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    hidden_dim = None
    saved = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting text seq-level"):
        uid = row[args.uid_col]
        text = str(row[args.text_col])

        # Tokenize
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Forward
        outputs = model(**enc)
        hidden = outputs.last_hidden_state.squeeze(0).cpu()   # [L, H]
        attn = enc["attention_mask"].squeeze(0).cpu()         # [L]

        if hidden_dim is None:
            hidden_dim = int(hidden.size(-1))

        # Save one file per utterance
        torch.save(
            {
                "hidden": hidden,        # [L, H]
                "attention_mask": attn,  # [L]
            },
            out_dir / f"{uid}.pt",
        )
        saved += 1

    meta = {
        "model": args.model_name,
        "hidden_dim": hidden_dim,
        "max_len": args.max_len,
        "note": "seq-level BERT features: hidden[L,H] + attention_mask[L]",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {saved} text seq-level embeddings to {out_dir}")


if __name__ == "__main__":
    main()
