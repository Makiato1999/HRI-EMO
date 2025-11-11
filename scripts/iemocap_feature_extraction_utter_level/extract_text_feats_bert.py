#!/usr/bin/env python3
import os, math, json, argparse, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/iemocap_index_splits.csv")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out_dir", default="features/utter_level/text")
    return ap.parse_args()

class TextDS(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(
            r["text"],
            padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "utter_id": r["utter_id"]
        }

# Main feature extraction
@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset index (text + utter_id + label)
    df = pd.read_csv(args.csv)

    # Create output directory if not exists
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    # Build DataLoader
    ds = TextDS(df, tokenizer, args.max_len)
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2
    )

    # Define mean pooling function to get a single embedding per sentence
    def mean_pool(last_hidden_state, attention_mask):
        # Mask out padding tokens and average valid token embeddings
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float() 
        # attention_mask is [B, T], unsqueeze to [B, T, 1], expand to [B, T, H]
        # mask is now [B, T, H] with 1.0 for valid tokens and 0.0 for padding
        masked = last_hidden_state * mask
        # masked is [B, T, H], element-wise multiply
        # masked.sum(1) is [B, H], sum of all valid token embeddings
        # mask.sum(1) is [B, 1], number of valid tokens per sentence
        # return is [B, H], mean embedding per sentence
        return masked.sum(1) / mask.sum(1).clamp(min=1e-9)
    
    # Loop through all batches and extract embeddings
    saved, total = 0, len(ds)
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        # Use mixed-precision (FP16) inference for faster performance on GPU
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            out = model(input_ids=input_ids, attention_mask=attn)
            # out.last_hidden_state shape: [batch_size, seq_len, hidden_dim], [64, 128, 768]
            emb = mean_pool(out.last_hidden_state, attn)   # Shape: [batch_size, hidden_dim]

        # Save each utterance embedding separately as a .pt tensor
        for vec, uid in zip(emb, batch["utter_id"]):
            torch.save(vec.detach().cpu(), out_dir / f"{uid}.pt")
            saved += 1

    # Save metadata about the model and embedding configuration
    meta = {
        "model": args.model_name,
        "dim": int(emb.size(-1)), # embedding dimension (e.g., 768 for BERT-base)
        "max_len": args.max_len
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {saved}/{total} text embeddings to {out_dir}")

if __name__ == "__main__":
    main()