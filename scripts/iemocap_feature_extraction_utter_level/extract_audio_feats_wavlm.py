#!/usr/bin/env python3
import os, json, argparse, random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoFeatureExtractor, AutoModel
from tqdm.auto import tqdm
import numpy as np 


SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/iemocap_index_splits.csv")
    ap.add_argument("--model_name", default="microsoft/wavlm-base-plus",
                    help="e.g., microsoft/wavlm-base-plus or facebook/hubert-base-ls960")
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--max_seconds", type=float, default=10.0,
                    help="truncate/pad to this length for batching")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_dir", default="features/utter_level/audio")
    return ap.parse_args()

class AudioDS(Dataset):
    def __init__(self, df, target_sr, max_seconds):
        self.df = df.reset_index(drop=True)
        self.sr = target_sr
        self.max_len = int(target_sr * max_seconds)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        wav, sr = torchaudio.load(r["audio_path"])  # torch.Tensor [C, T] on CPU
        wav = wav.mean(0, keepdim=True)             # stereo -> mono [1, T]
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)  # [1, T']

        wav = wav.squeeze(0)                        # [T]
        mx = float(wav.abs().max())
        if mx > 0:
            wav = wav / mx

        if self.max_len and wav.numel() > self.max_len:
            wav = wav[:self.max_len]

        wav_np = wav.detach().cpu().numpy().astype(np.float32)  # shape [T]
        return {"audio": wav_np, "utter_id": r["utter_id"]}

def collate(batch):
    audio_list = [b["audio"] for b in batch]   # list[np.ndarray (T_i,)]
    uids = [b["utter_id"] for b in batch]
    return {"audio_list": audio_list, "utter_id": uids}

def downsample_mask_linear(mask_B_L: torch.Tensor, Tprime: int) -> torch.Tensor:
    B, L = mask_B_L.shape
    idx = torch.linspace(0, L - 1, steps=Tprime, device=mask_B_L.device).round().long()
    idx = idx.clamp_(0, L - 1)
    return mask_B_L[:, idx]  # [B, T']

# Main feature extraction
@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset index (audio_path + utter_id + label)
    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only rows with valid audio paths
    df = df[df["audio_path"].notna()].copy()

    # Load pretrained speech model and processor from Hugging Face
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    # Build DataLoader
    ds = AudioDS(df, args.target_sr, args.max_seconds)
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=collate, 
        pin_memory=True
    )

    def time_mean_pool(last_hidden_state, attention_mask_down):
        # last_hidden_state: [B, T', H]; attn_mask_down: [B, T']
        m = attention_mask_down.unsqueeze(-1).float()   # [B, T', 1]
        masked = last_hidden_state * m                  # [B, T', H]
        return masked.sum(1) / m.sum(1).clamp_min(1e-9) # [B, H]

    saved, total = 0, len(ds)
    dim = None

    pbar = tqdm(dl, total=len(dl), desc="Extracting audio embeddings", dynamic_ncols=True)

    for batch in pbar:
        inputs = feature_extractor(
            batch["audio_list"],
            sampling_rate=args.target_sr,
            return_tensors="pt",
            padding=True,                 
            return_attention_mask=True
        )
        # inputs: input_values [B, T], attention_mask [B, T]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Mixed precision for faster GPU inference
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            out = model(**inputs) # out.last_hidden_state → [B, T', H]
        
        hs = out.last_hidden_state
        B, Tprime, H = hs.shape

        attn_down = downsample_mask_linear(inputs["attention_mask"], Tprime)  # [B, T']
        emb = time_mean_pool(hs, attn_down)  # [B, H]

        if dim is None:
            dim = int(emb.size(-1)) # record embedding dimension (e.g., 768 or 1024)

        # Save each utterance embedding individually
        for vec, uid in zip(emb, batch["utter_id"]):
            torch.save(vec.detach().cpu(), out_dir / f"{uid}.pt")
            saved += 1
        
        pbar.set_postfix({"saved": f"{saved}/{total}", "dim": dim})

    meta = {
        "model": args.model_name,
        "dim": dim,
        "sr": args.target_sr,
        "max_seconds": args.max_seconds
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {saved}/{total} audio embeddings to {out_dir}")

if __name__ == "__main__":
    main()
