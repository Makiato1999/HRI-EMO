#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
from tqdm import tqdm
import json

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/iemocap_index_splits.csv")
    ap.add_argument("--audio_root", default="data/wavs")
    ap.add_argument("--model_name", default="microsoft/wavlm-base-plus")
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--max_seconds", type=float, default=10.0,
                    help="truncate/pad raw wave to this length")
    ap.add_argument("--out_dir", default="features/seq_level/audio")
    return ap.parse_args()


def downsample_mask_linear(mask_B_L: torch.Tensor, Tprime: int) -> torch.Tensor:
    B, L = mask_B_L.shape
    idx = torch.linspace(0, L - 1, steps=Tprime, device=mask_B_L.device)
    idx = idx.round().long().clamp_(0, L - 1)
    return mask_B_L[:, idx]  # [B, T']


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    path_col = "audio_path"

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    max_len_samples = int(args.target_sr * args.max_seconds)

    dim = None
    saved = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio seq-level"):
        uid = row["utter_id"]
        wav_path = Path(args.audio_root) / row[path_col]

        if not wav_path.is_file():
            continue

        # Load wav
        wav, sr = torchaudio.load(str(wav_path)) # [C, T]
        wav = wav.mean(0, keepdim=True)          # [1, T] mono

        # Resample
        if sr != args.target_sr:
            wav = torchaudio.functional.resample(wav, sr, args.target_sr)

        wav = wav.squeeze(0) # [T]

        # Normalize [-1,1]
        mx = float(wav.abs().max())
        if mx > 0:
            wav = wav / mx

        # Truncate / pad to fixed length
        if wav.numel() > max_len_samples:
            wav = wav[:max_len_samples]
        else:
            pad = max_len_samples - wav.numel()
            if pad > 0:
                wav = torch.nn.functional.pad(wav, (0, pad))

        # Convert to numpy for feature_extractor（list[np.array]）
        wav_np = wav.detach().cpu().numpy().astype(np.float32)

        # Invoke feature_extractor build input_values + attention_mask
        inputs = feature_extractor(
            [wav_np], # batch size = 1
            sampling_rate=args.target_sr,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        # inputs["input_values"]: [1, T]
        # inputs["attention_mask"]: [1, T]

        # WavLM
        out = model(**inputs)
        hs = out.last_hidden_state # [1, T', H]
        B, Tprime, H = hs.shape

        # Downsample mask: [1, T] → [1, T']
        attn_down = downsample_mask_linear(inputs["attention_mask"], Tprime) # [1, T']
        hs = hs.squeeze(0).cpu()  # [T', H]
        attn_down = attn_down.squeeze(0).cpu().to(torch.long) # [T']

        if dim is None:
            dim = H

        torch.save(
            {
                "hidden": hs, # [T', H]
                "attention_mask": attn_down, # [T']
            },
            out_dir / f"{uid}.pt",
        )
        saved += 1

    meta = {
        "model": args.model_name,
        "hidden_dim": dim,
        "target_sr": args.target_sr,
        "max_seconds": args.max_seconds,
        "note": "seq-level WavLM features: hidden[T',H] + attention_mask[T'] (downsampled)",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {saved} audio seq-level embeddings to {out_dir}")


if __name__ == "__main__":
    main()
