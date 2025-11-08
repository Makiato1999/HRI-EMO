import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel
from pathlib import Path
import pandas as pd
from tqdm import tqdm

MODEL_NAME = "microsoft/wavlm-base-plus"
MAX_SECONDS = 10.0
TARGET_SR = 16000

feat_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

out_dir = Path("features/seq_level/audio")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/iemocap_index_splits.csv")

for _, row in tqdm(df.iterrows(), total=len(df)):
    uid = row["utt_id"]
    wav_path = row["wav_path"]

    wav, sr = torchaudio.load(wav_path)
    wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav.mean(dim=0)  # mono

    max_len = int(MAX_SECONDS * TARGET_SR)
    if len(wav) > max_len:
        wav = wav[:max_len]
    else:
        pad = max_len - len(wav)
        wav = torch.nn.functional.pad(wav, (0, pad))

    inputs = feat_extractor(
        wav,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # hidden_states: [1, T', d]
        hidden = outputs.last_hidden_state

    attn_mask = inputs["attention_mask"].squeeze(0)  # [T']

    torch.save(
        {
            "hidden": hidden.squeeze(0),   # [T', d]
            "attention_mask": attn_mask,
        },
        out_dir / f"{uid}.pt"
    )
