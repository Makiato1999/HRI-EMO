import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from pathlib import Path
from tqdm import tqdm

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

out_dir = Path("features/seq_level/text")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/iemocap_index_splits.csv") 

for _, row in tqdm(df.iterrows(), total=len(df)):
    uid = row["utt_id"]
    text = row["transcript"]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**enc)
        # [1, L, d]
        hidden = outputs.last_hidden_state

    torch.save(
        {
            "hidden": hidden.squeeze(0),          # [L, d]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [L]
        },
        out_dir / f"{uid}.pt"
    )
