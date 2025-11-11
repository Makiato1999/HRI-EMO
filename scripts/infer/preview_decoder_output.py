#!/usr/bin/env python3
import torch, json
from pathlib import Path
from models.fusion_with_emotion_decoder import FusionWithEmotionDecoder

# -------------------------------------------------
# 1. Load checkpoint
ckpt_path = Path("runs/fusion_seq_level_decoder_iemocap/best_fusion_seq_decoder.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
args = ckpt["args"]
label2id = ckpt["label2id"]
id2label = {v: k for k, v in label2id.items()}

# -------------------------------------------------
# 2. Rebuild model
model = FusionWithEmotionDecoder(
    d_model=args["d_model"],
    num_emotions=len(label2id),
    n_heads=args["n_heads"],
    num_layers_fusion=args["num_layers_fusion"],
    num_layers_decoder=args["num_layers_decoder"],
    beta_hidden=args["beta_hidden"],
    dropout=args["dropout"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# -------------------------------------------------
# 3. Prepare one test sample
# 假设你已经有 seq-level 特征文件: features/seq_level/audio/Ses01F_impro01_F000.pt
uid = "Ses01F_impro01_F000"
audio_feat = torch.load(f"features/seq_level/audio/{uid}.pt", map_location="cpu")
text_feat  = torch.load(f"features/seq_level/text/{uid}.pt", map_location="cpu")

h_a, m_a = audio_feat["hidden"].unsqueeze(0), (audio_feat["attention_mask"] == 0).unsqueeze(0)
h_t, m_t = text_feat["hidden"].unsqueeze(0), (text_feat["attention_mask"] == 0).unsqueeze(0)

# -------------------------------------------------
# 4. Run model inference
with torch.no_grad():
    logits, beta, z = model(h_a, h_t, m_a, m_t)
    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    pred_label = id2label[pred_idx]

print(f"Predicted emotion: {pred_label}")
print(f"Class probabilities: {probs.tolist()}")
print(f"Mean β: {beta.mean().item():.3f}")
