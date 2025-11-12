#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# === 根据你的工程实际改这三处 ===
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder
from dataset.mosei_seq_dataset import MoseiSeqDataset   # 你的数据集类
from dataset.collate import collate_seq_batch           # 你的 collate

@torch.no_grad()
def run_split(model, ds, batch_size, device, out_dir, split_name):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq_batch)
    probs, labels = [], []
    model.eval()
    for h_a, m_a, h_t, m_t, y in dl:
        h_a = h_a.to(device); m_a = m_a.to(device)
        h_t = h_t.to(device); m_t = m_t.to(device)
        y   = y.to(device)
        logits, beta, _ = model(h_a, h_t, m_a, m_t)  # 你的 forward 输出
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        labels.append(y.cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_dir}/{split_name}_y_prob.npy", probs)
    np.save(f"{out_dir}/{split_name}_y_true.npy", labels)
    print(f"[Saved] {split_name} y_prob/y_true -> {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best_mosei_fusion_decoder.pt")
    ap.add_argument("--index_csv_val", required=True)
    ap.add_argument("--index_csv_test", default=None)
    ap.add_argument("--features_root", required=True)  # 你放特征的根目录
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # === 构建模型 & 加载权重 ===
    model = MoseiFusionWithEmotionDecoder(
        d_model=768, num_emotions=6, n_heads=8,
        num_layers_fusion=2, num_layers_decoder=2, beta_hidden=256, dropout=0.1
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    # === 验证集推理 ===
    val_ds = MoseiSeqDataset(args.index_csv_val, args.features_root, split="val")
    run_split(model, val_ds, args.batch_size, device, args.out_dir, "val")

    # === 测试集（可选） ===
    if args.index_csv_test:
        test_ds = MoseiSeqDataset(args.index_csv_test, args.features_root, split="test")
        run_split(model, test_ds, args.batch_size, device, args.out_dir, "test")

if __name__ == "__main__":
    main()
