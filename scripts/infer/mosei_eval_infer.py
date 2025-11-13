#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any   # [MOD] 加 Dict, Any
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder

EMO_COLS = ["emo_happy","emo_sad","emo_anger","emo_fear","emo_disgust","emo_surprise"]

# ---------------- utils ----------------
def _load_array_or_hidden(p: Path):
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    if p.suffix in (".pt", ".pth"):
        obj = torch.load(p, map_location="cpu")
        # 支持两种写法：tensor 或 dict{"hidden": ...}
        if isinstance(obj, dict) and "hidden" in obj:
            arr = obj["hidden"]
            if isinstance(arr, torch.Tensor):
                return arr.numpy()
            return np.array(arr)
        if isinstance(obj, torch.Tensor):
            return obj.numpy()
        return np.array(obj)
    raise ValueError(f"Unsupported feature file ext: {p.suffix}")

def _resolve_feat_dir(root: Optional[str], audio_dir: Optional[str], text_dir: Optional[str]) -> Tuple[Path, Path]:
    if audio_dir and text_dir:
        return Path(audio_dir), Path(text_dir)
    if not root:
        raise ValueError("Either --features_root or (--audio_dir and --text_dir) must be provided.")
    r = Path(root)
    for a, t in [(r/"audio", r/"text"), (r/"seq_level"/"audio", r/"seq_level"/"text")]:
        if a.exists() and t.exists():
            return a, t
    raise FileNotFoundError(f"Cannot find audio/text dirs under {root}")

def _read_hidden_dims_from_meta(audio_dir: Path, text_dir: Path) -> Tuple[int, int]:
    audio_meta = json.loads((audio_dir / "meta.json").read_text())
    text_meta  = json.loads((text_dir  / "meta.json").read_text())
    d_audio = int(audio_meta["hidden_dim"])
    d_text  = int(text_meta["hidden_dim"])
    print(f"[✓] Loaded hidden dims from meta.json -> audio={d_audio}, text={d_text}")
    return d_audio, d_text

def crop_center(x: np.ndarray, max_len: int) -> np.ndarray:
    if max_len is None or max_len <= 0 or x.shape[0] <= max_len:
        return x
    s = (x.shape[0] - max_len) // 2
    return x[s:s+max_len]

# ---------------- Dataset ----------------
class SeqDataset(Dataset):
    """
    读 index CSV；优先用 'utter_id'，否则回退 'uid'。
    若有 split 列，则根据 split_name 过滤。
    从 audio_dir/text_dir 读取 {uid}.npy|pt|pth，并在 __getitem__ 内做中心截断。
    """
    def __init__(
        self,
        index_csv: str,
        audio_dir: Path,
        text_dir: Path,
        split_name: Optional[str],
        max_len_audio: int,
        max_len_text: int,
    ):
        df = pd.read_csv(index_csv)

        # 选择 uid 列
        if "utter_id" in df.columns:
            uid_col = "utter_id"
        elif "uid" in df.columns:
            uid_col = "uid"
        else:
            raise ValueError("CSV must contain 'utter_id' or 'uid' column")

        if split_name and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == split_name.lower()].reset_index(drop=True)

        self.uids: List[str] = df[uid_col].astype(str).tolist()

        # 标签（如果存在情绪列就带上；否则全 0）
        self.labels = np.zeros((len(self.uids), len(EMO_COLS)), dtype=np.float32)
        if any(c in df.columns for c in EMO_COLS):
            for k, c in enumerate(EMO_COLS):
                if c in df.columns:
                    self.labels[:, k] = df[c].astype(float).values

        self.audio_dir = Path(audio_dir)
        self.text_dir  = Path(text_dir)
        self.max_len_audio = max_len_audio
        self.max_len_text  = max_len_text

    def __len__(self): return len(self.uids)

    def _find(self, base: Path, uid: str) -> Path:
        for ext in (".npy", ".pt", ".pth"):
            p = base / f"{uid}{ext}"
            if p.exists(): return p
        raise FileNotFoundError(f"Feature for {uid} not found in {base}")

    def __getitem__(self, i: int):
        uid = self.uids[i]
        a = _load_array_or_hidden(self._find(self.audio_dir, uid))  # [L,Da] or [Da]
        t = _load_array_or_hidden(self._find(self.text_dir,  uid))  # [L,Dt] or [Dt]

        if a.ndim == 1: a = a[None, :]
        if t.ndim == 1: t = t[None, :]

        # 中心截断（与训练保持一致，避免注意力爆显存）
        a = crop_center(a, self.max_len_audio)
        t = crop_center(t, self.max_len_text)

        y = self.labels[i]
        return torch.tensor(a, dtype=torch.float32), torch.tensor(t, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ---------------- Collate ----------------
def collate_seq_batch(batch):
    As, Ts, Ys = zip(*batch)
    B = len(batch)
    La = max(x.shape[0] for x in As)
    Lt = max(x.shape[0] for x in Ts)
    Da = As[0].shape[1]
    Dt = Ts[0].shape[1]

    h_a = torch.zeros(B, La, Da, dtype=torch.float32)
    h_t = torch.zeros(B, Lt, Dt, dtype=torch.float32)
    m_a = torch.ones(B, La, dtype=torch.bool)
    m_t = torch.ones(B, Lt, dtype=torch.bool)

    for i, (a, t) in enumerate(zip(As, Ts)):
        la, lt = a.shape[0], t.shape[0]
        h_a[i, :la] = a; m_a[i, :la] = False
        h_t[i, :lt] = t; m_t[i, :lt] = False

    y = torch.stack(Ys, dim=0)
    return h_a, m_a, h_t, m_t, y

# ============== [MOD] hooks: 捕获 Emotion Decoder cross-attn ==============
def register_emodecoder_attn_hooks(model: torch.nn.Module, store: Dict[str, Any], max_layers: int = 2):
    """
    在 Emotion Decoder 的 cross-attn (encoder-decoder attention) 上注册 forward hooks，
    记录注意力矩阵。

    你的结构大致是：
      model.backbone.emotion_decoder.decoder.layers[i].multihead_attn

    我们只抓：
      - 模块类型是 MultiheadAttention
      - 名字里包含 "emotion_decoder.decoder.layers"
      - 名字里包含 ".multihead_attn"（排除 self_attn）
      - 只保留前 max_layers 个 layer
    """
    for name, module in model.named_modules():
        # 1) 只关心 MultiheadAttention
        if not isinstance(module, torch.nn.MultiheadAttention):
            continue

        # 2) 限定在 EmotionDecoder 的 decoder.layers 里
        if "emotion_decoder" not in name or "decoder.layers" not in name:
            continue

        # 3) 只抓 cross-attn：multihead_attn；跳过 self_attn
        if ".multihead_attn" not in name:
            continue

        # 4) 解析层编号，限制到前 max_layers 层
        layer_idx = None
        try:
            after = name.split("decoder.layers.", 1)[1]
            layer_idx = int(after.split(".", 1)[0])
        except Exception:
            pass

        if layer_idx is not None and layer_idx >= max_layers:
            continue

        def _hook(m, inp, out, _n=name):
            """
            MultiheadAttention forward 返回:
              - out[0]: attn_output
              - out[1]: attn_weights 或 None（当 need_weights=False 时）

            这里必须确认 out[1] 不为 None 再存。
            """
            if not (isinstance(out, tuple) and len(out) == 2):
                return

            attn_weights = out[1]
            if attn_weights is None:
                # 当前 PyTorch 默认 need_weights=False，会走到这里；先安全退出，避免报错。
                return

            store.setdefault("decoder_attn", {})[_n] = attn_weights.detach().cpu()

        module.register_forward_hook(_hook)

    return store
"""
attn_weights 很可能仍然是 None，因为 PyTorch 在 decoder 里调用 MultiheadAttention 时默认 need_weights=False。
也就是说：现在这版是“安全的”：不会再崩；如果将来你改 EmotionDecoder 的实现让它返回权重，这里的 hook 会自动开始工作。
但要真想拿到 非 None 的 decoder cross-attn heatmap，下一步我们得改 EmotionDecoder，手动循环 decoder.layers
在调用层时传 need_weights=True 或绕过 nn.TransformerDecoder 的默认逻辑。
"""
# ---------------- Inference ----------------
@torch.no_grad()
def run_split(
    model,
    ds,
    batch_size,
    device,
    out_dir,
    split_name,
    amp_dtype: Optional[torch.dtype],
    dump_beta: bool = False,          # [MOD] 是否导出 β
    dump_attn: bool = False,          # [MOD] 是否导出 Emotion Decoder cross-attn
    attn_max_samples: int = 64,       # [MOD] 最多抓多少样本的注意力（避免太大）
):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq_batch)
    probs, labels = [], []
    betas = []                        # [MOD] 收集 β

    # [MOD] 注意力容器
    attn_store: Dict[str, Any] = {}
    if dump_attn:
        register_emodecoder_attn_hooks(model, attn_store, max_layers=2)

    model.eval()
    use_amp = (device.type == "cuda") and (amp_dtype is not None)
    seen = 0                          # [MOD] 已处理样本数（控制 attn_max_samples）

    for h_a, m_a, h_t, m_t, y in dl:
        h_a = h_a.to(device); m_a = m_a.to(device)
        h_t = h_t.to(device); m_t = m_t.to(device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, beta, _ = model(h_a, h_t, m_a, m_t)
        else:
            logits, beta, _ = model(h_a, h_t, m_a, m_t)

        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())

        # [MOD] 收集 β（统一转成标量）
        if dump_beta and beta is not None:
            b = beta.detach().float().cpu()
            if b.ndim > 1:
                # 如果是 [B, d_model] 等，按维度取 mean
                for _ in range(1, b.ndim):
                    b = b.mean(dim=-1)
            betas.append(b.numpy())

        seen += y.size(0)
        # [MOD] 仅限制注意力采样的样本数（模型仍然跑完整个 dataset）
        # 如果你想在达到 attn_max_samples 后整段提前 break，可以把下面 if 打开：
        # if dump_attn and seen >= attn_max_samples:
        #     break

    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_dir}/{split_name}_y_prob.npy", probs)
    np.save(f"{out_dir}/{split_name}_y_true.npy", labels)
    print(f"[Saved] {split_name} -> {out_dir}/{split_name}_y_prob.npy, {out_dir}/{split_name}_y_true.npy")

    # [MOD] 保存 β
    if dump_beta and betas:
        beta_arr = np.concatenate(betas, axis=0)
        np.save(f"{out_dir}/{split_name}_beta_mean.npy", beta_arr)
        print(f"[Saved] {split_name} beta -> {out_dir}/{split_name}_beta_mean.npy")

    # [MOD] 保存 Emotion Decoder cross-attn（前 K 个样本）
    if dump_attn and "decoder_attn" in attn_store:
        save_path = Path(out_dir) / f"{split_name}_decoder_attn_first{attn_max_samples}.npz"
        to_save = {}
        for k, v in attn_store["decoder_attn"].items():
            # v 通常形状: [B, num_heads, num_emotions, seq_len]
            arr = v.numpy() if torch.is_tensor(v) else np.array(v)
            # 可选：只裁到前 attn_max_samples 个样本
            if arr.shape[0] > attn_max_samples:
                arr = arr[:attn_max_samples]
            to_save[k] = arr
        np.savez_compressed(save_path, **to_save)
        print(f"[Saved] {split_name} decoder-attn -> {save_path}")

def _amp_dtype_from_str(s: str) -> Optional[torch.dtype]:
    s = (s or "off").lower()
    if s == "bf16": return torch.bfloat16
    if s == "fp16": return torch.float16
    return None  # "off"

def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--index_csv_val", required=True)
    ap.add_argument("--index_csv_test", default=None)
    ap.add_argument("--features_root", default=None, help="根目录包含 audio|text 或 seq_level/audio|text")
    ap.add_argument("--audio_dir", default=None)
    ap.add_argument("--text_dir",  default=None)
    ap.add_argument("--batch_size", type=int, default=8)          # 推理默认更小，稳妥
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--max_len_audio", type=int, default=300)     # 与训练一致
    ap.add_argument("--max_len_text",  type=int, default=128)     # 与训练一致
    ap.add_argument("--amp_dtype",     type=str, default="bf16", choices=["bf16","fp16","off"])
    # 模型（会被 ckpt['args'] 覆盖）
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--num_layers_fusion", type=int, default=2)
    ap.add_argument("--num_layers_decoder", type=int, default=2)
    ap.add_argument("--beta_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    # [MOD] 额外导出控制项
    ap.add_argument("--dump_beta", action="store_true",
                    help="保存每个样本的 beta（mean）到 {split}_beta_mean.npy")
    ap.add_argument("--dump_attn", action="store_true",
                    help="保存 Emotion Decoder cross-attn 到 {split}_decoder_attn_firstK.npz")
    ap.add_argument("--attn_max_samples", type=int, default=64,
                    help="保存注意力时最多保留多少个样本（避免文件过大）")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 解析特征目录 & 输入维度
    a_dir, t_dir = _resolve_feat_dir(args.features_root, args.audio_dir, args.text_dir)
    d_audio, d_text = _read_hidden_dims_from_meta(a_dir, t_dir)

    # 先加载 ckpt，并用 ckpt["args"] 覆盖超参（与训练完全一致）
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "args" in ckpt:
        cfg = ckpt["args"]
        args.d_model = cfg.get("d_model", args.d_model)
        args.n_heads = cfg.get("n_heads", args.n_heads)
        args.num_layers_fusion = cfg.get("num_layers_fusion", args.num_layers_fusion)
        args.num_layers_decoder = cfg.get("num_layers_decoder", args.num_layers_decoder)
        args.beta_hidden = cfg.get("beta_hidden", args.beta_hidden)
        args.dropout = cfg.get("dropout", args.dropout)

    # 构建模型（用覆盖后的参数）
    model = MoseiFusionWithEmotionDecoder(
        d_audio=d_audio,
        d_text=d_text,
        d_model=args.d_model,
        num_emotions=len(EMO_COLS),
        n_heads=args.n_heads,
        num_layers_fusion=args.num_layers_fusion,
        num_layers_decoder=args.num_layers_decoder,
        beta_hidden=args.beta_hidden,
        dropout=args.dropout
    ).to(device)

    # 加载权重
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)

    amp_dtype = _amp_dtype_from_str(args.amp_dtype)

    # 推理（val）
    val_ds = SeqDataset(
        args.index_csv_val, a_dir, t_dir, split_name="val",
        max_len_audio=args.max_len_audio, max_len_text=args.max_len_text
    )
    run_split(
        model, val_ds, args.batch_size, device, args.out_dir, "val",
        amp_dtype,
        dump_beta=args.dump_beta,
        dump_attn=args.dump_attn,
        attn_max_samples=args.attn_max_samples,
    )

    # 推理（test，若提供）
    if args.index_csv_test:
        test_ds = SeqDataset(
            args.index_csv_test, a_dir, t_dir, split_name="test",
            max_len_audio=args.max_len_audio, max_len_text=args.max_len_text
        )
        run_split(
            model, test_ds, args.batch_size, device, args.out_dir, "test",
            amp_dtype,
            dump_beta=args.dump_beta,
            dump_attn=args.dump_attn,
            attn_max_samples=args.attn_max_samples,
        )

if __name__ == "__main__":
    main()
