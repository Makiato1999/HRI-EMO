import torch
from pathlib import Path

pt = torch.load("features/seq_level/text/Ses01F_impro01_F005.pt")
h = pt["hidden"]           # [L, d]
m = pt["attention_mask"]   # [L]  (1=valid,0=pad)

print(h.shape, h.abs().mean(), h.abs().max())
print(m[:50], m.sum())     # 看看是不是有正常的 1/0 分布
