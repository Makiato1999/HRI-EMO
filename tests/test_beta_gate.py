import torch
from models.cross_modal_block import CrossModalTransformer
from models.beta_gate import BetaGate

B, d = 32, 768

# 假设这是你从 pooled 特征加载来的 [B, d]
h_a_utt = torch.randn(B, d)
h_t_utt = torch.randn(B, d)

# 变成 [B,1,d] 喂给 CrossModal
h_a = h_a_utt.unsqueeze(1)
h_t = h_t_utt.unsqueeze(1)

cross = CrossModalTransformer(num_layers=2, d_model=d, n_heads=8)
beta_gate = BetaGate(d_model=d, hidden_dim=256)

# 1) cross-modal semantic alignment
h_a_tilde, h_t_tilde = cross(h_a, h_t)  # [B,1,d], [B,1,d]

# 2) β-gating fusion
h_fusion, beta = beta_gate(h_a_tilde, h_t_tilde)  # [B,1,d], [B,1]

print(h_fusion.shape, beta.shape)
# → torch.Size([32, 1, 768]) torch.Size([32, 1])
