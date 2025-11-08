# models/cross_beta_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def ensure_2d_seq(x: torch.Tensor) -> torch.Tensor:
    # x: [B,D] or [B,T,D] or [D]
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(1)  # [1,1,D]
    elif x.dim() == 2:
        x = x.unsqueeze(1)               # [B,1,D]
    return x  # [B,T,D]

class CrossModalBlock(nn.Module):
    """One TACFN-style cross-modal block with bidirectional MHA and residuals."""
    def __init__(self, d_in_a: int, d_in_t: int, d_model: int = 256,
                 n_heads: int = 4, ff_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.proj_a = nn.Linear(d_in_a, d_model)
        self.proj_t = nn.Linear(d_in_t, d_model)

        self.mha_a_q_t_kv = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)  # A<-T
        self.mha_t_q_a_kv = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)  # T<-A

        self.ln_a1 = nn.LayerNorm(d_model)
        self.ln_t1 = nn.LayerNorm(d_model)

        # Position-wise FFN for each stream
        self.ff_a = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.ff_t = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.ln_a2 = nn.LayerNorm(d_model)
        self.ln_t2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, a, t):
        # a,t: [B,D] or [B,T,D]
        a = ensure_2d_seq(a)
        t = ensure_2d_seq(t)

        a = self.proj_a(a)
        t = self.proj_t(t)

        # A queries, T provides K/V  -> align A to T
        a2, _ = self.mha_a_q_t_kv(query=a, key=t, value=t)  # [B,Ta,D]
        a = self.ln_a1(a + self.drop(a2))

        # T queries, A provides K/V  -> align T to A
        t2, _ = self.mha_t_q_a_kv(query=t, key=a, value=a)  # [B,Tt,D]
        t = self.ln_t1(t + self.drop(t2))

        # Position-wise FFN + residual
        a = self.ln_a2(a + self.drop(self.ff_a(a)))
        t = self.ln_t2(t + self.drop(self.ff_t(t)))

        return a, t  # [B,Ta,D], [B,Tt,D]

class CrossBetaFusionClassifier(nn.Module):
    """
    Bidirectional Cross-Attention + β-gating fusion + classifier.
    If inputs are [B,D], they are treated as length-1 sequences.
    """
    def __init__(self,
                 in_dim_audio: int = 768,
                 in_dim_text: int = 768,
                 d_model: int = 256,
                 n_heads: int = 4,
                 ff_dim: int = 512,
                 num_layers: int = 1,
                 num_classes: int = 6,
                 dropout: float = 0.2,
                 tau: float = 1.0):   # temperature for beta softmax
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossModalBlock(in_dim_audio if i == 0 else d_model,
                            in_dim_text if i == 0 else d_model,
                            d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
            for i in range(num_layers)
        ])
        self.tau = tau

        # β gating based on pooled cross-aligned features
        self.beta_net = nn.Sequential(
            nn.Linear(d_model * 2, ff_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 2)
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes)
        )

    @staticmethod
    def time_pool(x: torch.Tensor, mode: str = "mean"):
        # x: [B,T,D]
        if mode == "mean":
            return x.mean(1)
        elif mode == "max":
            return x.max(1).values
        else:
            return x[:, 0]  # first

    def forward(self, a, t):
        # a,t: [B,D] or [B,T,D] or [D]
        for blk in self.blocks:
            a, t = blk(a, t)  # [B,Ta,D], [B,Tt,D]

        ha = self.time_pool(a, "mean")  # [B,D]
        ht = self.time_pool(t, "mean")  # [B,D]

        beta_logits = self.beta_net(torch.cat([ha, ht], dim=-1)) / self.tau
        beta = F.softmax(beta_logits, dim=-1)  # [B,2]
        z = beta[:, :1] * ha + beta[:, 1:] * ht  # fused [B,D]

        logits = self.head(z)  # [B,C]
        return logits, beta  # beta for analysis
