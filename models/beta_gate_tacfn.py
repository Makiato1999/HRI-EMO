#!/usr/bin/env python3
import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    Mean pooling over sequence dimension with optional padding mask.

    Args:
        x:    [B, L, d]
        mask: [B, L] with True = PAD (ignored), or None.

    Returns:
        pooled: [B, d]
    """
    if mask is None:
        return x.mean(dim=1)

    # True=PAD -> 0 ; False=valid -> 1
    valid = (~mask).float()  # [B, L]
    denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    weighted = x * valid.unsqueeze(-1)
    return weighted.sum(dim=1) / denom


class BetaGate(nn.Module):
    """
    TACFN-inspired adaptive fusion gate (vector-wise).

    Instead of a single scalar β per sample, we predict a per-dimension
    weight vector w ∈ [0,1]^d:

        w = σ(MLP([a_pool, t_pool, |a-t|, a⊙t]))
        h_fusion = w ⊙ h_a_norm + (1-w) ⊙ h_t_norm

    This:
      - makes fusion more fine-grained (closer to TACFN weight vector)
      - reduces hard "all-or-nothing" behavior
      - still returns a scalar beta (mean of w) for logging.

    Inputs:
        h_a:    [B, L_a, d]
        h_t:    [B, L_t, d]
        mask_a: [B, L_a] (True = PAD) or None
        mask_t: [B, L_t] (True = PAD) or None

    Outputs:
        h_fusion: [B, L_f, d] fused sequence
        beta:     [B, 1]      mean gate value for monitoring
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.d_model = d_model

        # modality-wise normalization
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)

        # gating network: 4d -> d (vector gate)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_t: torch.Tensor | None = None,
    ):
        B = h_a.size(0)
        d = self.d_model

        # 1) normalize sequences
        h_a_n = self.norm_a(h_a)  # [B, L_a, d]
        h_t_n = self.norm_t(h_t)  # [B, L_t, d]

        # 2) pooled (normalized) representations
        a_pool = masked_mean(h_a_n, mask_a)  # [B, d]
        t_pool = masked_mean(h_t_n, mask_t)  # [B, d]

        # 3) build gate input
        diff = torch.abs(a_pool - t_pool)
        prod = a_pool * t_pool
        gate_input = torch.cat([a_pool, t_pool, diff, prod], dim=-1)  # [B, 4d]

        # 4) vector gate w ∈ [0,1]^d
        w = torch.sigmoid(self.mlp(gate_input))  # [B, d]

        # scalar beta for logging / interpretability
        beta_scalar = w.mean(dim=-1, keepdim=True)  # [B, 1]

        # 5) choose fusion length
        L_a = h_a_n.size(1)
        L_t = h_t_n.size(1)
        if L_a == L_t:
            L = L_a
        else:
            # simple & consistent: align to text length
            L = L_t

        # align sequences
        if L_a != L:
            h_a_n = h_a_n[:, :L, :]
        if L_t != L:
            h_t_n = h_t_n[:, :L, :]

        # 6) broadcast gate to sequence
        w_b = w.view(B, 1, d).expand(B, L, d)  # [B, L, d]

        # 7) per-dimension fusion on normalized features
        h_fusion = w_b * h_a_n + (1.0 - w_b) * h_t_n  # [B, L, d]

        return h_fusion, beta_scalar
