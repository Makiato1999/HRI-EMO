#!/usr/bin/env python3
import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    Compute mean over sequence dimension with optional padding mask.

    x:    [B, L, d]
    mask: [B, L] with True for PAD positions (to be ignored),
          or None if no padding.

    Returns:
        pooled: [B, d]
    """
    if mask is None:
        # Simple mean over L
        return x.mean(dim=1)

    # mask: True = pad → convert to 0 for valid, 1 for pad
    # we want weight 1 for valid, 0 for pad
    valid = (~mask).float()  # [B, L]

    # avoid division by zero
    denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]

    # [B, L, d] * [B, L, 1]
    weighted = x * valid.unsqueeze(-1)

    pooled = weighted.sum(dim=1) / denom  # [B, d]
    return pooled


class BetaGate(nn.Module):
    """
    Adaptive β-gating module for modality-wise fusion.

    - Takes aligned audio/text representations (can be [B,1,d] or [B,L,d])
    - Optionally uses padding masks
    - Learns a scalar β in [0,1] for each sample:
          h_fusion = β * h_a + (1-β) * h_t

    Design:
      - First, masked-mean pool each modality → h_a_pool, h_t_pool ∈ [B,d]
      - Build rich gate input by concatenating:
            [h_a_pool, h_t_pool, |h_a_pool - h_t_pool|, h_a_pool * h_t_pool]
      - Feed through small MLP + sigmoid → β ∈ [B,1]
      - Broadcast β back to match h_a/h_t shape, then fuse.
    """

    def __init__(self, d_model=768, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_t: torch.Tensor | None = None,
    ):
        """
        Args:
            h_a:    [B, L_a, d]
            h_t:    [B, L_t, d]
            mask_a: [B, L_a] bool or None (True = pad to ignore)
            mask_t: [B, L_t] bool or None

        Returns:
            h_fusion: [B, L_out, d]  (same shape as broadcasted inputs)
            beta:     [B, 1]        (scalar gate per sample, before broadcast)
        """
        B, _, d = h_a.shape

        # 1) pooled representations (works for utter & seq)
        h_a_pool = masked_mean(h_a, mask_a)  # [B, d]
        h_t_pool = masked_mean(h_t, mask_t)  # [B, d]

        # 2) build gate input
        diff = torch.abs(h_a_pool - h_t_pool)
        prod = h_a_pool * h_t_pool
        gate_input = torch.cat([h_a_pool, h_t_pool, diff, prod], dim=-1)  # [B, 4d]

        # 3) MLP + sigmoid -> β in [0,1]
        beta = torch.sigmoid(self.mlp(gate_input))  # [B, 1]

        # 4) broadcast β to match sequence shapes
        #    We want h_fusion same length as each modality.
        #    For simplicity, require h_a and h_t already aligned / same L,
        #    or choose one (e.g., text length) as reference.
        #    For now: broadcast to h_t's length if shapes differ.
        if h_a.size(1) == h_t.size(1):
            L = h_a.size(1)
        else:
            # simple choice: use text length as fusion length
            L = h_t.size(1)

        beta_broadcast = beta.view(B, 1, 1).expand(B, L, 1)  # [B, L, 1]

        # align lengths if needed
        if h_a.size(1) != L:
            h_a = h_a[:, :L, :]
        if h_t.size(1) != L:
            h_t = h_t[:, :L, :]

        # 5) convex combination
        h_fusion = beta_broadcast * h_a + (1.0 - beta_broadcast) * h_t  # [B, L, d]

        return h_fusion, beta
