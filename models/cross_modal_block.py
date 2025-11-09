#!/usr/bin/env python3
import torch
import torch.nn as nn

class CrossModalBlock(nn.Module):
    """
    Bidirectional cross-modal Transformer block:
      - audio→text attention
      - text→audio attention
    Works for both utterance-level ([B,1,d]) and sequence-level ([B,L,d]) inputs.
    """

    def __init__(self, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_a2t = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.attn_t2a = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )

        # Feed-Forward sub-layers
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        # LayerNorms
        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)
        self.norm_t1 = nn.LayerNorm(d_model)
        self.norm_t2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_a, h_t,
        mask_a: torch.Tensor | None = None,
        mask_t: torch.Tensor | None = None,
    ):
        """
        h_a: [B, L_a, d]
        h_t: [B, L_t, d]
        mask_a, mask_t: optional padding masks (True = ignore)
        """
        # --- audio attends to text ---
        a2t, _ = self.attn_a2t(
            query=h_a, key=h_t, value=h_t,
            key_padding_mask=mask_t
        )
        h_a_tilde = self.norm_a1(h_a + self.dropout(a2t))
        h_a_tilde = self.norm_a2(h_a_tilde + self.dropout(self.ffn_a(h_a_tilde)))

        # --- text attends to audio ---
        t2a, _ = self.attn_t2a(
            query=h_t, key=h_a, value=h_a,
            key_padding_mask=mask_a
        )
        h_t_tilde = self.norm_t1(h_t + self.dropout(t2a))
        h_t_tilde = self.norm_t2(h_t_tilde + self.dropout(self.ffn_t(h_t_tilde)))

        return h_a_tilde, h_t_tilde


class CrossModalTransformer(nn.Module):
    """
    Stack multiple CrossModalBlocks.
    """

    def __init__(self, num_layers=2, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        h_a, h_t,
        mask_a: torch.Tensor | None = None,
        mask_t: torch.Tensor | None = None,
    ):
        for layer in self.layers:
            h_a, h_t = layer(h_a, h_t, mask_a, mask_t)
        return h_a, h_t

"""
from models.cross_modal_block import CrossModalTransformer
import torch

# utter-level example
h_a = torch.randn(32, 1, 768)
h_t = torch.randn(32, 1, 768)
model = CrossModalTransformer(num_layers=2)
h_a_tilde, h_t_tilde = model(h_a, h_t)
print(h_a_tilde.shape)  # [32,1,768]

# sequence-level example (未来直接替换)
h_a = torch.randn(8, 400, 768)
h_t = torch.randn(8, 128, 768)
mask_a = torch.zeros(8, 400, dtype=torch.bool)
mask_t = torch.zeros(8, 128, dtype=torch.bool)
h_a_tilde, h_t_tilde = model(h_a, h_t, mask_a, mask_t)
print(h_a_tilde.shape, h_t_tilde.shape)
"""