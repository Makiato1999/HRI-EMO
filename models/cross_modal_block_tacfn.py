#!/usr/bin/env python3
import torch
import torch.nn as nn


class CrossModalBlock(nn.Module):
    """
    TACFN-inspired cross-modal block with:
      1) Intra-modal self-attention (redundancy reduction)
      2) Cross-modal attention in both directions
      3) FFN + residual + LayerNorm

    Supports:
      - Utterance-level:  [B, 1, d]
      - Sequence-level:   [B, L, d]
      - Optional key_padding_mask with True = PAD
    """

    def __init__(self, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # --- Intra-modal self-attention (per modality) ---
        self.self_attn_a = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_t = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.self_norm_a = nn.LayerNorm(d_model)
        self.self_norm_t = nn.LayerNorm(d_model)

        # --- Cross-modal attention (audio ↔ text) ---
        self.attn_a2t = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_t2a = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # FFN for each modality after cross-modal attention
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

        # Norms for cross-modal residual paths
        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)
        self.norm_t1 = nn.LayerNorm(d_model)
        self.norm_t2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_a: torch.Tensor,                  # [B, L_a, d]
        h_t: torch.Tensor,                  # [B, L_t, d]
        mask_a: torch.Tensor | None = None, # [B, L_a], True = PAD
        mask_t: torch.Tensor | None = None, # [B, L_t]
        return_attention: bool = False  # <--- 新增参数
    ):
        attn_maps = {} # 用于存储注意力权重

        # 1) Intra-modal self-attention
        # Audio self-attention
        a_sa, w_a_sa = self.self_attn_a(
            query=h_a,
            key=h_a,
            value=h_a,
            key_padding_mask=mask_a,
            need_weights=return_attention, # <--- 动态开关
        )
        h_a_self = self.self_norm_a(h_a + self.dropout(a_sa))
        if return_attention: attn_maps['audio_self'] = w_a_sa

        # Text self-attention
        t_sa, w_t_sa = self.self_attn_t(
            query=h_t,
            key=h_t,
            value=h_t,
            key_padding_mask=mask_t,
            need_weights=return_attention, # <--- 动态开关
        )
        h_t_self = self.self_norm_t(h_t + self.dropout(t_sa))
        if return_attention: attn_maps['text_self'] = w_t_sa

        # 2) Cross-modal attention: audio attends to text
        # Audio query → text key/value (Audio关注Text哪里)
        # 这就是 Cross-Modal Alignment
        a2t, w_a2t = self.attn_a2t(
            query=h_a_self,
            key=h_t_self,
            value=h_t_self,
            key_padding_mask=mask_t,
            need_weights=return_attention,
        )
        h_a_cm = self.norm_a1(h_a_self + self.dropout(a2t))
        h_a_cm = self.norm_a2(h_a_cm + self.dropout(self.ffn_a(h_a_cm)))
        if return_attention: attn_maps['audio_queries_text'] = w_a2t

        # 3) Cross-modal attention: text attends to audio
        # Text query -> Audio key/value (Text关注Audio哪里)
        t2a, w_t2a = self.attn_t2a(
            query=h_t_self,
            key=h_a_self,
            value=h_a_self,
            key_padding_mask=mask_a,
            need_weights=return_attention,
        )
        h_t_cm = self.norm_t1(h_t_self + self.dropout(t2a))
        h_t_cm = self.norm_t2(h_t_cm + self.dropout(self.ffn_t(h_t_cm)))
        if return_attention: attn_maps['text_queries_audio'] = w_t2a

        if return_attention:
            return h_a_cm, h_t_cm, attn_maps
        else:
            return h_a_cm, h_t_cm

        return h_a_cm, h_t_cm


class CrossModalTransformer(nn.Module):
    """
    Stacks multiple TACFN-inspired cross-modal blocks.

    Works for both:
      - Utter-level: pass [B,1,d]
      - Seq-level:   pass [B,L,d] (+ masks)
    """

    def __init__(self, num_layers=2, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_t: torch.Tensor | None = None,
        return_attention: bool = False # <--- 传递给 Block
    ):
        all_layers_attn = [] # 存储每一层的注意力

        for i, layer in enumerate(self.layers):
            if return_attention:
                h_a, h_t, attn_maps = layer(h_a, h_t, mask_a, mask_t, return_attention=True)
                all_layers_attn.append(attn_maps)
            else:
                h_a, h_t = layer(h_a, h_t, mask_a, mask_t, return_attention=False)

        if return_attention:
            return h_a, h_t, all_layers_attn
        else:
            return h_a, h_t
