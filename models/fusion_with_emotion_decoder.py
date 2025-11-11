#!/usr/bin/env python3
import torch
import torch.nn as nn

from .cross_modal_block_tacfn import CrossModalTransformer
from .beta_gate_tacfn import BetaGate


class FusionWithEmotionDecoder(nn.Module):
    """
    Full model:
      Cross-modal Transformer + Vector-wise β-Gate + Emotion-Level Decoder.

    Used for:
      - Sequence-level multimodal inputs
      - Single-label or multi-label emotion prediction
      - Emotion-level interpretability via learnable queries
    """

    def __init__(
        self,
        d_model: int = 768,
        num_emotions: int = 4,
        n_heads: int = 8,
        num_layers_fusion: int = 2,
        num_layers_decoder: int = 2,
        beta_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        from .emotion_decoder import EmotionDecoder

        # 1) Cross-modal encoder (TACFN-style)
        self.cross_modal = CrossModalTransformer(
            num_layers=num_layers_fusion,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # 2) Vector-wise β-gating
        self.beta_gate = BetaGate(
            d_model=d_model,
            hidden_dim=beta_hidden,
        )

        # 3) Emotion-level Transformer decoder
        self.emotion_decoder = EmotionDecoder(
            d_model=d_model,
            num_emotions=num_emotions,
            n_heads=n_heads,
            num_layers=num_layers_decoder,
            dropout=dropout,
            use_output_layer=True,  # directly output per-emotion logits
        )

    # -----------------------------
    # helpers
    # -----------------------------
    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure input has shape [B, L, d].
        If [B, d], expand to [B, 1, d].
        """
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Expected 2D or 3D tensor, got {x.shape}")

    def _build_fused_mask(
        self,
        mask_a: torch.Tensor | None,
        mask_t: torch.Tensor | None,
        L_fused: int,
    ) -> torch.Tensor | None:
        """
        Build key_padding_mask for fused sequence.

        Conventions:
          - Input masks: True = PAD, False = valid.
          - Fused sequence length = L_fused (same as h_fusion.size(1)).
          - We align both masks to L_fused (truncate or slice).
          - Conservative rule: a position is PAD if any modality marks it PAD.
        """
        if mask_a is None and mask_t is None:
            return None

        ma = None
        mt = None

        if mask_a is not None:
            if mask_a.size(1) < L_fused:
                # pad with PAD=True if shorter (rare)
                pad_len = L_fused - mask_a.size(1)
                pad = torch.ones(mask_a.size(0), pad_len, dtype=torch.bool, device=mask_a.device)
                ma = torch.cat([mask_a, pad], dim=1)
            else:
                ma = mask_a[:, :L_fused]

        if mask_t is not None:
            if mask_t.size(1) < L_fused:
                pad_len = L_fused - mask_t.size(1)
                pad = torch.ones(mask_t.size(0), pad_len, dtype=torch.bool, device=mask_t.device)
                mt = torch.cat([mask_t, pad], dim=1)
            else:
                mt = mask_t[:, :L_fused]

        if ma is None:
            return mt
        if mt is None:
            return ma

        # True = PAD if any modality is PAD at that position
        return ma | mt

    # -----------------------------
    # forward
    # -----------------------------
    def forward(
        self,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_t: torch.Tensor | None = None,
    ):
        """
        Args:
            h_a: [B, d] or [B, L_a, d] - audio features
            h_t: [B, d] or [B, L_t, d] - text features
            mask_a: [B, L_a] bool or None (True = PAD)
            mask_t: [B, L_t] bool or None (True = PAD)

        Returns:
            logits: [B, num_emotions]
            beta:   [B, d] or [B, 1] depending on BetaGate design
            z:      [B, num_emotions, d] emotion-specific embeddings
        """
        # 1) normalize shapes
        h_a = self._ensure_3d(h_a)
        h_t = self._ensure_3d(h_t)

        # 2) cross-modal semantic alignment (can handle different lengths)
        h_a_tilde, h_t_tilde = self.cross_modal(h_a, h_t, mask_a, mask_t)
        # shapes: [B, L_a, d], [B, L_t, d]

        # 3) β-gated fusion (this will internally align L)
        h_fusion, beta = self.beta_gate(h_a_tilde, h_t_tilde, mask_a, mask_t)
        # h_fusion: [B, L_fused, d]

        L_fused = h_fusion.size(1)

        # 4) build fused key_padding_mask aligned to h_fusion
        fused_mask = self._build_fused_mask(mask_a, mask_t, L_fused)
        # fused_mask: [B, L_fused] or None

        # 5) emotion-level decoding
        z, logits = self.emotion_decoder(
            memory=h_fusion,
            memory_key_padding_mask=fused_mask,
        )
        # z: [B, num_emotions, d]
        # logits: [B, num_emotions]

        return logits, beta, z
