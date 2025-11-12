#!/usr/bin/env python3
import torch
import torch.nn as nn

from .fusion_with_emotion_decoder import FusionWithEmotionDecoder


class MoseiFusionWithEmotionDecoder(nn.Module):
    """
    Wrapper for MOSEI:
      - Projects text/audio features (different dims) to shared d_model
      - Then applies TACFN-style fusion + emotion-level decoder.

    Inputs:
        h_a: [B, L_a, d_audio]
        h_t: [B, L_t, d_text]
        mask_a: [B, L_a] bool (True = PAD) or None
        mask_t: [B, L_t] bool (True = PAD) or None

    Outputs:
        logits: [B, num_emotions]
        beta:   [B, 1] or [B, d_model] (depending on your BetaGate impl)
        fused:  [B, d_model] pooled fusion representation
    """

    def __init__(
        self,
        d_audio: int,
        d_text: int,
        d_model: int = 384,
        num_emotions: int = 6,
        n_heads: int = 6,
        num_layers_fusion: int = 3,
        num_layers_decoder: int = 3,
        beta_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Project raw MOSEI features to shared space
        self.audio_proj = nn.Linear(d_audio, d_model)
        self.text_proj = nn.Linear(d_text, d_model)

        # Reuse your existing backbone (TACFN-style cross-modal + β + decoder)
        self.backbone = FusionWithEmotionDecoder(
            d_model=d_model,
            num_emotions=num_emotions,
            n_heads=n_heads,
            num_layers_fusion=num_layers_fusion,
            num_layers_decoder=num_layers_decoder,
            beta_hidden=beta_hidden,
            dropout=dropout,
        )

    def forward(self, h_a, h_t, mask_a=None, mask_t=None):
        # [B, L_a, d_audio] -> [B, L_a, d_model]
        h_a_proj = self.audio_proj(h_a)
        # [B, L_t, d_text] -> [B, L_t, d_model]
        h_t_proj = self.text_proj(h_t)

        # Call backbone (it expects aligned dims already)
        logits, beta, fused = self.backbone(h_a_proj, h_t_proj, mask_a, mask_t)
        return logits, beta, fused
