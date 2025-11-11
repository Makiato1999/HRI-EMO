#!/usr/bin/env python3
import torch
import torch.nn as nn


class EmotionDecoder(nn.Module):
    """
    Emotion-Level Transformer Decoder with Learnable Emotion Queries.

    Each emotion is represented by a learnable query vector.
    The decoder uses cross-attention from these queries to the fused
    multimodal sequence (memory) to extract emotion-specific evidence.

    Inputs:
        memory: [B, L, d]  - fused sequence from cross-modal fusion
        memory_key_padding_mask: [B, L] bool or None (True = PAD)

    Outputs:
        z:      [B, num_emotions, d] - emotion-specific embeddings
        logits: [B, num_emotions] or None (if classifier provided outside)
    """

    def __init__(
        self,
        d_model: int = 768,
        num_emotions: int = 4,
        n_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_output_layer: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_emotions = num_emotions
        self.use_output_layer = use_output_layer

        # Learnable emotion queries: [num_emotions, d]
        self.emotion_queries = nn.Parameter(
            torch.randn(num_emotions, d_model)
        )

        # Stacked Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Optional output projection: one logit per emotion
        if use_output_layer:
            self.out_proj = nn.Linear(d_model, 1)
        else:
            self.out_proj = None

    def forward(
        self,
        memory: torch.Tensor,                      # [B, L, d]
        memory_key_padding_mask: torch.Tensor | None = None,  # [B, L] or None
    ):
        B = memory.size(0)

        # 1) Expand emotion queries for the batch
        #    queries: [B, num_emotions, d]
        queries = self.emotion_queries.unsqueeze(0).expand(B, -1, -1)

        # 2) Run Transformer decoder
        #    - queries attend over "memory" (fused multimodal sequence)
        #    - no causal mask (we want full context)
        z = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, num_emotions, d]

        # 3) Optionally map each emotion embedding to a logit
        if self.out_proj is not None:
            # out_proj per emotion: [B, num_emotions, 1] -> [B, num_emotions]
            logits = self.out_proj(z).squeeze(-1)
            return z, logits

        return z, None
