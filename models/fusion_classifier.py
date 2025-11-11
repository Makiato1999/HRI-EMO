#!/usr/bin/env python3
import torch
import torch.nn as nn

from .cross_modal_block_tacfn import CrossModalTransformer
from .beta_gate_tacfn import BetaGate


class FusionClassifier(nn.Module):
    """
    TACFN-inspired Multimodal Fusion Classifier

    This architecture integrates three major modules:

        1. CrossModalTransformer
           - Implements TACFN-style intra-modal self-attention for redundancy reduction.
           - Performs bidirectional cross-modal attention for semantic alignment.
           - Uses residual connections and LayerNorm for training stability.

        2. BetaGate (vector-wise adaptive fusion)
           - Predicts a per-dimension gating vector β ∈ [0,1]^d instead of a scalar.
           - Learns fine-grained modality weighting using normalized pooled representations.
           - Produces adaptive, differentiable fusion between modalities.

        3. Classification Head
           - Maps the fused multimodal representation into emotion logits.
           - Includes LayerNorm, Linear, ReLU, and Dropout layers for regularization.

    The model supports both:
        - Utterance-level features: [B, d]
        - Sequence-level features:  [B, L, d]

    Its unified design allows direct reuse in both quick β-Fusion (utterance-level)
    and full β + Decoder (sequence-level) pipelines.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_classes: int = 4,
        n_heads: int = 8,
        num_layers: int = 2,
        beta_hidden: int = 256,
        dropout: float = 0.2,
    ):
        """
        Args:
            d_model:     Dimensionality of the hidden representations for all modalities.
            num_classes: Number of emotion categories for classification.
            n_heads:     Number of attention heads in each cross-modal layer.
            num_layers:  Number of stacked TACFN-style CrossModalBlocks.
            beta_hidden: Hidden dimension inside the β-gate MLP.
            dropout:     Dropout rate used across attention and classifier modules.
        """
        super().__init__()

        # --- 1. Cross-modal interaction encoder (TACFN-style) ---
        self.cross_modal = CrossModalTransformer(
            num_layers=num_layers,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # --- 2. Adaptive β-Gating Mechanism (vector-wise) ---
        self.beta_gate = BetaGate(
            d_model=d_model,
            hidden_dim=beta_hidden,
        )

        # --- 3. Classification Head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    # -------------------------------------------------------------
    # Utility: ensure the input has a sequence dimension
    # -------------------------------------------------------------
    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensures input tensor has shape [B, L, d].
        Expands [B, d] → [B, 1, d] automatically for utterance-level features.
        """
        if x.dim() == 2:
            return x.unsqueeze(1)
        elif x.dim() == 3:
            return x
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

    # -------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------
    def forward(
        self,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        mask_a: torch.Tensor | None = None,
        mask_t: torch.Tensor | None = None,
    ):
        """
        Forward Pass:

        1. Input Normalization
           - Standardizes input shapes to [B, L, d].

        2. Cross-modal Semantic Alignment
           - Each modality first undergoes intra-modal self-attention (redundancy filtering).
           - Bidirectional cross-modal attention then aligns audio and text semantics.

        3. Adaptive β-Fusion
           - The BetaGate predicts a per-dimension gating vector β ∈ [0,1]^d
             based on pooled, normalized modality statistics.
           - The fusion is vector-wise: h_fusion = β ⊙ h_a + (1 - β) ⊙ h_t.
           - Returns both the fused sequence and mean β for interpretability.

        4. Sequence Pooling
           - Mean-pools the fused sequence across temporal dimension
             to obtain a compact utterance-level representation.

        5. Emotion Classification
           - The pooled vector passes through a lightweight MLP head
             to produce class logits.

        Returns:
            logits:          [B, num_classes] – emotion predictions
            beta:            [B, 1]           – mean gating weight for interpretability
            h_fusion_pooled: [B, d]           – pooled fused representation
        """
        # --- 1) Ensure inputs are 3D ---
        h_a = self._ensure_3d(h_a)
        h_t = self._ensure_3d(h_t)

        # --- 2) Cross-modal semantic alignment ---
        h_a_tilde, h_t_tilde = self.cross_modal(h_a, h_t, mask_a, mask_t)

        # --- 3) Adaptive β-fusion ---
        h_fusion, beta = self.beta_gate(h_a_tilde, h_t_tilde, mask_a, mask_t)

        # --- 4) Sequence pooling ---
        h_fusion_pooled = h_fusion.mean(dim=1)

        # --- 5) Classification ---
        logits = self.classifier(h_fusion_pooled)

        return logits, beta, h_fusion_pooled
