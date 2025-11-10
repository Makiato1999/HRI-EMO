#!/usr/bin/env python3
import torch
import torch.nn as nn

from .cross_modal_block import CrossModalTransformer
from .beta_gate import BetaGate


class FusionClassifier(nn.Module):
    """
    Cross-Modal Transformer + Beta-Gating + Classification Head

    This model integrates three key modules:
        1. CrossModalTransformer  performs bidirectional semantic alignment
        2. BetaGate               learns adaptive modality weighting β
        3. Classifier Head        maps the fused representation to emotion classes

    It supports both:
        - Utterance-level features: [B, d]
        - Sequence-level features:  [B, L, d]

    The internal logic automatically handles dimensionality
    and can therefore be reused in both the quick β-Fusion (utterance)
    and full β + Decoder (sequence) pipelines.
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
            d_model:     Hidden dimension of both modalities.
            num_classes: Number of output emotion classes.
            n_heads:     Number of attention heads in cross-modal layers.
            num_layers:  Number of stacked CrossModalBlocks.
            beta_hidden: Hidden dimension inside the β-gate MLP.
            dropout:     Dropout rate used in both attention and classifier.
        """
        super().__init__()

        # --- 1. Cross-modal semantic alignment ---
        self.cross_modal = CrossModalTransformer(
            num_layers=num_layers,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # --- 2. Adaptive β-Gating Mechanism ---
        self.beta_gate = BetaGate(
            d_model=d_model,
            hidden_dim=beta_hidden,
        )

        # --- 3. Classifier Head ---
        # Maps pooled fusion representation → emotion logits
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
        If x is [B, d], it is automatically expanded to [B, 1, d].
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
        Args:
            h_a: [B, d] or [B, L_a, d] audio embeddings
            h_t: [B, d] or [B, L_t, d] text embeddings
            mask_a: Optional [B, L_a] boolean padding mask
            mask_t: Optional [B, L_t] boolean padding mask

        Returns:
            logits:          [B, num_classes] classification outputs
            beta:            [B, 1]           learned modality gate
            h_fusion_pooled: [B, d]           pooled fused representation
        """
        # 1) Standardize inputs to [B, L, d]
        h_a = self._ensure_3d(h_a)
        h_t = self._ensure_3d(h_t)

        # 2) Cross-modal semantic alignment
        h_a_tilde, h_t_tilde = self.cross_modal(h_a, h_t, mask_a, mask_t)

        # 3) Adaptive β-Fusion
        h_fusion, beta = self.beta_gate(h_a_tilde, h_t_tilde, mask_a, mask_t)
        # h_fusion: [B, L, d]

        # 4) Pool along sequence dimension to obtain utterance-level vector
        h_fusion_pooled = h_fusion.mean(dim=1)  # identical if L=1

        # 5) Feed through classification head
        logits = self.classifier(h_fusion_pooled)

        return logits, beta, h_fusion_pooled
