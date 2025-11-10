#!/usr/bin/env python3
import torch
from models.fusion_classifier import FusionClassifier

def main():
    B, d, C = 16, 768, 4
    model = FusionClassifier(d_model=d, num_classes=C)

    # --- Utterance-level test ---
    h_a = torch.randn(B, d)
    h_t = torch.randn(B, d)
    logits, beta, h_fused = model(h_a, h_t)
    print("Utter-level:")
    print(" logits:", logits.shape)   # [B, C]
    print(" beta:", beta.shape)       # [B, 1]
    print(" h_fused:", h_fused.shape) # [B, d]

    # --- Sequence-level test ---
    h_a = torch.randn(B, 400, d)
    h_t = torch.randn(B, 128, d)
    mask_a = torch.zeros(B, 400, dtype=torch.bool)
    mask_t = torch.zeros(B, 128, dtype=torch.bool)
    logits, beta, h_fused = model(h_a, h_t, mask_a, mask_t)
    print("\nSeq-level:")
    print(" logits:", logits.shape)
    print(" beta:", beta.shape)
    print(" h_fused:", h_fused.shape)

if __name__ == "__main__":
    main()
