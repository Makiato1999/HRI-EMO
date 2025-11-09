#!/usr/bin/env python3
import torch
from models.cross_modal_block import CrossModalTransformer

def main():
    print("=== Utter-level test ===")
    h_a = torch.randn(32, 1, 768)
    h_t = torch.randn(32, 1, 768)
    model = CrossModalTransformer(num_layers=2)
    h_a_tilde, h_t_tilde = model(h_a, h_t)
    print("Output shapes:", h_a_tilde.shape, h_t_tilde.shape)

    print("\n=== Sequence-level test ===")
    h_a = torch.randn(8, 400, 768)
    h_t = torch.randn(8, 128, 768)
    mask_a = torch.zeros(8, 400, dtype=torch.bool)
    mask_t = torch.zeros(8, 128, dtype=torch.bool)
    h_a_tilde, h_t_tilde = model(h_a, h_t, mask_a, mask_t)
    print("Output shapes:", h_a_tilde.shape, h_t_tilde.shape)

if __name__ == "__main__":
    main()

"""
python -m tests.test_cross_modal_block
"""