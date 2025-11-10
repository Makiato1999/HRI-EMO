# TACFN-inspired Cross-Modal Fusion Network

This folder implements a **Transformer-based adaptive fusion network** for multimodal emotion recognition, inspired by TACFN-style cross-modal interaction and extended to pretrained encoders (BERT, WavLM).

We assume utterance-level or sequence-level features have already been extracted, e.g.:

- Text: hidden states from BERT
- Audio: hidden states from WavLM

The fusion network operates purely on these features.

---

## 1. cross_modal_block_tacfn.py — CrossModalTransformer

**Goal:** Reduce intra-modal redundancy and align audio–text representations.

**Key ideas:**

1. **Intra-modal self-attention**
   - Each modality (audio/text) first passes through a self-attention layer.
   - Highlights salient frames/tokens and suppresses noisy ones.
   - Implemented with residual connections and LayerNorm.

2. **Bidirectional cross-modal attention**
   - Audio attends to text, and text attends to audio.
   - Captures complementary cues across modalities.
   - Followed by FFN + residual + LayerNorm (Transformer encoder style).

**Input/Output:**

- Input:
  - `h_a`: `[B, L_a, d]` audio sequence
  - `h_t`: `[B, L_t, d]` text sequence
  - optional padding masks `mask_a`, `mask_t` (True = PAD)
- Output:
  - `h_a_tilde`: `[B, L_a, d]`
  - `h_t_tilde`: `[B, L_t, d]`

This module is fully compatible with:
- utterance-level (use `L=1`),
- sequence-level (use full `[L, d]` features).

---

## 2. beta_gate_tacfn.py — Vector-wise BetaGate

**Goal:** Learn an adaptive, fine-grained fusion of the two modalities.

Instead of a single scalar β per sample, we use a **per-dimension gating vector**
to decide, for each latent feature dimension, how much to trust audio vs text.

**Steps:**

1. Apply LayerNorm to each modality sequence:
   $$
   \tilde{h}_a = LN(h_a),\quad \tilde{h}_t = LN(h_t)
   $$

2. Masked mean-pooling:
   $$
   a = \text{Pool}(\tilde{h}_a),\quad t = \text{Pool}(\tilde{h}_t)
   $$

3. Construct gate input:
   $$
   g = [a,\ t,\ |a-t|,\ a \odot t]
   $$

4. Predict vector gate:
   $$
   w = \sigma(\text{MLP}(g)) \in [0,1]^d
   $$

5. Broadcast and fuse:
   - Expand `w` to `[B, L_f, d]`
   - Fuse normalized sequences:
     $$
     h_\text{fusion} = w \odot \tilde{h}_a + (1-w) \odot \tilde{h}_t
     $$

6. For interpretability, we log:
   $$
   \beta_\text{mean} = \frac{1}{d}\sum_j w_j
   $$
   as a scalar indicator of modality preference.

**Output:**

- `h_fusion`: `[B, L_f, d]` fused sequence representation
- `beta`: `[B, 1]` mean gate value for analysis/plots

This design:
- is **TACFN-inspired** (weight vector instead of scalar),
- is robust against scale mismatch between pretrained encoders,
- avoids trivial single-modality dominance.

---

## 3. fusion_classifier.py — End-to-End Fusion Model

Wraps everything into a classification-ready module.

**Pipeline:**

1. Ensure inputs are `[B, L, d]` (utterance `[B, d]` → `[B, 1, d]`).
2. Pass through `CrossModalTransformer`:
   - get `h_a_tilde`, `h_t_tilde`.
3. Apply `BetaGate`:
   - get fused sequence `h_fusion` and `beta`.
4. Mean-pool over time:
   - `h_fusion_pooled ∈ [B, d]`.
5. Classifier head:
   - `LayerNorm → Linear → ReLU → Dropout → Linear`
   - outputs logits `[B, num_classes]`.

**Signature:**

```python
logits, beta, h_fused = model(h_a, h_t, mask_a=None, mask_t=None)
```