# Adaptive Unified Multimodal Emotion Recognition Framework

This directory implements the core architecture of the **Adaptive Unified Multimodal Emotion Recognition Framework**.

The framework is designed to be **modular and unified**, capable of handling both MOSEI and IEMOCAP features by swapping encoders and decoders.

---

## 🏗️ 1. Core Framework (Sequence-Level / MOSEI)
*Current SOTA configuration used in the main experiments.*

| File Name | Component | Description |
| :--- | :--- | :--- |
| **`mosei_fusion_with_emotion_decoder.py`** | **Wrapper** | **[Top-Level]** Projects audio/text features to a shared `d_model` and calls the backbone. |
| **`fusion_with_emotion_decoder.py`** | **Backbone** | **[The Pipeline]** Connects Encoder $\to$ Fusion $\to$ Decoder. Manages `return_attention` flow for explainability. |
| **`cross_modal_block_tacfn.py`** | Encoder | **[Module 1]** Cross-Modal Transformer (Alignment). |
| **`beta_gate_tacfn.py`** | Fusion | **[Module 2]** Vector-wise Beta Gating (Adaptive Fusion). |
| **`emotion_decoder.py`** | Decoder | **[Module 3]** Query-based Emotion Decoder (Attribution). |

---

## 📘 2. Technical Details (Core Modules)

### A. Encoder: `cross_modal_block_tacfn.py`
**Goal:** Reduce intra-modal redundancy and align audio–text representations.

**Key Mechanism:**
1.  **Intra-modal self-attention**: Each modality first passes through a self-attention layer to highlight salient cues.
2.  **Bidirectional cross-modal attention**: Audio attends to text, and text attends to audio.
3.  **Explainability (Issue #1)**: Supports `return_attention=True` to export the Audio-Text alignment map.

**Input/Output:**
* Input: `h_a` $[B, L_a, d]$, `h_t` $[B, L_t, d]$
* Output: `h_a_tilde`, `h_t_tilde` (Aligned features)

### B. Fusion: `beta_gate_tacfn.py`
**Goal:** Learn an adaptive, fine-grained fusion using a **per-dimension gating vector**.

**Algorithm:**
1.  **Normalization**: $\tilde{h}_a = LN(h_a),\quad \tilde{h}_t = LN(h_t)$
2.  **Pooling**: $a = \text{Pool}(\tilde{h}_a),\quad t = \text{Pool}(\tilde{h}_t)$
3.  **Gate Construction**: $g = [a,\ t,\ |a-t|,\ a \odot t]$
4.  **Prediction**: $w = \sigma(\text{MLP}(g)) \in [0,1]^d$
5.  **Fusion**: $h_\text{fusion} = w \odot \tilde{h}_a + (1-w) \odot \tilde{h}_t$

**Outputs**:
* `h_fusion`: Fused sequence $[B, L, d]$.
* `beta`: Scalar mean of $w$ for analyzing modality dominance.

### C. Decoder: `emotion_decoder.py`
**Goal:** Extract emotion-specific evidence using learnable queries.

**Key Mechanism:**
1.  **Learnable Queries**: Initialized with distinct vectors for each emotion (e.g., Happy, Sad).
2.  **Cross-Attention**: Queries attend to the `h_fusion` memory to find supporting evidence.
3.  **Explainability (Issue #2)**: Exports the attention weights to show *where* in the sequence the model found evidence for a specific emotion.

---

## 🧩 3. Variants & Baselines (Deprecated / Legacy / IEMOCAP)
*These files demonstrate the framework's adaptability to different granularities and provide baselines for ablation studies.*

| File Name | Role | Description |
| :--- | :--- | :--- |
| `cross_modal_block.py` | Encoder | Standard Cross-Modal Attention (simplified for utterance-level). |
| `beta_gate.py` | Fusion | Scalar-based Beta Gating (simpler version of vector-wise gating). |
| `fusion_classifier.py` | **Baseline** | A standard MLP classifier (No Decoder). Used to demonstrate the improvement brought by the `EmotionDecoder`. |

---

## 📐 Architecture Diagram

```
      [ Audio & Text Inputs ]
                 │
                 ▼
       [ Linear Projection ]
                 │
                 ▼
    +-------------------------------------------------------+
    |  WRAPPER: MoseiFusionWrapper                          |
    |                                                       |
    |  +--- BACKBONE: FusionWithEmotionDecoder -----------+ |
    |  |                                                  | |
    |  |   [ CrossModalBlock (Encoder) ]                  | |
    |  |          │      │                                | |
    |  |          │      +-----> [ Alignment Map ]        | |
    |  |          ▼              (Explainability #1)      | |
    |  |                                                  | |
    |  |   [ BetaGate (Fusion) ]                          | |
    |  |          │      │                                | |
    |  |          │      +-----> [ Beta Values ]          | |
    |  |          ▼              (Modality Weights)       | |
    |  |                                                  | |
    |  |   [ Fused Sequence ]                             | |
    |  |          │                                       | |
    |  |          ▼                                       | |
    |  |   [ EmotionDecoder (Decoder) ]                   | |
    |  |          │      │                                | |
    |  |          │      +-----> [ Attribution Map ]      | |
    |  |          ▼              (Explainability #2)      | |
    |  +--------------------------------------------------+ |
    |             │                                         |
    +-------------------------------------------------------+
                  │
                  ▼
   [ Logits (Emotion Predictions) ]
```