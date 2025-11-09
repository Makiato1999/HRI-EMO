# Multimodal Sequence-Level Representation Extraction via Transfer Learning

We adopt a **feature-based transfer learning** pipeline to obtain fine-grained, temporally structured representations from both text and audio modalities.  
Instead of training encoders from scratch, we reuse large pretrained Transformer models — **BERT** for text and **WavLM** for audio — to extract token/frame-level embeddings that preserve the internal temporal and contextual information of each utterance.  
These sequence-level embeddings form the foundation for subsequent **Cross-Modal Semantic Alignment**, **β-Adaptive Fusion**, and **Emotion-Level Transformer Decoding**.

---

## Environment Setup

#### Create virtual environment with Conda
```bash
conda create -n beta_decoder python=3.10 -y
conda activate beta_decoder

pip install -U torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers pandas scikit-learn tqdm
```

## Text Modality: Contextual Sequence Representation
Model

Backbone: bert-base-uncased (or roberta-base, etc.)
Objective: Retain token-level contextualized embeddings for each utterance.

Preprocessing

Text is tokenized, truncated, and padded using the model tokenizer (AutoTokenizer).
Each utterance is represented as a sequence of L tokens with corresponding attention masks.

Encoding

The tokenized sequence is fed into the pretrained language model.
Instead of taking only the [CLS] token or pooled output, we retain all token hidden states from the final encoder layer.

#### Pipeline Overview

```bash
Raw transcript
   ↓
Tokenizer (BERT)
   ↓
Transformer Encoder (BERT)
   ↓
Hidden States [L, d]
   ↓
Saved directly (no pooling)
```

#### Equation
$$
h_\text{text} = \text{Pool}\big(\text{BERT}(\text{Tokenizer}(x_\text{text}))\big)
\quad\in\mathbb{R}^{L×d_t}
$$

#### Characteristics
- Preserves word-level temporal and contextual structure.

- Enables fine-grained cross-modal attention in fusion layers.

- Output dimension per token: 𝑑_𝑡 =768.

#### How to run

```bash
python scripts/feature_extraction_seq_level/extract_text_feats_bert_seq.py \
  --csv data/iemocap_index_splits.csv \
  --model_name bert-base-uncased \
  --max_len 128 \
  --out_dir features/seq_level/text
```
## Audio Modality: Speech Frame Representation
Model

Backbone: microsoft/wavlm-base-plus
(trained on 94k hours of speech; captures phonetic, prosodic, and emotional cues)

Preprocessing

Raw waveforms are normalized to [-1, 1] and truncated/padded to a fixed duration.
AutoFeatureExtractor performs zero-padding and produces an attention mask to indicate valid frames.

Encoding

The pretrained WavLM encoder transforms the waveform into a sequence of frame-level hidden states (T' frames).
The temporal structure is preserved — no mean pooling is applied.

#### Pipeline Overview
```bash
Raw waveform (.wav)
   ↓
Resample → Normalize → Padding (FeatureExtractor)
   ↓
Transformer Encoder (WavLM)
   ↓
Frame-level Hidden States [T', d]
   ↓
Saved directly (no pooling)
```

#### Equation
$$
h_\text{audio} = \text{MeanPool}\big(\text{WavLM}(\text{FeatureExtractor}(x_\text{audio}))\big)
\quad\in\mathbb{R}^{T^′× d_a}
$$

#### Characteristics

- Retains phonetic and prosodic structure at frame level.

- Enables time-aligned semantic fusion with text modality.

- Output embedding dimension 𝑑_𝑎 = 768.

#### How to run
```bash
python scripts/feature_extraction_seq_level/extract_audio_feats_wavlm_seq.py \
  --csv data/iemocap_index_splits.csv \
  --audio_root data/wavs \
  --model_name microsoft/wavlm-base-plus \
  --target_sr 16000 \
  --max_seconds 10 \
  --out_dir features/seq_level/audio
```

---

## Output Summary

| Modality  | Encoder | Feature Extractor | Output Shape       | Temporal Info | Training Strategy      |
| --------- | ------- | ----------------- | ------------------ | ------------- | ---------------------- |
| **Text**  | BERT    | Tokenizer         | `[batch, L, d_t]`  | preserved   | Frozen (feature-based) |
| **Audio** | WavLM   | FeatureExtractor  | `[batch, T', d_a]` | preserved   | Frozen (feature-based) |

Both 
𝐻_text and 𝐻_audio are sequence-level hidden states (no pooling), designed for:

- Cross-modal semantic alignment via bidirectional attention

- β-adaptive gating for reliability-aware fusion

- Emotion-level Transformer decoding for interpretable emotion query extraction

---

## Conceptual Diagram (for Figure)

```bash
                 ┌──────────────────────────────────────────┐
                 │          Pretrained Encoders             │
                 │   (Sequence-level Feature Extraction)    │
                 └──────────────────────────────────────────┘
                             ┌────────────┬─────────────┐
                             │            │              │
                             ▼            ▼              ▼
                    [Text Modality]   [Audio Modality]   [Other...]
                 ┌────────────────┐ ┌─────────────────┐
                 │  BERT Encoder  │ │  WavLM Encoder  │
                 └──────┬─────────┘ └──────┬──────────┘
                        │                  │
              H_text ∈ ℝ^(L×768)      H_audio ∈ ℝ^(T'×768)
                        └──────────┬──────────┘
                                   ▼
                     Cross-Modal Transformer + β-Fusion
                                   ▼
                         Emotion-Level Transformer Decoder
                                   ▼
                             Emotion Prediction
```

---

## Verification and Meta Information

Each modality directory includes a meta.json file describing feature properties:

Text
```bash
{
  "model": "bert-base-uncased",
  "hidden_dim": 768,
  "max_len": 128,
  "note": "seq-level BERT features: hidden[L,H] + attention_mask[L]"
}
```

Audio
```bash
{
  "model": "microsoft/wavlm-base-plus",
  "hidden_dim": 768,
  "target_sr": 16000,
  "max_seconds": 10.0,
  "note": "seq-level WavLM features: hidden[T',H] + attention_mask[T'] (downsampled)"
}
```

- Features verified (3,694 utterances per modality)

- Shapes consistent: [L, 768] (text) / [T', 768] (audio)

- No NaN or Inf values detected

- Aligned with iemocap_index_splits.csv

## Summary

We extract sequence-level multimodal representations using pretrained Transformer encoders, preserving temporal and contextual dynamics of both speech and text.

These rich embeddings 𝐻_audio and 𝐻_text form the basis for the subsequent Cross-Modal Transformer, β-Adaptive Fusion, and Emotion-Level Decoder components that together constitute the multimodal emotion understanding framework.