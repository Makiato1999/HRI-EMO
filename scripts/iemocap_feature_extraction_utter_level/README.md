# Multimodal Representation Extraction via Transfer Learning

We employ a feature-based transfer learning pipeline to obtain rich, high-level representations from both text and audio modalities.
Instead of training encoders from scratch, we reuse large pretrained Transformer models — BERT for text and WavLM for audio — to extract semantic and prosodic embeddings that serve as the foundation for subsequent cross-modal fusion and emotion decoding.

#### Create virtual environment - Conda
```
conda create -n beta_decoder python=3.10 -y
conda activate beta_decoder

pip install -U torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers pandas scikit-learn tqdm
```

## Text Modality: Contextual Representation Learning
#### Model
Backbone: bert-base-uncased (or RoBERTa, etc.)

Preprocessing: Text is tokenized and padded using the model tokenizer (AutoTokenizer).

Encoding: The token sequence is passed through the pretrained language model.

Pooling Strategy: The [CLS] embedding or mean pooling of all token representations is taken as ℎ_text.

#### Pipeline Overview
```
Raw transcript
   ↓
Tokenizer (BERT)
   ↓
Transformer Encoder (BERT)
   ↓
Hidden States [T, d]
   ↓
Mean/CLS Pooling
   ↓
h_text ∈ ℝ^(d_t)
```
#### Equation

$$
h_\text{text} = \text{Pool}\big(\text{BERT}(\text{Tokenizer}(x_\text{text}))\big)
\quad\in\mathbb{R}^{d_t}
$$

#### Characteristics
- Encodes contextual meaning of the utterance.
- Sensitive to emotional and semantic cues in spoken text.
- Output embedding dimension 𝑑_t = 768.

#### How to run
```
python scripts/iemocap_feature_extraction_utter_level/extract_text_feats_bert.py \
  --csv data/iemocap_index_splits.csv \
  --model_name bert-base-uncased \
  --batch_size 64 \
  --max_len 128 \
  --out_dir features/utter_level/text
```

```
python scripts/iemocap_feature_extraction_utter_level/train_text_baseline.py \
  --csv data/iemocap_index_splits.csv \
  --feat_dir features/text \
  --mlp_hidden 256 \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-3 \
  --out_dir runs/train_text_baseline
```

## Audio Modality: Speech Representation Learning
#### Model
Backbone: microsoft/wavlm-base-plus
(trained on 94k hours of diverse speech data; captures phonetic and prosodic cues)

Preprocessing: Raw waveforms are normalized to [-1, 1] and truncated/padded to fixed duration.
AutoFeatureExtractor performs zero-padding and generates attention_mask.

Encoding:
The pretrained WavLM model converts the waveform into frame-level hidden states.

Pooling Strategy: A mask-weighted mean pooling aligns and averages time steps to produce a fixed-length embedding ℎ_audio.

#### Pipeline Overview
```
Raw waveform (.wav)
   ↓
Resample → Normalize → Padding (FeatureExtractor)
   ↓
Transformer Encoder (WavLM)
   ↓
Mask-weighted Mean Pooling (Time-wise)
   ↓
h_audio ∈ ℝ^(d_a)
```
#### Equation

$$
h_\text{audio} = \text{MeanPool}\big(\text{WavLM}(\text{FeatureExtractor}(x_\text{audio}))\big)
\quad\in\mathbb{R}^{d_a}
$$

#### Characteristics
- Captures acoustic, prosodic, and speaker-related features.
- Retains temporal expressiveness (before pooling).
- Output embedding dimension 𝑑_𝑎 = 768.

#### How to run
```
python scripts/iemocap_feature_extraction_utter_level/extract_audio_feats_wavlm.py \
  --csv data/iemocap_index_splits.csv \
  --model_name microsoft/wavlm-base-plus \
  --target_sr 16000 \
  --max_seconds 10 \
  --batch_size 16 \
  --out_dir features/utter_level/audio
```
## Output Summary
| Modality  | Encoder | Feature Extractor | Output Shape   | Training Strategy      |
| --------- | ------- | ----------------- | -------------- | ---------------------- |
| **Text**  | BERT    | Tokenizer         | `[batch, d_t]` | Fine-tuned or frozen   |
| **Audio** | WavLM   | FeatureExtractor  | `[batch, d_a]` | Frozen (feature-based) |

Both ℎ_text and ℎ_audio are fixed-length representations, suitable for fusion layers such as:

- Concatenation
- Cross-attention
- β-adaptive fusion (e.g., via TACFN β-adaptive fusion)
- Emotion-level decoder

## Conceptual Diagram (for Figure)
```
                 ┌──────────────────────────────────────────┐
                 │             Pretrained Encoders          │
                 │ (Feature-based Transfer Learning Stage)  │
                 └──────────────────────────────────────────┘
                             ┌────────────┬─────────────┐
                             │            │              │
                             ▼            ▼              ▼
                    [Text Modality]   [Audio Modality]   [Other...]
                 ┌────────────────┐ ┌─────────────────┐
                 │ BERT Encoder   │ │ WavLM Encoder   │
                 └──────┬─────────┘ └──────┬──────────┘
                        │                  │
                   h_text ∈ ℝ^768      h_audio ∈ ℝ^768
                        └──────────┬──────────┘
                                   ▼
                        Cross-modal Fusion + Decoder
                                   ▼
                            Emotion Prediction
```

## Testing
#### Stage 1: Feature Validation
- Audio & text features verified (3,694 samples each), consistent shape (768,).
- No NaN/Inf detected, aligned with iemocap_index_splits.csv.
   → Features validated and ready for fusion.

#### Stage 2: Linear Probe Results
| Model               | Input            | Acc      | F1       | Key Insight                         |
| ------------------- | ---------------- | -------- | -------- | ----------------------------------- |
| Audio-only          | Audio (768D)     | ~0.40    | ~0.33    | Captures tonal cues only            |
| Text-only           | Text (768D)      | ~0.47    | ~0.39    | Stronger semantic signal            |
| Audio+Text (Concat) | `[a; t]` (1536D) | **0.53** | **0.42** | Confirms multimodal complementarity |

→ Even a linear probe shows robust emotional information in embeddings.

## Summary
We adopt a feature-based transfer learning paradigm where modality-specific pretrained encoders (BERT for text and WavLM for audio) are used to extract semantically and acoustically rich embeddings ℎ_text and ℎ_audio, which serve as inputs to the multimodal fusion and emotion decoding stages.
