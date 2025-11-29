# Training Scripts (MOSEI)

This directory contains the training pipelines for the HRI-EMO model.
It focuses on training the **Cross-Modal Fusion + Emotion Decoder** architecture on the CMU-MOSEI dataset.

## 📂 File Descriptions

| File Name | Type | Description |
| :--- | :--- | :--- |
| **`mosei_train.ipynb`** | Notebook | **[Main Launcher]** The recommended way to run experiments. It sets up the environment and triggers the training script. |
| **`train_mosei_fusion_seq_level_decoder.py`** | **Core Script** | **[Main Logic]** The primary training script containing the training loop, validation logic, loss calculation (BCE), and checkpoint saving. |
| `train_fusion_*.py` | Legacy | Older variations of training scripts (e.g., utter-level or without decoder). Kept for reference. |

---

## 🚀 Quick Start

You can run experiments interactively via the Notebook or directly via the Command Line Interface (CLI).

### Option 1: Using the Notebook (Recommended)
Open **`mosei_train.ipynb`**. Uncomment the "v2" configuration cell to run the optimized training routine.

### Option 2: Command Line Interface (CLI)

Below is the **v2 Configuration (Best Generalization)**. This setup solves overfitting by simplifying the model and increasing regularization.

```bash
# 1. Setup path
cd /content/HRI-EMO
export PYTHONPATH=.

# 2. Run Training (v2 Config)
python scripts/fusion/train_mosei_fusion_seq_level_decoder.py \
  --index_csv data/mosei_index_splits.csv \
  --audio_dir features/mosei/seq_level/audio \
  --text_dir features/mosei/seq_level/text \
  --out_dir /content/drive/MyDrive/.../mosei_fusion_decoder_v2 \
  --epochs 20 \
  --batch_size 16 \
  --grad_accum 2 \
  --lr 5e-5 \
  --weight_decay 0.05 \
  --dropout 0.4 \
  --d_model 256 \
  --n_heads 4 \
  --num_layers_fusion 1 \
  --num_layers_decoder 2 \
  --beta_hidden 64 \
  --beta_entropy 1e-3 \
  --warmup_ratio 0.1 \
  --select_by calibrated_macro_f1 \
  --save_calibrated_ths \
  --seed 1234