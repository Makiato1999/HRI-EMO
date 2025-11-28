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

## 🚀 How to Run

### Option 1: Using the Notebook (Recommended)
Open **`mosei_train.ipynb`**. It contains pre-configured cells for different experiment versions (e.g., v1, v2). Just uncomment the version you want to run and execute the cell.

### Option 2: Command Line Interface (CLI)
You can run the python script directly from the terminal.

#### ➤ Recommended Configuration (v2)
*This configuration is optimized for generalization (prevents overfitting).*

**Key Hyperparameters:**
* `--num_layers_fusion 1` (Simplified fusion to force learning core patterns)
* `--dropout 0.4` (High dropout for regularization)
* `--select_by calibrated_macro_f1` (Save best model based on F1, not just Loss)

```bash
# Ensure you are in the project root
export PYTHONPATH=.

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






#### train_fusion_utter_level.py
```bash
python -m scripts.fusion.train_fusion_utter_level \
  --csv data/iemocap_index_splits.csv \
  --audio_dir features/utter_level/audio \
  --text_dir features/utter_level/text \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-4 \
  --dropout 0.3 \
  --out_dir runs/fusion_utter_level_dp03
```

#### train_fusion_seq_level.py
```bash
python -m scripts.fusion.train_fusion_seq_level \
  --csv data/iemocap_index_splits.csv \
  --audio_dir features/seq_level/audio \
  --text_dir features/seq_level/text \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4 \
  --dropout 0.2 \
  --out_dir runs/fusion_seq_level_tacfn_like
```


#### Colab 
```bash
!python -m scripts.fusion.train_mosei_fusion_seq_level_decoder \
  --index_csv ../data/mosei_index_splits.csv \
  --audio_dir ../features/mosei/seq_level/audio \
  --text_dir ../features/mosei/seq_level/text \
  --epochs 20 \
  --batch_size 8 \
  --grad_accum 4 \
  --warmup_ratio 0.1 \
  --beta_entropy 1e-3 \
  --max_len_audio 300 \
  --max_len_text 128 \
  --d_model 384 \
  --n_heads 6 \
  --num_layers_fusion 2 \
  --num_layers_decoder 2 \
  --dropout 0.2 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --num_workers 2 \
  --select_by macro_auc \
  --save_calibrated_ths \
  --out_dir /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_small \
  --seed 1234


```