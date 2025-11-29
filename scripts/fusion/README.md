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

## ✅ Output
```
/content/HRI-EMO
[Dataset] Final size: 16327 samples
[Dataset] Final size: 1871 samples
[pos_weight] {'emo_happy': np.float32(0.87), 'emo_sad': np.float32(2.82), 'emo_anger': np.float32(3.63), 'emo_fear': np.float32(8.94), 'emo_disgust': np.float32(4.53), 'emo_surprise': np.float32(11.27)}

=== Epoch 1/20 ===
[Val Calibrated] macro-F1=0.377 | thresholds=[0.05 0.05 0.15 0.1  0.15 0.05]
[Val Metrics] Loss=0.4340 | micro-F1=0.000 | macro-F1=0.000 | macro-AUC=0.574
Train Loss: 0.4887 | F1 micro/macro: 0.029/0.036 | AUC macro: 0.503 | Mean β: 0.504  ||  Val Loss: 0.4340 | F1 micro/macro: 0.000/0.000 | AUC macro: 0.574 | Mean β: 0.503 | Calib macro-F1: 0.377

=== Epoch 2/20 ===
[Val Calibrated] macro-F1=0.403 | thresholds=[0.05 0.05 0.15 0.1  0.15 0.05]
[Val Metrics] Loss=0.4213 | micro-F1=0.027 | macro-F1=0.033 | macro-AUC=0.615
Train Loss: 0.4495 | F1 micro/macro: 0.006/0.008 | AUC macro: 0.574 | Mean β: 0.502  ||  Val Loss: 0.4213 | F1 micro/macro: 0.027/0.033 | AUC macro: 0.615 | Mean β: 0.502 | Calib macro-F1: 0.403

=== Epoch 3/20 ===
[Val Calibrated] macro-F1=0.412 | thresholds=[0.05 0.1  0.2  0.2  0.35 0.15]
[Val Metrics] Loss=0.4278 | micro-F1=0.121 | macro-F1=0.100 | macro-AUC=0.668
Train Loss: 0.4327 | F1 micro/macro: 0.029/0.032 | AUC macro: 0.647 | Mean β: 0.503  ||  Val Loss: 0.4278 | F1 micro/macro: 0.121/0.100 | AUC macro: 0.668 | Mean β: 0.505 | Calib macro-F1: 0.412

=== Epoch 4/20 ===
[Val Calibrated] macro-F1=0.413 | thresholds=[0.05 0.1  0.2  0.15 0.25 0.1 ]
[Val Metrics] Loss=0.4148 | micro-F1=0.067 | macro-F1=0.073 | macro-AUC=0.673
Train Loss: 0.4264 | F1 micro/macro: 0.037/0.040 | AUC macro: 0.671 | Mean β: 0.503  ||  Val Loss: 0.4148 | F1 micro/macro: 0.067/0.073 | AUC macro: 0.673 | Mean β: 0.504 | Calib macro-F1: 0.413

=== Epoch 5/20 ===
[Val Calibrated] macro-F1=0.413 | thresholds=[0.1  0.05 0.1  0.15 0.25 0.15]
[Val Metrics] Loss=0.4136 | micro-F1=0.058 | macro-F1=0.066 | macro-AUC=0.680
Train Loss: 0.4229 | F1 micro/macro: 0.046/0.049 | AUC macro: 0.682 | Mean β: 0.505  ||  Val Loss: 0.4136 | F1 micro/macro: 0.058/0.066 | AUC macro: 0.680 | Mean β: 0.506 | Calib macro-F1: 0.413

=== Epoch 6/20 ===
[Val Calibrated] macro-F1=0.420 | thresholds=[0.05 0.1  0.2  0.2  0.3  0.2 ]
[Val Metrics] Loss=0.4210 | micro-F1=0.107 | macro-F1=0.099 | macro-AUC=0.683
Train Loss: 0.4187 | F1 micro/macro: 0.052/0.055 | AUC macro: 0.698 | Mean β: 0.503  ||  Val Loss: 0.4210 | F1 micro/macro: 0.107/0.099 | AUC macro: 0.683 | Mean β: 0.502 | Calib macro-F1: 0.420

=== Epoch 7/20 ===
[Val Calibrated] macro-F1=0.424 | thresholds=[0.1  0.1  0.2  0.25 0.3  0.2 ]
[Val Metrics] Loss=0.4254 | micro-F1=0.098 | macro-F1=0.107 | macro-AUC=0.685
Train Loss: 0.4151 | F1 micro/macro: 0.058/0.062 | AUC macro: 0.707 | Mean β: 0.500  ||  Val Loss: 0.4254 | F1 micro/macro: 0.098/0.107 | AUC macro: 0.685 | Mean β: 0.500 | Calib macro-F1: 0.424

=== Epoch 8/20 ===
[Val Calibrated] macro-F1=0.421 | thresholds=[0.05 0.1  0.15 0.2  0.3  0.3 ]
[Val Metrics] Loss=0.4189 | micro-F1=0.085 | macro-F1=0.093 | macro-AUC=0.687
Train Loss: 0.4120 | F1 micro/macro: 0.065/0.069 | AUC macro: 0.714 | Mean β: 0.499  ||  Val Loss: 0.4189 | F1 micro/macro: 0.085/0.093 | AUC macro: 0.687 | Mean β: 0.500 | Calib macro-F1: 0.421

=== Epoch 9/20 ===
[Val Calibrated] macro-F1=0.419 | thresholds=[0.05 0.05 0.2  0.25 0.35 0.35]
[Val Metrics] Loss=0.4340 | micro-F1=0.129 | macro-F1=0.143 | macro-AUC=0.685
Train Loss: 0.4085 | F1 micro/macro: 0.071/0.074 | AUC macro: 0.722 | Mean β: 0.499  ||  Val Loss: 0.4340 | F1 micro/macro: 0.129/0.143 | AUC macro: 0.685 | Mean β: 0.499 | Calib macro-F1: 0.419

=== Epoch 10/20 ===
[Val Calibrated] macro-F1=0.420 | thresholds=[0.05 0.1  0.15 0.25 0.25 0.3 ]
[Val Metrics] Loss=0.4334 | micro-F1=0.114 | macro-F1=0.138 | macro-AUC=0.683
Train Loss: 0.4059 | F1 micro/macro: 0.075/0.079 | AUC macro: 0.729 | Mean β: 0.498  ||  Val Loss: 0.4334 | F1 micro/macro: 0.114/0.138 | AUC macro: 0.683 | Mean β: 0.497 | Calib macro-F1: 0.420

=== Epoch 11/20 ===
[Val Calibrated] macro-F1=0.424 | thresholds=[0.05 0.05 0.15 0.2  0.25 0.35]
[Val Metrics] Loss=0.4291 | micro-F1=0.109 | macro-F1=0.134 | macro-AUC=0.687
Train Loss: 0.4035 | F1 micro/macro: 0.079/0.083 | AUC macro: 0.734 | Mean β: 0.497  ||  Val Loss: 0.4291 | F1 micro/macro: 0.109/0.134 | AUC macro: 0.687 | Mean β: 0.496 | Calib macro-F1: 0.424

=== Epoch 12/20 ===
[Val Calibrated] macro-F1=0.418 | thresholds=[0.05 0.05 0.15 0.2  0.2  0.4 ]
[Val Metrics] Loss=0.4368 | micro-F1=0.119 | macro-F1=0.148 | macro-AUC=0.683
Train Loss: 0.4007 | F1 micro/macro: 0.086/0.093 | AUC macro: 0.739 | Mean β: 0.496  ||  Val Loss: 0.4368 | F1 micro/macro: 0.119/0.148 | AUC macro: 0.683 | Mean β: 0.496 | Calib macro-F1: 0.418

=== Epoch 13/20 ===
[Val Calibrated] macro-F1=0.420 | thresholds=[0.05 0.05 0.15 0.2  0.35 0.4 ]
[Val Metrics] Loss=0.4330 | micro-F1=0.123 | macro-F1=0.150 | macro-AUC=0.688
Train Loss: 0.3978 | F1 micro/macro: 0.089/0.098 | AUC macro: 0.745 | Mean β: 0.496  ||  Val Loss: 0.4330 | F1 micro/macro: 0.123/0.150 | AUC macro: 0.688 | Mean β: 0.496 | Calib macro-F1: 0.420

=== Epoch 14/20 ===
[Val Calibrated] macro-F1=0.416 | thresholds=[0.05 0.05 0.1  0.25 0.25 0.45]
[Val Metrics] Loss=0.4453 | micro-F1=0.126 | macro-F1=0.152 | macro-AUC=0.682
Train Loss: 0.3960 | F1 micro/macro: 0.090/0.101 | AUC macro: 0.749 | Mean β: 0.495  ||  Val Loss: 0.4453 | F1 micro/macro: 0.126/0.152 | AUC macro: 0.682 | Mean β: 0.494 | Calib macro-F1: 0.416

=== Epoch 15/20 ===
[Val Calibrated] macro-F1=0.416 | thresholds=[0.05 0.05 0.1  0.2  0.35 0.3 ]
[Val Metrics] Loss=0.4563 | micro-F1=0.153 | macro-F1=0.177 | macro-AUC=0.683
Train Loss: 0.3932 | F1 micro/macro: 0.094/0.107 | AUC macro: 0.754 | Mean β: 0.494  ||  Val Loss: 0.4563 | F1 micro/macro: 0.153/0.177 | AUC macro: 0.683 | Mean β: 0.494 | Calib macro-F1: 0.416

=== Epoch 16/20 ===
[Val Calibrated] macro-F1=0.417 | thresholds=[0.05 0.05 0.1  0.25 0.3  0.4 ]
[Val Metrics] Loss=0.4540 | micro-F1=0.150 | macro-F1=0.175 | macro-AUC=0.683
Train Loss: 0.3923 | F1 micro/macro: 0.099/0.112 | AUC macro: 0.756 | Mean β: 0.493  ||  Val Loss: 0.4540 | F1 micro/macro: 0.150/0.175 | AUC macro: 0.683 | Mean β: 0.493 | Calib macro-F1: 0.417

=== Epoch 17/20 ===
[Val Calibrated] macro-F1=0.416 | thresholds=[0.05 0.05 0.2  0.25 0.35 0.4 ]
[Val Metrics] Loss=0.4563 | micro-F1=0.151 | macro-F1=0.177 | macro-AUC=0.682
Train Loss: 0.3910 | F1 micro/macro: 0.101/0.117 | AUC macro: 0.758 | Mean β: 0.493  ||  Val Loss: 0.4563 | F1 micro/macro: 0.151/0.177 | AUC macro: 0.682 | Mean β: 0.493 | Calib macro-F1: 0.416

=== Epoch 18/20 ===
[Val Calibrated] macro-F1=0.416 | thresholds=[0.05 0.05 0.2  0.3  0.3  0.4 ]
[Val Metrics] Loss=0.4614 | micro-F1=0.160 | macro-F1=0.185 | macro-AUC=0.682
Train Loss: 0.3901 | F1 micro/macro: 0.101/0.117 | AUC macro: 0.759 | Mean β: 0.493  ||  Val Loss: 0.4614 | F1 micro/macro: 0.160/0.185 | AUC macro: 0.682 | Mean β: 0.493 | Calib macro-F1: 0.416

=== Epoch 19/20 ===
[Val Calibrated] macro-F1=0.416 | thresholds=[0.05 0.05 0.1  0.2  0.35 0.4 ]
[Val Metrics] Loss=0.4600 | micro-F1=0.157 | macro-F1=0.183 | macro-AUC=0.682
Train Loss: 0.3900 | F1 micro/macro: 0.103/0.122 | AUC macro: 0.759 | Mean β: 0.493  ||  Val Loss: 0.4600 | F1 micro/macro: 0.157/0.183 | AUC macro: 0.682 | Mean β: 0.493 | Calib macro-F1: 0.416

=== Epoch 20/20 ===
[Val Calibrated] macro-F1=0.416 | thresholds=[0.05 0.05 0.1  0.25 0.35 0.4 ]
[Val Metrics] Loss=0.4594 | micro-F1=0.157 | macro-F1=0.183 | macro-AUC=0.682
Train Loss: 0.3900 | F1 micro/macro: 0.104/0.123 | AUC macro: 0.759 | Mean β: 0.493  ||  Val Loss: 0.4594 | F1 micro/macro: 0.157/0.183 | AUC macro: 0.682 | Mean β: 0.493 | Calib macro-F1: 0.416

[Saved] Best model to /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/best_mosei_fusion_decoder.pt (select_by=calibrated_macro_f1, val_metric=0.4237, macroAUC=0.687, macroF1=0.134, calibMacroF1=0.424)
[Saved] Per-class thresholds: [0.05 0.05 0.15 0.2  0.25 0.35]
```