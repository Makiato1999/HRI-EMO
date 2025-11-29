# Inference & Evaluation Scripts (MOSEI)

This directory contains the core pipelines for model inference, performance evaluation, and visualization.
It focuses on generating **Quantitative Results** (Tables & Plots) and exporting data for **Qualitative Analysis** (Heatmaps).

## 📂 File Descriptions

| File Name | Type | Description |
| :--- | :--- | :--- |
| **`mosei_infer.ipynb`** | Notebook | **[Launcher]** The recommended entry point. It automates the full pipeline: Inference $\to$ Metrics $\to$ Plotting. |
| **`mosei_eval_infer.py`** | Core Script | **[Inference Engine]** Loads the model, runs inference on Test split, and saves `.npy` predictions and `.pt` attention maps. |
| **`mosei_summary_metrics.py`** | Script | **[Metrics Calculator]** Loads predictions, applies calibrated thresholds, and prints detailed F1/AUC tables. |
| **`mosei_plot_metrics.py`** | Script | **[Plotting Tool]** Generates the **Combined P-R Curve** (for the paper) and other statistical charts. |

---

## 🚀 Quick Start

### Option 1: Using the Notebook (Recommended)
Open **`mosei_infer.ipynb`**. Update the `CKPT_PATH` (e.g., pointing to your v2 model) and run all cells.

### Option 2: Command Line Interface (CLI)
You can also run the steps individually via terminal.

#### Step 1: Run Inference
*Exports predictions and attention maps. Note: `beta_hidden` must match training (64 for v2).*

```bash
# 1. Setup path
%cd /content/HRI-EMO

# 2. Run Inferring (v2 Config)
!PYTHONPATH=. python scripts/infer/mosei_eval_infer.py \
  --ckpt /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/best_mosei_fusion_decoder.pt \
  --index_csv_val ../data/mosei_index_splits.csv \
  --index_csv_test ../data/mosei_index_splits.csv \
  --features_root ../features/mosei/seq_level \
  --out_dir /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs \
  --batch_size 8 \
  --max_len_audio 300 \
  --max_len_text 128 \
  --amp_dtype fp16 \
  --beta_hidden 64 \
  --dump_beta \
  --dump_attn \
  --attn_max_samples 100
```

## 🛠 Key Arguments

If you need to debug the scripts via CLI, here are the key arguments:

* `--dump_attn`: (in `infer.py`) **Must be enabled** to export attention maps for explainability.
* `--beta_hidden`: Must match the training configuration (e.g., `64` for the v2 model).
* `--amp_dtype`: Recommended to use `fp16` for faster inference.

## ✅ Output
```
/content/HRI-EMO
[✓] Loaded hidden dims from meta.json -> audio=74, text=300
[Saved] val -> /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs/val_y_prob.npy, /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs/val_y_true.npy
[Saved] val beta -> /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs/val_beta_mean.npy
[Saved] val attentions (Issue 1 & 2) -> /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs/val_attentions.pt
        Captured 104 samples.
[Saved] test -> /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs/test_y_prob.npy, /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs/test_y_true.npy
[Saved] test beta -> /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs/test_beta_mean.npy
[Saved] test attentions (Issue 1 & 2) -> /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2/infer_outputs/test_attentions.pt
        Captured 104 samples.
```