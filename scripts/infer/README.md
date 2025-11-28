# Inference & Evaluation Scripts

This directory contains the core scripts for model inference and performance evaluation.
It focuses on generating **Quantitative Results** (Tables & Plots), including inference data generation, F1/AUC calculation, and P-R curve plotting.

## 📂 File Structure

### 1. Main Entry Point (Launcher)
* **`mosei_infer.ipynb`**: **[Control Center]**
    * This is the **only file you need to run**.
    * It acts as a pipeline that sequentially triggers the backend scripts below to complete the "Inference -> Evaluation -> Plotting" workflow.

### 2. Backend Scripts
These scripts are called automatically by `mosei_infer.ipynb` (no need to run manually):
* **`mosei_eval_infer.py`**: **[Inference Engine]** Loads model weights, runs inference on Val/Test splits, and saves `.npy` (predictions) and `.pt` (attention maps).
* **`mosei_summary_metrics.py`**: **[Metrics Calculator]** Loads predictions, applies calibrated thresholds from the checkpoint, and prints detailed Micro/Macro F1 & AUC tables.
* **`mosei_plot_metrics.py`** : **[Plotting Tool]** Generates the **Combined P-R Curve (6-in-1 plot)** and other statistical charts for the paper.

---

## 🚀 Workflow

### Step 1: Quantitative Results (Hard Metrics)
1.  Open **`mosei_infer.ipynb`**.
2.  Update the **Configuration** cell:
    * `CKPT_PATH`: Path to your `.pt` checkpoint file.
    * `OUT_DIR`: Directory to save results.
3.  Click **Run All**.
4.  Upon completion, you will find the following in the output directory:
    * `test_summary_metrics.csv` (Metrics Table)
    * `plots/test_combined_PR_curve.png` (Figure for Paper Results)
    * `test_attentions.pt` (Data for Step 2)

### Step 2: Qualitative Results (Explainability)
**Note**: This directory does not handle specific case studies or heatmap visualization.

* Please navigate to the **`notebooks/`** directory in the project root.
* Use the notebook there to load the `test_attentions.pt` generated here and visualize Cross-Modal Alignment & Emotion Attribution heatmaps.

---

## 🛠 Key Arguments

If you need to debug the scripts via CLI, here are the key arguments:

* `--dump_attn`: (in `infer.py`) **Must be enabled** to export attention maps for explainability.
* `--beta_hidden`: Must match the training configuration (e.g., `64` for the v2 model).
* `--amp_dtype`: Recommended to use `fp16` for faster inference.