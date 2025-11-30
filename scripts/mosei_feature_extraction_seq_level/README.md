# MOSEI Data Processor

This directory contains scripts for managing MOSEI computational sequences (CSD), extracting raw features, verifying data integrity, and building the final dataset index file (`.csv`).

The primary goal is to prepare sequence-level features and the corresponding multi-label emotion and sentiment annotations for model training.

## 📜 File Descriptions

| File Name | Description | Purpose |
| :--- | :--- | :--- |
| **`build_mosei_index_splits.py`** | **Index Builder (Labels & Splits)** | **Core script.** Loads the raw labels from `CMU_MOSEI_Labels.csd`, aligns them with standard video IDs, applies the official train/val/test splits, and exports the final annotation CSV. |
| `extract_text_feats_from_csd.py` | **Text Feature Extractor** | Processes the raw text computational sequence data (`.csd`) to extract features (e.g., word embeddings) and saves them as `.npy` files. |
| `extract_audio_feats_from_csd.py` | **Audio Feature Extractor** | Processes the raw audio computational sequence data (`.csd`) to extract time-aligned features and saves them as `.npy` files. |
| `check_feature_integrity.py` | **Integrity Check** | Verifies that all expected feature files (audio, text, video) exist for every entry in the final index CSV and checks for sequence length consistency. |

## 🛠️ Data Processing Pipeline (Feature Extraction)

This pipeline assumes the raw MOSEI CSD files (including `CMU_MOSEI_Labels.csd`) have already been downloaded to the `data/MOSEI` directory.

## 🚀 Quick Start

### Step 1: Build the Index and Splits (Labels)

This step creates the master CSV file (`data/mosei_index_splits.csv`) containing all segment IDs, labels, and the official train/val/test split assignments.

```bash
python -m scripts.mosei_feature_extraction_seq_level.build_mosei_index_splits \
    --data_root data/MOSEI
```

### Step 2: Extract Text Features

This step reads the raw text CSD data, aligns it using the index created in Step 1, and saves the extracted features into the specified output directory.

```bash
python -m scripts.mosei_feature_extraction_seq_level.extract_text_feats_from_csd \
    --data_root data/MOSEI \
    --index_csv data/mosei_index_splits.csv \
    --out_dir features/mosei/seq_level/text
```

### Step 3: Extract Audio Features

This step reads the raw audio CSD data, aligns it using the index created in Step 1, and saves the extracted features into the specified output directory.

```bash
python -m scripts.mosei_feature_extraction_seq_level.extract_audio_feats_from_csd \
    --data_root data/MOSEI \
    --index_csv data/mosei_index_splits.csv \
    --out_dir features/mosei/seq_level/audio
```