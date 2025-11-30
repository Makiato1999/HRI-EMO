# IEMOCAP Data Processor

This script preprocesses the raw IEMOCAP dataset, aligns audio paths, text transcripts, and emotion labels, and generates the final CSV index files required for model training.

## ⚙️ Requirements

* Python 3.x
* `pandas`
* `tqdm` (for progress bars)

## 🚀 Execution

1.  **Ensure Data Path:** Make sure the raw IEMOCAP full release data is accessible via the path defined in the script's `BASE` variable (`../data/IEMOCAP_full_release`).
2.  **Run the script:**

    ```bash
    python iemocap_data_processor.py
    ```

## 📊 Output Files

The script generates the following files in the `../data` directory:

1.  `iemocap_index.csv`: The clean, aligned index containing utterance ID, session, audio path, text, and the 6 core emotion labels.
2.  `iemocap_index_splits.csv`: The final index file, which includes an additional `split` column (`train`, `val`, `test`) based on the standard Session-based split (Session 5 for test, Session 4 for validation).