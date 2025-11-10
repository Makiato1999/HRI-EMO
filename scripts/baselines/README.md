
#### train_concat_baseline_utter_level.py
```bash
python -m scripts.baselines.train_concat_baseline_utter_level \
  --csv data/iemocap_index_splits.csv \
  --audio_dir features/utter_level/audio \
  --text_dir features/utter_level/text \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-4 \
  --out_dir runs/concat_utter_level_baseline

```