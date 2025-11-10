
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
  --out_dir runs/fusion_seq_level_default
```