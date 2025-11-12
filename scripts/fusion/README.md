
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
!python scripts/fusion/train_mosei_fusion_seq_level_decoder.py \
  --index_csv ../data/mosei_index_splits.csv \
  --audio_dir ../features/mosei/seq_level/audio \
  --text_dir ../features/mosei/seq_level/text \
  --epochs 5 \
  --batch_size 4 \
  --d_model 128 \
  --n_heads 4 \
  --num_layers_fusion 1 \
  --num_layers_decoder 1 \
  --beta_hidden 64 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --max_train_samples 10000 \
  --out_dir /content/drive/MyDrive/HRI-EMO-results/mosei_fusion_decoder_colab

```