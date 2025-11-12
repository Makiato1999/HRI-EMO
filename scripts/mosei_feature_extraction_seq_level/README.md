
#### build_mosei_index_splits.py
```bash
python -m scripts.mosei_feature_extraction_seq_level.build_mosei_index_splits   --data_root data/MOSEI
```

#### extract_text_feats_from_csd.py
```bash
python -m scripts.mosei_feature_extraction_seq_level.extract_text_feats_from_csd   --data_root data/MOSEI   --index_csv data/mosei_index_splits.csv   --out_dir features/mosei/seq_level/text
```

#### extract_audio_feats_from_csd.py
```bash
python -m scripts.mosei_feature_extraction_seq_level.extract_audio_feats_from_csd   --data_root data/MOSEI   --index_csv data/mosei_index_splits.csv   --out_dir features/mosei/seq_level/audio
```