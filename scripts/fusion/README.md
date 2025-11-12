
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

[Dataset] Final size: 16327 samples
[Dataset] Final size: 1871 samples
[pos_weight] {'emo_happy': np.float32(0.87), 'emo_sad': np.float32(2.82), 'emo_anger': np.float32(3.63), 'emo_fear': np.float32(8.94), 'emo_disgust': np.float32(4.53), 'emo_surprise': np.float32(11.27)}

=== Epoch 1/20 ===
[Val Calibrated] macro-F1=0.403 | thresholds=[0.05 0.05 0.1  0.1  0.05 0.1 ]
[Val Metrics] Loss=0.4239 | micro-F1=0.002 | macro-F1=0.003 | macro-AUC=0.645
Train Loss: 0.4915 | F1 micro/macro: 0.084/0.075 | AUC macro: 0.555 | Mean β: 0.495  ||  Val Loss: 0.4239 | F1 micro/macro: 0.002/0.003 | AUC macro: 0.645 | Mean β: 0.493 | Calib macro-F1: 0.403

=== Epoch 2/20 ===
[Val Calibrated] macro-F1=0.423 | thresholds=[0.05 0.15 0.2  0.15 0.25 0.2 ]
[Val Metrics] Loss=0.4156 | micro-F1=0.018 | macro-F1=0.024 | macro-AUC=0.684
Train Loss: 0.4310 | F1 micro/macro: 0.035/0.038 | AUC macro: 0.662 | Mean β: 0.490  ||  Val Loss: 0.4156 | F1 micro/macro: 0.018/0.024 | AUC macro: 0.684 | Mean β: 0.484 | Calib macro-F1: 0.423

=== Epoch 3/20 ===
[Val Calibrated] macro-F1=0.423 | thresholds=[0.1  0.1  0.1  0.15 0.25 0.15]
[Val Metrics] Loss=0.4044 | micro-F1=0.029 | macro-F1=0.036 | macro-AUC=0.690
Train Loss: 0.4215 | F1 micro/macro: 0.052/0.056 | AUC macro: 0.692 | Mean β: 0.476  ||  Val Loss: 0.4044 | F1 micro/macro: 0.029/0.036 | AUC macro: 0.690 | Mean β: 0.462 | Calib macro-F1: 0.423

=== Epoch 4/20 ===
[Val Calibrated] macro-F1=0.423 | thresholds=[0.05 0.1  0.15 0.15 0.25 0.15]
[Val Metrics] Loss=0.4053 | micro-F1=0.049 | macro-F1=0.059 | macro-AUC=0.694
Train Loss: 0.4120 | F1 micro/macro: 0.067/0.070 | AUC macro: 0.716 | Mean β: 0.460  ||  Val Loss: 0.4053 | F1 micro/macro: 0.049/0.059 | AUC macro: 0.694 | Mean β: 0.446 | Calib macro-F1: 0.423

=== Epoch 5/20 ===
[Val Calibrated] macro-F1=0.422 | thresholds=[0.05 0.1  0.1  0.15 0.15 0.1 ]
[Val Metrics] Loss=0.4127 | micro-F1=0.082 | macro-F1=0.094 | macro-AUC=0.693
Train Loss: 0.4046 | F1 micro/macro: 0.082/0.087 | AUC macro: 0.733 | Mean β: 0.444  ||  Val Loss: 0.4127 | F1 micro/macro: 0.082/0.094 | AUC macro: 0.693 | Mean β: 0.436 | Calib macro-F1: 0.422

=== Epoch 6/20 ===
[Val Calibrated] macro-F1=0.417 | thresholds=[0.05 0.05 0.1  0.15 0.25 0.1 ]
[Val Metrics] Loss=0.4146 | micro-F1=0.057 | macro-F1=0.069 | macro-AUC=0.689
Train Loss: 0.3950 | F1 micro/macro: 0.098/0.108 | AUC macro: 0.753 | Mean β: 0.432  ||  Val Loss: 0.4146 | F1 micro/macro: 0.057/0.069 | AUC macro: 0.689 | Mean β: 0.419 | Calib macro-F1: 0.417

=== Epoch 7/20 ===
[Val Calibrated] macro-F1=0.414 | thresholds=[0.05 0.1  0.1  0.15 0.2  0.2 ]
[Val Metrics] Loss=0.4193 | micro-F1=0.073 | macro-F1=0.090 | macro-AUC=0.679
Train Loss: 0.3829 | F1 micro/macro: 0.122/0.149 | AUC macro: 0.774 | Mean β: 0.422  ||  Val Loss: 0.4193 | F1 micro/macro: 0.073/0.090 | AUC macro: 0.679 | Mean β: 0.404 | Calib macro-F1: 0.414

=== Epoch 8/20 ===
[Val Calibrated] macro-F1=0.414 | thresholds=[0.05 0.15 0.1  0.1  0.2  0.2 ]
[Val Metrics] Loss=0.4397 | micro-F1=0.121 | macro-F1=0.142 | macro-AUC=0.675
Train Loss: 0.3699 | F1 micro/macro: 0.154/0.197 | AUC macro: 0.794 | Mean β: 0.410  ||  Val Loss: 0.4397 | F1 micro/macro: 0.121/0.142 | AUC macro: 0.675 | Mean β: 0.403 | Calib macro-F1: 0.414

=== Epoch 9/20 ===
[Val Calibrated] macro-F1=0.410 | thresholds=[0.05 0.1  0.05 0.05 0.1  0.15]
[Val Metrics] Loss=0.4560 | micro-F1=0.102 | macro-F1=0.125 | macro-AUC=0.669
Train Loss: 0.3566 | F1 micro/macro: 0.179/0.233 | AUC macro: 0.815 | Mean β: 0.400  ||  Val Loss: 0.4560 | F1 micro/macro: 0.102/0.125 | AUC macro: 0.669 | Mean β: 0.374 | Calib macro-F1: 0.410

=== Epoch 10/20 ===
[Val Calibrated] macro-F1=0.416 | thresholds=[0.05 0.05 0.1  0.05 0.3  0.1 ]
[Val Metrics] Loss=0.4693 | micro-F1=0.097 | macro-F1=0.121 | macro-AUC=0.668
Train Loss: 0.3415 | F1 micro/macro: 0.209/0.278 | AUC macro: 0.836 | Mean β: 0.389  ||  Val Loss: 0.4693 | F1 micro/macro: 0.097/0.121 | AUC macro: 0.668 | Mean β: 0.368 | Calib macro-F1: 0.416

=== Epoch 11/20 ===
[Val Calibrated] macro-F1=0.408 | thresholds=[0.05 0.05 0.05 0.05 0.15 0.05]
[Val Metrics] Loss=0.5057 | micro-F1=0.119 | macro-F1=0.140 | macro-AUC=0.662
Train Loss: 0.3291 | F1 micro/macro: 0.240/0.320 | AUC macro: 0.853 | Mean β: 0.381  ||  Val Loss: 0.5057 | F1 micro/macro: 0.119/0.140 | AUC macro: 0.662 | Mean β: 0.370 | Calib macro-F1: 0.408

=== Epoch 12/20 ===
[Val Calibrated] macro-F1=0.408 | thresholds=[0.05 0.05 0.05 0.05 0.15 0.15]
[Val Metrics] Loss=0.5132 | micro-F1=0.141 | macro-F1=0.163 | macro-AUC=0.661
Train Loss: 0.3182 | F1 micro/macro: 0.262/0.350 | AUC macro: 0.866 | Mean β: 0.377  ||  Val Loss: 0.5132 | F1 micro/macro: 0.141/0.163 | AUC macro: 0.661 | Mean β: 0.368 | Calib macro-F1: 0.408

=== Epoch 13/20 ===
[Val Calibrated] macro-F1=0.403 | thresholds=[0.05 0.05 0.05 0.05 0.35 0.1 ]
[Val Metrics] Loss=0.5388 | micro-F1=0.121 | macro-F1=0.145 | macro-AUC=0.657
Train Loss: 0.3080 | F1 micro/macro: 0.282/0.378 | AUC macro: 0.878 | Mean β: 0.373  ||  Val Loss: 0.5388 | F1 micro/macro: 0.121/0.145 | AUC macro: 0.657 | Mean β: 0.357 | Calib macro-F1: 0.403

=== Epoch 14/20 ===
[Val Calibrated] macro-F1=0.401 | thresholds=[0.05 0.05 0.1  0.2  0.15 0.15]
[Val Metrics] Loss=0.5467 | micro-F1=0.127 | macro-F1=0.146 | macro-AUC=0.653
Train Loss: 0.2984 | F1 micro/macro: 0.305/0.406 | AUC macro: 0.889 | Mean β: 0.368  ||  Val Loss: 0.5467 | F1 micro/macro: 0.127/0.146 | AUC macro: 0.653 | Mean β: 0.355 | Calib macro-F1: 0.401

=== Epoch 15/20 ===
[Val Calibrated] macro-F1=0.403 | thresholds=[0.05 0.05 0.05 0.05 0.15 0.15]
[Val Metrics] Loss=0.5554 | micro-F1=0.130 | macro-F1=0.155 | macro-AUC=0.652
Train Loss: 0.2908 | F1 micro/macro: 0.325/0.431 | AUC macro: 0.897 | Mean β: 0.364  ||  Val Loss: 0.5554 | F1 micro/macro: 0.130/0.155 | AUC macro: 0.652 | Mean β: 0.348 | Calib macro-F1: 0.403

=== Epoch 16/20 ===
[Val Calibrated] macro-F1=0.402 | thresholds=[0.05 0.05 0.05 0.1  0.2  0.05]
[Val Metrics] Loss=0.5798 | micro-F1=0.133 | macro-F1=0.156 | macro-AUC=0.650
Train Loss: 0.2844 | F1 micro/macro: 0.340/0.450 | AUC macro: 0.905 | Mean β: 0.359  ||  Val Loss: 0.5798 | F1 micro/macro: 0.133/0.156 | AUC macro: 0.650 | Mean β: 0.341 | Calib macro-F1: 0.402

=== Epoch 17/20 ===
[Val Calibrated] macro-F1=0.393 | thresholds=[0.05 0.05 0.05 0.05 0.1  0.05]
[Val Metrics] Loss=0.5944 | micro-F1=0.118 | macro-F1=0.142 | macro-AUC=0.646
Train Loss: 0.2799 | F1 micro/macro: 0.348/0.460 | AUC macro: 0.910 | Mean β: 0.357  ||  Val Loss: 0.5944 | F1 micro/macro: 0.118/0.142 | AUC macro: 0.646 | Mean β: 0.342 | Calib macro-F1: 0.393

=== Epoch 18/20 ===
[Val Calibrated] macro-F1=0.394 | thresholds=[0.05 0.05 0.05 0.1  0.1  0.05]
[Val Metrics] Loss=0.6044 | micro-F1=0.129 | macro-F1=0.151 | macro-AUC=0.647
Train Loss: 0.2771 | F1 micro/macro: 0.355/0.468 | AUC macro: 0.913 | Mean β: 0.356  ||  Val Loss: 0.6044 | F1 micro/macro: 0.129/0.151 | AUC macro: 0.647 | Mean β: 0.340 | Calib macro-F1: 0.394

=== Epoch 19/20 ===
[Val Calibrated] macro-F1=0.394 | thresholds=[0.05 0.05 0.05 0.1  0.1  0.1 ]
[Val Metrics] Loss=0.6039 | micro-F1=0.125 | macro-F1=0.145 | macro-AUC=0.645
Train Loss: 0.2750 | F1 micro/macro: 0.362/0.475 | AUC macro: 0.914 | Mean β: 0.356  ||  Val Loss: 0.6039 | F1 micro/macro: 0.125/0.145 | AUC macro: 0.645 | Mean β: 0.340 | Calib macro-F1: 0.394

=== Epoch 20/20 ===
[Val Calibrated] macro-F1=0.394 | thresholds=[0.05 0.05 0.05 0.1  0.1  0.1 ]
[Val Metrics] Loss=0.6074 | micro-F1=0.126 | macro-F1=0.147 | macro-AUC=0.645
Train Loss: 0.2734 | F1 micro/macro: 0.365/0.478 | AUC macro: 0.916 | Mean β: 0.356  ||  Val Loss: 0.6074 | F1 micro/macro: 0.126/0.147 | AUC macro: 0.645 | Mean β: 0.340 | Calib macro-F1: 0.394

[Saved] Best model to /content/drive/MyDrive/HRI-EMO-results/mosei_fusion_decoder_small/best_mosei_fusion_decoder.pt (select_by=macro_auc, val_metric=0.6937, macroAUC=0.694, macroF1=0.059, calibMacroF1=0.423)
[Saved] Per-class thresholds: [0.05 0.1  0.15 0.15 0.25 0.15]
