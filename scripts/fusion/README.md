
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



## New training result
```
!python -m scripts.fusion.train_mosei_fusion_seq_level_decoder \
  --index_csv ../data/mosei_index_splits.csv \
  --audio_dir ../features/mosei/seq_level/audio \
  --text_dir ../features/mosei/seq_level/text \
  --out_dir /content/drive/MyDrive/ColabNotebooks/beta_decoder_project/HRI-EMO-results/mosei_fusion_decoder_v2 \
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
```
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