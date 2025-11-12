# tools/export_per_class_metrics.py
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

y_true = np.load("outputs/val_y_true.npy")     # [N,6] 0/1
y_prob = np.load("outputs/val_y_prob.npy")     # [N,6] sigmoid probs
thr_cal = np.load("outputs/best_thresholds.npy")  # [6]

rows = []
for k, name in enumerate(["happy","sad","anger","fear","disgust","surprise"]):
    yk, pk = y_true[:,k], y_prob[:,k]
    auc = roc_auc_score(yk, pk)
    ap  = average_precision_score(yk, pk)
    f1_raw = f1_score(yk, pk>=0.5, zero_division=0)
    f1_cal = f1_score(yk, pk>=thr_cal[k], zero_division=0)
    prev = float(yk.mean())
    rows.append(dict(class_=name, prevalence=prev, auc=auc, auprc=ap, f1_raw=f1_raw, f1_cal=f1_cal, thr=thr_cal[k]))

df = pd.DataFrame(rows).round(4)
df.to_csv("outputs/metrics_per_class.csv", index=False)
print(df)
