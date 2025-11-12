# tools/plot_training_curves.py
import pandas as pd, matplotlib.pyplot as plt, os
os.makedirs("figs", exist_ok=True)
log = pd.read_csv("outputs/train_log.csv")  # 你训练时顺手写出的日志

plt.figure(); log.plot(x="epoch", y=["train_loss","val_loss"]); plt.savefig("figs/loss.png", dpi=200)
plt.figure(); log.plot(x="epoch", y=["val_auc"]); plt.savefig("figs/val_auc.png", dpi=200)
plt.figure(); log.plot(x="epoch", y=["mean_beta"]); plt.savefig("figs/mean_beta.png", dpi=200)
