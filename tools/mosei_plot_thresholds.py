# tools/plot_thresholds.py
import numpy as np, matplotlib.pyplot as plt
thr = np.load("outputs/best_thresholds.npy")
names = ["happy","sad","anger","fear","disgust","surprise"]
plt.figure()
plt.bar(names, thr)
plt.ylabel("Optimal threshold")
plt.ylim(0, 0.5)
plt.tight_layout()
plt.savefig("figs/thresholds.png", dpi=200)
