import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import signal
import sys

# === HANDLE Ctrl+C CLEANLY ===
def signal_handler(sig, frame):
    print("\n[Interrupted] Exiting gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === CONFIG ===
PKL_PATH = "runs/data/LunarLander-v2__data_collection_1__2025-04-26_12-49-15__1745651955/maml_training_data.pkl"

# === Load data ===
if not os.path.exists(PKL_PATH):
    print(f"[ERROR] File not found: {PKL_PATH}")
    sys.exit(1)

with open(PKL_PATH, "rb") as f:
    states = pickle.load(f)

states = np.array(states, dtype=np.float32)
print(f"Loaded {states.shape[0]} states of dimension {states.shape[1]}")

# === PCA ===
pca = PCA(n_components=2)
proj = pca.fit_transform(states)

explained = pca.explained_variance_ratio_
print(f"Explained variance by PC1: {explained[0]*100:.2f}%")
print(f"Explained variance by PC2: {explained[1]*100:.2f}%")
print(f"Total (PC1 + PC2): {explained[:2].sum()*100:.2f}%")

# === Plot ===
plt.figure(figsize=(8, 6))
plt.scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.3, cmap="viridis")
plt.title("2D PCA Projection of MAML States")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()