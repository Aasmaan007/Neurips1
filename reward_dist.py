import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import signal

# === Handle Ctrl+C cleanly ===
def signal_handler(sig, frame):
    print("\n[Interrupted] Exiting cleanly.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === CONFIG: update this if your data path changes ===
PKL_PATH = "runs/data/LunarLander-v2__5__2025-05-04_04-01-39__1746311499/task_regression_data.pkl"

if not os.path.isfile(PKL_PATH):
    print(f"[ERROR] File not found: {PKL_PATH}")
    sys.exit(1)

# === Load rewards ===
print(f"Loading φ(s), r, w tuples from: {PKL_PATH}")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

# Extract rewards (2nd element of each tuple)
rewards = np.array([r for (r) in data], dtype=np.float32)

# === Print summary ===
print(f"Loaded {len(rewards)} rewards")
print(f"Min: {rewards.min():.4f}, Max: {rewards.max():.4f}, Mean: {rewards.mean():.4f}, Std: {rewards.std():.4f}")

# === Plot reward distribution ===
plt.figure(figsize=(8, 5))
plt.hist(rewards, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
plt.title("Reward Distribution from φ(s) Training Data")
plt.xlabel("Reward")
plt.ylabel("Density")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()