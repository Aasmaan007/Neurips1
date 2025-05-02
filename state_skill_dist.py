import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import signal

# === Ctrl+C handling ===
def signal_handler(sig, frame):
    print("\n[Interrupted] Exiting.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === CONFIG ===
RUN_NAME = "LunarLander-v2__unified_collection_1__2025-05-02_00-57-07__1746127627"
BASE_DIR = f"runs/data/{RUN_NAME}/per_skill_states"
ALLOWED_SKILLS = [1, 2, 5, 6, 11, 22]

# === Load states and labels ===
all_states = []
all_labels = []

for z in ALLOWED_SKILLS:
    path = os.path.join(BASE_DIR, f"skill_{z}.pkl")
    if not os.path.isfile(path):
        continue
    with open(path, "rb") as f:
        states = pickle.load(f)
    all_states.extend(states)
    all_labels.extend([z] * len(states))

all_states = np.array(all_states, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int32)

# === PCA ===
pca = PCA(n_components=2)
proj = pca.fit_transform(all_states)
explained = pca.explained_variance_ratio_

print(f"Loaded {len(all_states)} states")
print(f"PC1: {explained[0]*100:.2f}%, PC2: {explained[1]*100:.2f}%")

# === Plotting ===
plt.figure(figsize=(10, 8))
cmap = ListedColormap(plt.get_cmap("tab20").colors[:len(ALLOWED_SKILLS)])
norm_labels = np.array([ALLOWED_SKILLS.index(l) for l in all_labels])  # index into cmap

scatter = plt.scatter(
    proj[:, 0], proj[:, 1],
    c=norm_labels,
    cmap=cmap,
    s=3,
    alpha=0.6
)

# Build custom legend with actual skill numbers
handles = []
for i, z in enumerate(ALLOWED_SKILLS):
    handles.append(plt.Line2D([], [], marker='o', color=cmap(i), linestyle='', label=f"Skill {z}"))

plt.legend(handles=handles, title="Skill", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("State Distribution (Unified Skills View)")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()