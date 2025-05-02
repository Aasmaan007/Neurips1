import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import tyro
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gymnasium as gym

from cleanrl.diayn.models import QNetwork

@dataclass
class Args:
    run_name: str = "LunarLander-v2__unified_collection_1__2025-05-01_23-50-37__1746123637"
    model_path: str = "runs/checkpoints/qtargetmaml/LunarLander-v2__q_online__1__2025-05-01_15-57-00__1746095220/latest.pth"
    env_id: str = "LunarLander-v2"
    n_skills_selected: int = 6
    allowed_skills: list = (1, 2, 5, 6, 11, 22)
    max_states_per_skill: int = 20000
    cuda: bool = True

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # === Load QNetwork ===
    env = gym.make(args.env_id)
    qnet = QNetwork(env, args.n_skills_selected).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    qnet.load_state_dict(checkpoint["q_network_state_dict"])
    qnet.eval()

    def concat_state_latent(states, z, n_skills):
        z_vec = np.zeros((states.shape[0], n_skills), dtype=np.float32)
        z_vec[:, z] = 1.0
        return np.concatenate([states, z_vec], axis=-1)

    # === Load per-skill states ===
    skill_states = {}
    base_dir = f"runs/data/{args.run_name}/per_skill_states"
    for z in args.allowed_skills:
        path = os.path.join(base_dir, f"skill_{z}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            np.random.shuffle(data)
            skill_states[z] = np.array(data[:args.max_states_per_skill], dtype=np.float32)

    # === Compute Q-values ===
    results = {}
    true_skill_to_model_idx = {s: i for i, s in enumerate(args.allowed_skills)}  #22 ->5

    for z in args.allowed_skills:
        in_states = skill_states[z]
        out_states = np.concatenate([skill_states[k] for k in args.allowed_skills if k != z])
        np.random.shuffle(out_states)
        out_states = out_states[:len(in_states)]

        in_input = torch.tensor(concat_state_latent(in_states, true_skill_to_model_idx[z], args.n_skills_selected), dtype=torch.float32).to(device)
        out_input = torch.tensor(concat_state_latent(out_states, true_skill_to_model_idx[z], args.n_skills_selected), dtype=torch.float32).to(device)

        with torch.no_grad():
            q_in = qnet(in_input).max(dim=1)[0].cpu().numpy()
            q_out = qnet(out_input).max(dim=1)[0].cpu().numpy()

        results[z] = {
            "in_skill_qvals": q_in,
            "out_skill_qvals": q_out
        }

    # === Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, z in enumerate(args.allowed_skills):
        df = pd.DataFrame({
            "Q-Value": np.concatenate([results[z]["in_skill_qvals"], results[z]["out_skill_qvals"]]),
            "Context": ["In-Skill"] * len(results[z]["in_skill_qvals"]) + ["Out-of-Skill"] * len(results[z]["out_skill_qvals"])
        })
        sns.histplot(data=df, x="Q-Value", hue="Context", kde=True, bins=30, ax=axes[i], palette=["green", "red"])
        axes[i].set_title(f"Skill {z}")
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
    

    with open(f"runs/data/{args.run_name}/maml_training_data.pkl", "rb") as f:
        all_states = np.array(pickle.load(f), dtype=np.float32)
        np.random.shuffle(all_states)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, z in enumerate(args.allowed_skills):
        input_tensor = torch.tensor(concat_state_latent(all_states, true_skill_to_model_idx[z], args.n_skills_selected), dtype=torch.float32).to(device)
        with torch.no_grad():
            qvals = qnet(input_tensor).max(dim=1)[0].cpu().numpy()
        
        sns.histplot(qvals, bins=40, kde=True, ax=axes[i], color="blue")
        axes[i].set_title(f"Q-Value Dist — Skill {z}")
        axes[i].set_xlabel("Q-Value")
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()