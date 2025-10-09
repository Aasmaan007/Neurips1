import torch
import gymnasium as gym
from cleanrl.diayn.models import QNetwork
import gym_maze
import time

# --- User settings ---
env_id = "maze-random-10x10-plus-v0"
model_path = "runs/checkpoints/qtargetmaml/maze-random-10x10-plus-v0__q_online__1__2025-07-07_20-41-56__1751901116/latest.pth"
n_skills_selected = 6  # set to what your QNetwork expects
skill_idx = 0         # which skill to use (0 to n_skills_selected-1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------

# Create environment
env = gym.make(env_id)
obs, _ = env.reset(seed=42)

# Load QNetwork
q_network = QNetwork(env, n_skills_selected)
checkpoint = torch.load(model_path)
q_network.load_state_dict(checkpoint["q_network_state_dict"])
q_network.to(device)
q_network.eval()

# Prepare initial observation (concatenate skill one-hot)
def concat_state_latent(s, z, n_skills):
    import numpy as np
    z_one_hot = np.zeros(n_skills, dtype=np.float32)
    z_one_hot[z] = 1.0
    return np.concatenate([s, z_one_hot], axis=-1)

obs = concat_state_latent(obs, skill_idx, n_skills_selected)

done = False
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        action = torch.argmax(q_network(obs_tensor), dim=1).item()
        print(obs, action)
    next_obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # <-- This will show the maze in your terminal
    time.sleep(1)
    obs = concat_state_latent(next_obs, skill_idx, n_skills_selected)
    done = terminated or truncated

env.close()