import os
import random
import time
from dataclasses import dataclass
import tyro

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
import torch.nn.functional as F
from gymnasium import spaces

from cleanrl.diayn.models import Discriminator, QNetwork
import wandb
import pickle

@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 150000
    max_timesteps: int = 500
    n_skills: int = 25
    model_path: str = "runs/checkpoints/diayn/LunarLander-v2__diayn__1__2025-04-25_22-19-35__1745599775/latest.pth"
    start_e: float = 1
    end_e: float = 0.35
    exploration_fraction: float = 0.1
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    wandb_project_name: str = "data_collection"
    wandb_entity: str = None
    track: bool = True
    """wandb tracking"""
    exp_name: str = "data_collection"
    torch_deterministic: bool = True

def concat_state_latent(s, z, n_skills):
    z_one_hot = np.zeros(n_skills, dtype=np.float32)
    z_one_hot[z] = 1.0
    return np.concatenate([s, z_one_hot], axis=-1)

def make_env(env_id, seed, max_timesteps):
    env = gym.make(env_id)
    env = TimeLimit(env, max_timesteps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# === Main Starts ===
if __name__ == "__main__":
        
        args = tyro.cli(Args)
        timestamp = int(time.time())
        run_name = f"{args.env_id}__{args.exp_name}_{args.seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}__{timestamp}"
        


         # W&B Tracking  if args.track:
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name = run_name
            )
            wandb.define_metric("episode")
            wandb.define_metric("episodic/*", step_metric="episode")      
            wandb.config.update(vars(args), allow_val_change=True)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        print("-----------------------------------------------------------------------------------------------------------------------------------------------")
        print("Using device:", device)

        env = make_env(args.env_id, args.seed, args.max_timesteps)

           
        # Load models
       
        q_net = QNetwork(env, args.n_skills).to(device)
        discriminator = Discriminator(env.observation_space.shape[0], args.n_skills).to(device)

        checkpoint = torch.load(args.model_path)
        q_net.load_state_dict(checkpoint["q_network_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

       

        # Replay-like buffer for φ(s)
        phi_training_data = []
        maml_training_data = []

        global_step = 0
        episode = 0

        while global_step < args.total_timesteps:
            z = np.random.choice(args.n_skills)
            obs, _ = env.reset(seed=args.seed + episode)
            obs_aug = concat_state_latent(obs, z, args.n_skills)
            episode += 1

            for _ in range(args.max_timesteps+5):
                epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
                
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_vals = q_net(torch.tensor(obs_aug, dtype=torch.float32).unsqueeze(0).to(device))
                        action = torch.argmax(q_vals, dim=1).item()

                next_obs, _, terminated, truncated, _ = env.step(action)

                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    logits = discriminator(next_obs_tensor.unsqueeze(0))  # shape: [1, n_skills]
                    q_zs = F.softmax(logits , dim = -1)
                    q_zs_clamped = torch.clamp(q_zs,min = 1e-6)
                    logq_zs = torch.log(q_zs_clamped)
                    logpz = torch.tensor(1.0 / args.n_skills + 1e-6).log().to(device)  # scalar
                    # r = (logq_zs[z] - logpz).item()
                
                for z_idx in range(args.n_skills):
                    r = (logq_zs[0,z_idx] - logpz).item()
                    w = discriminator.q.weight[z_idx].detach().cpu().numpy()
                    w = w / (np.linalg.norm(w) + 1e-8)
                    phi_training_data.append((next_obs.copy(),r, w))
                
                maml_training_data.append(obs.copy())

                obs = next_obs
                obs_aug = concat_state_latent(obs, z, args.n_skills)
                global_step += 1

                if terminated or truncated:
                    break
            if(args.track):
                wandb.log({
                    "episodic/len_SF_data": float(len(phi_training_data)),
                    "episodic/len_MAML_data": float(len(maml_training_data)),
                    "episodic/global_steps": float(global_step),
                },step = int(episode))

        env.close()
        model_dir = f"runs/data/{run_name}"
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "phi_training_data.pkl"), "wb") as f:
            pickle.dump(phi_training_data, f)
        with open(os.path.join(model_dir, "maml_training_data.pkl"), "wb") as f:
            pickle.dump(maml_training_data, f)

        print(f"Saved φ(s) training data to {os.path.join(model_dir, 'phi_training_data.pkl')}")
        print(f"Saved MAML training data to {os.path.join(model_dir, 'maml_training_data.pkl')}")
        print(f"Collected {len(phi_training_data)} (s, z, r, w) tuples for φ(s) training.")