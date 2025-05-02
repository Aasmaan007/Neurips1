import os
import random
import time
import pickle
from dataclasses import dataclass
from collections import namedtuple

import tyro
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

import gymnasium as gym
from cleanrl.diayn.models import QNetwork

Transition = namedtuple("Transition", ["s", "a", "r", "s_next", "z", "done"])

@dataclass
class Args:
    env_id: str = "LunarLander-v2"
    buffer_path: str = "runs/data/LunarLander-v2__unified_collection_1__2025-04-30_22-30-58__1746032458/offline_q_buffer.pkl"  # path to your offline_q_buffer.pkl
    n_skills: int = 6  # mapped skills: 1 → 0, 2 → 1, etc.
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 5e-6
    total_updates: int = 4000000
    target_update_freq: int = 120
    tau: float = 1
    seed: int = 1
    cuda: bool = True
    wandb_project_name: str = "offline_q_learning"
    wandb_entity: str = None
    exp_name: str = "offline_q_train"
    max_timesteps: int = 1000  # env constraint for QNetwork
    track: bool = True
    gradient_freq: int  = 200

def concat_state_skill(s, z, n_skills):
    z_one_hot = np.zeros(n_skills, dtype=np.float32)
    z_one_hot[z] = 1.0
    return np.concatenate([s, z_one_hot], axis=-1)

def make_env(env_id, seed):
    env = gym.make(env_id)
    return env

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = tyro.cli(Args)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    set_seed(args.seed)

    # Logging
    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}__{timestamp}"

    if args.track:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args), name=run_name)
        wandb.define_metric("iteration")
        wandb.define_metric("per_iteration/*", step_metric="iteration")      
        wandb.config.update(vars(args), allow_val_change=True)

    # Load buffer
    with open(args.buffer_path, "rb") as f:
        raw_data = pickle.load(f)
        np.random.shuffle(raw_data)
        offline_data = [Transition(*t) for t in raw_data]

    print(f"Loaded {len(offline_data)} transitions from {args.buffer_path}")
    # wandb.log({"buffer_size": len(offline_data)})

    # Env for Q-network spec
    env = make_env(args.env_id, args.seed)

    # Q-network setup
    q_net = QNetwork(env, args.n_skills).to(device)
    q_target = QNetwork(env, args.n_skills).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)


    if args.track:
        wandb.watch(
            models=[q_net],
            log="all",          # can also use "gradients" or "parameters"
            log_freq=args.gradient_freq,     # every 1000 backward() calls
            log_graph=False
        )

    # Training loop
    for iteration in range(1, args.total_updates + 1):
        batch = random.sample(offline_data, args.batch_size)
        s_batch = torch.tensor([concat_state_skill(t.s, t.z, args.n_skills) for t in batch], dtype=torch.float32).to(device)
        a_batch = torch.tensor([t.a for t in batch], dtype=torch.long).unsqueeze(1).to(device)
        r_batch = torch.tensor([t.r for t in batch], dtype=torch.float32).to(device)
        s_next_batch = torch.tensor([concat_state_skill(t.s_next, t.z, args.n_skills) for t in batch], dtype=torch.float32).to(device)
        done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32).to(device)

        with torch.no_grad():
           # Action selection using q_net
            next_actions = q_net(s_next_batch).argmax(dim=1, keepdim=True)  # shape (B, 1)
            # Action evaluation using q_target
            next_q_vals_target = q_target(s_next_batch).gather(1, next_actions).squeeze()  # shape (B,)
            td_target = r_batch + args.gamma * next_q_vals_target * (1.0 - done_batch)


        q_vals = q_net(s_batch).gather(1, a_batch).squeeze()
        loss = F.mse_loss(q_vals, td_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % args.target_update_freq == 0:
            for target_network_param, q_network_param in zip(q_target.parameters(), q_net.parameters()):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                )
     
        if(iteration % 200 == 0):
            print(f"Iteration {iteration} completed")
            wandb.log({
                    "per_iteration/td_loss": float(loss.item()),
                    "per_iteration/qvals": float(q_vals.mean().item()),
                    "per_iteration/td_target": float(td_target.mean().item()),
                    "per_iteration/next_qvals": float(next_q_vals_target.mean().item()),
                    "per_iteration/rbatch": float(r_batch.mean().item()),
                     
                } , step = int(iteration))

    # Save
    save_path = f"runs/checkpoints/q_offline/{run_name}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(q_net.state_dict(), os.path.join(save_path, "q_network_offline.pth"))
    print(f"Model saved to {save_path}/q_network_offline.pth")

if __name__ == "__main__":
    main()