import os
import random
import time
from dataclasses import dataclass
import sys

import minigrid
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from cleanrl.diayn.models import Discriminator, QNetwork  # <<<<<<<< changed import
from cleanrl.diayn.utils import train_dqn_online
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from collections import defaultdict
import pickle

# --- MiniGrid observation extraction utility ---
def extract_obs(obs):
    if isinstance(obs, dict) and "image" in obs:
        return obs["image"].flatten().astype(np.float32)
    else:
        return np.asarray(obs, dtype=np.float32)

@dataclass
class Args:
    exp_name: str = "q_online"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "q_online"
    wandb_entity: str = None
    capture_video: bool = False
    env_id: str = "MiniGrid-Unlock-v0"  # Default to MiniGrid!
    total_timesteps: int = 100000
    max_timesteps: int = 1000
    learning_rate_qnet: float = 6e-5
    num_env: int = 1
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 1
    target_network_frequency: int = 750
    batch_size: int = 64
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.2
    learning_starts: int = 10000
    n_skills_selected: int = 6
    n_skills_total: int = 25
    model_save: int  = 250000
    gradient_freq: int  = 1000
    train_frequency: int = 6
    rewardclipping: bool = True
    ddqn: bool = True
    reward_history_path: str = "runs/checkpoints/qtargetmaml/MiniGrid-Unlock-v0__q_online__1__2025-06-14_14-31-01__1749891661/intrinsic_rewards_history.pkl"

def concat_state_latent(s, z, n_skills):
    z_one_hot = np.zeros(n_skills, dtype=np.float32)
    z_one_hot[z] = 1.0
    return np.concatenate([s, z_one_hot], axis=-1)

def make_env(env_id, seed, idx, capture_video, run_name, max_timesteps):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = TimeLimit(env, max_timesteps)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            env = TimeLimit(env, max_timesteps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_env == 1, "vectorized env are not supported at the moment"
    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}__{timestamp}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("episode")
        wandb.define_metric("episodic/*", step_metric="episode")      
        wandb.config.update(vars(args), allow_val_change=True)

    writer = SummaryWriter(f"runs/train/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 1. Load the pickle file
    with open(args.reward_history_path, "rb") as f:
        intrinsic_rewards_history = pickle.load(f)  # This will be a list of floats
    
    intrinsic_rewards_arr = np.array(intrinsic_rewards_history)

    # --- seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("-----------------------------------------------------------------------------------------------------------------------------------------------")
    print("Using device:", device)

    env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name, args.max_timesteps)()

    # --- Handle flattened obs shape for MiniGrid ---
    if isinstance(env.observation_space, spaces.Dict):
        image_shape = env.observation_space["image"].shape
        obs_dim = np.prod(image_shape)
    else:
        obs_dim = np.prod(env.observation_space.shape)
    obs_shape = obs_dim + args.n_skills_selected
    augmented_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    q_network = QNetwork(obs_dim, env.action_space.n, args.n_skills_selected).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate_qnet)
    target_network = QNetwork(obs_dim, env.action_space.n, args.n_skills_selected).to(device)
    target_network.load_state_dict(q_network.state_dict())

    discriminator = Discriminator(obs_dim, args.n_skills_total).to(device)

    if args.track:
        wandb.watch(
            models=[q_network],
            log="all",
            log_freq=args.gradient_freq,
            log_graph=False
        )

    rb = ReplayBuffer(
        args.buffer_size,
        augmented_obs_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    action_space = env.action_space
    start_time = time.time()
    global_step = 0
    episode = 0
    allowed_skills = [1, 2, 5, 6, 11, 22]
    model_idx_to_true_skill = {i: s for i, s in enumerate(allowed_skills)}
    true_skill_to_model_idx = {s: i for i, s in enumerate(allowed_skills)}  #22 ->5

    while global_step < args.total_timesteps:
        # --- Sample skill ---
        z_true = np.random.choice(allowed_skills)
        z_model = true_skill_to_model_idx[z_true]
        state_raw, _ = env.reset(seed=args.seed + episode)
        state = extract_obs(state_raw)
        state = concat_state_latent(state, z_model, args.n_skills_selected)
        episode_reward = 0
        logq_zses = []

        for steps in range(args.max_timesteps+5):
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            if random.random() < epsilon:
                action = action_space.sample()
            else:
                q_values = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=1).item()

            next_state_raw, reward, termination, truncation, info = env.step(action)
            next_state = extract_obs(next_state_raw)
            next_state_aug = concat_state_latent(next_state, z_model, args.n_skills_selected)
            rb.add(
                np.array([state]),
                np.array([next_state_aug]),
                np.array([action]),
                np.array([reward]),
                np.array([termination]),
                [info]
            )

            state = next_state_aug
            episode_reward += reward
            global_step += 1

            # --- training ---
            if global_step > args.learning_starts:
                if(global_step % args.train_frequency == 0):
                    data = rb.sample(args.batch_size)
                    loss, old_val , intrinsic_rewards , logqz , td_target , bootstrapping = train_dqn_online(
                        q_network, target_network, discriminator, data, device, args, global_step, optimizer, model_idx_to_true_skill, intrinsic_rewards_arr.mean(), intrinsic_rewards_arr.std()
                    )
                if global_step % args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )
            if (global_step > args.learning_starts and global_step % args.model_save== 0):
                model_dir = f"runs/checkpoints/qtargetmaml/{run_name}"
                os.makedirs(model_dir, exist_ok=True)
                torch.save({
                    "q_network_state_dict": q_network.state_dict(),
                    "disc_state_dict": discriminator.state_dict(),
                    "episode": episode
                }, os.path.join(model_dir, f"latest.pth"))

            if termination or truncation:
                break

        episode += 1
        if(global_step % 100 == 0):
            print(f"global steps {global_step}")

        if(global_step > args.learning_starts and args.track):
            wandb.log({
                "episodic/td_loss": float(loss.item()),
                "episodic/q_values": float(old_val.mean().item()),
                "episodic/global_steps": float(global_step),
                "episodic/td_target": float(td_target.mean().item()),
                "episodic/intrinsic_reward": float(intrinsic_rewards.mean().item()),
            }, step=int(episode))

    model_dir = f"runs/checkpoints/qtargetmaml/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        "q_network_state_dict": q_network.state_dict(),
        "disc_state_dict": discriminator.state_dict(),
        "episode": episode
    }, os.path.join(model_dir, f"latest.pth"))
    env.close()
    writer.close()
