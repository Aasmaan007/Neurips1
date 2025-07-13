import os
import random
import time
from dataclasses import dataclass

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
from gymnasium.spaces import Dict, Box


def extract_obs_image(obs_dict):
    return obs_dict["image"]


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "MiniGrid-Unlock-v0"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        c, h, w = env.single_observation_space['image'].shape[::-1]  # Convert (H, W, C) -> (C, H, W)
        self.network = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, env.single_action_space.n),
        )

    def forward(self, obs_dict):
        x = obs_dict['image'].permute(0, 3, 1, 2).float() / 255.0
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Only use 'image' from observation
    image_obs_space = Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
    rb = ReplayBuffer(
        args.buffer_size,
        image_obs_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    episode_returns = []

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            obs_tensor = {k: torch.tensor(v).to(device) for k, v in obs.items() if k == "image"}
            q_values = q_network(obs_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ret = info['episode']['r']
                    print(f"global_step={global_step}, episodic_return={ret}")
                    writer.add_scalar("charts/episodic_return", ret, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    episode_returns.append(ret)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs["image"], real_next_obs["image"], actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            obs_batch = {"image": data.observations.to(device)}
            next_obs_batch = {"image": data.next_observations.to(device)}

            with torch.no_grad():
                target_max, _ = target_network(next_obs_batch).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())

            old_val = q_network(obs_batch).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

    if episode_returns:
        avg_return = np.mean(episode_returns)
        print(f"Average episodic return: {avg_return:.2f}")
        writer.add_scalar("charts/average_return", avg_return, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
