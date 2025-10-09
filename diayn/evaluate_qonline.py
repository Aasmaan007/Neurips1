import os
import time
import torch
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from cleanrl.diayn.models import Discriminator, QNetwork
from cleanrl.cleanrl.dqn2 import concat_state_latent
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers import TimeLimit
import tyro
import wandb
import gym_maze


class KeepAliveAfterFail(gym.Wrapper):
    """
    After the env reports terminated=True, freeze the dynamics and keep
    returning the last observation with reward=0 until TimeLimit truncates.
    For visualization only.
    """
    def __init__(self, env):
        super().__init__(env)
        self._failed = False
        self._last_obs = None

    def reset(self, **kwargs):
        self._failed = False
        self._last_obs, info = self.env.reset(**kwargs)
        return self._last_obs, info

    def step(self, action):
        if self._failed:
            obs = np.array(self._last_obs, copy=True)
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        if terminated:
            self._failed = True
            terminated = False
            truncated = False
            reward = 0.0

        return obs, reward, terminated, truncated, info


@dataclass
class Args:
    exp_name: str = "evaluate_diayn"
    seed: int = 10
    cuda: bool = True
    capture_video: bool = True
    env_id: str = "CartPole-v1"
    n_skills: int = 25
    n_skills_selected: int = 6
    eval_episodes_per_skill: int = 15
    model_path: str = "runs/checkpoints/qtargetmaml/CartPole-v1__q_online__1__2025-09-16_22-08-38__1758040718/latest.pth"
    wandb_project_name: str = "Diayn_Maze_Evaluate"
    wandb_entity: str = None
    track: bool = True
    max_timesteps: int = 1000
    record_every_x_episode: int = 3


def make_env(env_id, seed, skill, run_name, capture_video, record_every_x_episodes):
    def episode_trigger(episode_id):
        return episode_id % record_every_x_episodes == 0

    def thunk():
        # IMPORTANT: create with render_mode for RecordVideo
        env = gym.make(env_id, render_mode="rgb_array")
        env = KeepAliveAfterFail(env)
        env = TimeLimit(env, args.max_timesteps)

        if capture_video:
            video_folder = os.path.join("videos", run_name)
            name_prefix = f"skill_{skill}"
            env = RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=episode_trigger,
                name_prefix=name_prefix,
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + skill)
        return env

    return thunk


def evaluate_skill_policy(q_network, env_fn, device, skill_model_idx, n_skills_selected, eval_episodes, timesteps):
    returns = []
    env = env_fn(skill_model_idx)()
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=args.seed + ep + skill_model_idx)
        obs = concat_state_latent(obs, skill_model_idx, n_skills_selected)
        episode_return = 0.0
        for _ in range(timesteps + 5):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = torch.argmax(q_network(obs_tensor), dim=1).item()
            next_obs, reward, termination, truncation, _ = env.step(action)
            # Do NOT call env.render(); RecordVideo handles frames via rgb_array
            obs = concat_state_latent(next_obs, skill_model_idx, n_skills_selected)
            episode_return += reward
            if termination or truncation:
                break
        returns.append(float(episode_return))
    env.close()
    return sum(returns) / len(returns), returns


if __name__ == "__main__":
    args = tyro.cli(Args)
    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}__{timestamp}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/evaluate/{run_name}")
    writer.add_text("eval_hyperparams", str(vars(args)))

    selected_skills = [1, 3, 7, 8, 15, 18]  # replace if needed
    model_idx_to_true_skill = {i: s for i, s in enumerate(selected_skills)}
    true_skill_to_model_idx = {s: i for i, s in enumerate(selected_skills)}

    env_fn = lambda skill_model_idx: make_env(
        args.env_id, args.seed, model_idx_to_true_skill[skill_model_idx], run_name,
        capture_video=args.capture_video,
        record_every_x_episodes=args.record_every_x_episode
    )

    temp_env = gym.make(args.env_id, render_mode="rgb_array")
    q_network = QNetwork(temp_env, args.n_skills_selected)
    discriminator = Discriminator(temp_env.observation_space.shape[0], args.n_skills)
    temp_env.close()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    q_network.load_state_dict(checkpoint["q_network_state_dict"])
    discriminator.load_state_dict(checkpoint["disc_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    q_network.to(device)
    discriminator.to(device)

    mean_returns_per_skill = []

    for model_idx in range(args.n_skills_selected):
        avg_return, returns = evaluate_skill_policy(
            q_network,
            env_fn,
            device,
            model_idx,
            args.n_skills_selected,
            args.eval_episodes_per_skill,
            args.max_timesteps,
        )
        returns_array = np.array(returns, dtype=np.float32)
        writer.add_histogram(
            "eval/skill_reward_distribution_skill",
            returns_array,
            model_idx
        )
        mean_returns_per_skill.append(avg_return)

    rewards_array = np.array(mean_returns_per_skill, dtype=np.float32)

    writer.add_histogram(
        "eval/mean_reward_distribution_across_skills",
        rewards_array,
        global_step=0,
    )

    if args.track:
        wandb.log({
            "eval/mean_reward_distribution_across_skills": wandb.Histogram(rewards_array)
        }, step=0)

        table = wandb.Table(data=[
            [f"Skill {model_idx_to_true_skill[i]}", float(rewards_array[i])] for i in range(len(rewards_array))
        ], columns=["Skill", "MeanReturn"])

        wandb.log({
            "eval/returns_bar": wandb.plot.bar(
                table, "Skill", "MeanReturn", title="Returns per Skill"
            )
        }, step=0)

    print("Evaluation complete. Mean returns per selected skill:")
    for i, mean_ret in enumerate(mean_returns_per_skill):
        print(f"Selected Skill {model_idx_to_true_skill[i]} (model idx {i}): Mean Return = {mean_ret:.2f}")

    writer.close()
