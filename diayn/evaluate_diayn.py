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


@dataclass
class Args:
    exp_name: str = "evaluate_diayn"
    seed: int = 10
    cuda: bool = True
    capture_video: bool = True
    env_id: str = "LunarLander-v2"
    n_skills: int = 25
    eval_episodes_per_skill: int = 15
    model_path: str = "runs/checkpoints/latest.pth"
    wandb_project_name: str = "Diayn_LunarLander_Evaluate"
    wandb_entity: str = None
    track: bool = True
    max_timesteps: int = 1000
    record_every_x_episode: int = 3
    

def make_env(env_id, seed, skill, run_name, capture_video, record_every_x_episodes):
    """Creates the environment for each skill and records video every 'x' episodes as specified."""
    
    def episode_trigger(episode_id):
        """Trigger recording every 'record_every_x_episodes' episodes."""
        return episode_id % record_every_x_episodes == 0

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = TimeLimit(env, args.max_timesteps)
        if capture_video:
            video_folder = os.path.join("videos", run_name)
            name_prefix = f"skill_{skill}"
            env = RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=episode_trigger,  # Use custom episode trigger
                name_prefix=name_prefix
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + skill)  # seed different per skill
        return env

    return thunk


def evaluate_skill_policy(q_network, env_fn, device, skill, n_skills, eval_episodes, timesteps):
    """Evaluate the policy for a given skill using one environment for the entire skill."""
    returns = []
    env = env_fn(skill)()  # Create the environment only once per skill
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed = args.seed+ep+skill)
        obs = concat_state_latent(obs, skill, n_skills)
        episode_return = 0
        for steps in range(timesteps+5):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = torch.argmax(q_network(obs_tensor), dim=1).item()
            next_obs, reward, termination, truncation, _ = env.step(action)
            obs = concat_state_latent(next_obs, skill, n_skills)
            episode_return += reward
            if termination or truncation:
                break
        returns.append(episode_return)
    env.close()
    return sum(returns) / len(returns) , returns


if __name__ == "__main__":
    args = tyro.cli(Args)
    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}__{timestamp}"

    # Initialize W&B (no sync_tensorboard!)
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TensorBoard writer
    writer = SummaryWriter(f"runs/evaluate/{run_name}")
    writer.add_text("eval_hyperparams", str(vars(args)))

    
    # Create env factory for each skill (creates only one environment per skill)
    env_fn = lambda skill: make_env(args.env_id, args.seed, skill, run_name, capture_video=args.capture_video , record_every_x_episodes = args.record_every_x_episode)

    # Initialize the model
    temp_env = gym.make(args.env_id)
    q_network = QNetwork(temp_env, args.n_skills)
    discriminator = Discriminator(temp_env.observation_space.shape[0], args.n_skills)
    temp_env.close()

    # Load model weights
    checkpoint = torch.load(args.model_path , map_location=torch.device("cpu"))
    q_network.load_state_dict(checkpoint["q_network_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    q_network.to(device)
    discriminator.to(device)



    mean_returns_per_skill = []

    # Evaluate for each skill
    for skill in range(args.n_skills):
        
        avg_return , returns = evaluate_skill_policy(
            q_network,
            env_fn,
            device,
            skill,
            args.n_skills,
            args.eval_episodes_per_skill,
            args.max_timesteps,
        )
        returns_array = np.array(returns, dtype=np.float32)
        writer.add_histogram(
            f"eval/skill_reward_distribution_skill",
            returns_array,
            skill
        )

        # # TensorBoard scalar
        # writer.add_scalar(f"eval/skill_{skill}_mean_reward", avg_return, 0)

        # # W&B scalar
        # if args.track:
        #     wandb.log({f"eval/skill_{skill}_mean_reward": avg_return}, step=0)

        mean_returns_per_skill.append(avg_return)

    rewards_array = np.array(mean_returns_per_skill, dtype=np.float32)

    # TensorBoard histogram
    writer.add_histogram(
        "eval/mean_reward_distribution_across_skills",
        rewards_array,
        global_step=0,
    )

    # W&B histogram
    if args.track:
        wandb.log({
            "eval/mean_reward_distribution_across_skills": wandb.Histogram(rewards_array)
        }, step=0)

        table = wandb.Table(data=[
            [f"Skill {i}", rewards_array[i]] for i in range(len(rewards_array))
        ], columns=["Skill", "MeanReturn"])

        wandb.log({
            "eval/returns_bar": wandb.plot.bar(
                table, "Skill", "MeanReturn", title="Returns per Skill"
            )
        }, step=0)

    print("Evaluation complete. Mean returns per skill:")
    for skill, mean_ret in enumerate(mean_returns_per_skill):
        print(f"Skill {skill}: Mean Return = {mean_ret:.2f}")

    writer.close()
