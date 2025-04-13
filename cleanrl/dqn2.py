# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass
import sys
print(sys.path)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from cleanrl.diayn.models import Discriminator, QNetwork
from cleanrl.diayn.utils import train_dqn, train_discriminator
from gymnasium import spaces
from collections import defaultdict




@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Diayn_LunarLander_Train"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "LunarLander-v2"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """"total timesteps"""
    # max_episodes: int = 5001
    # """ number of episodes """
    max_timesteps: int = 1000
    """timesteps per episode"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_env: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    n_skills: int = 20
    """ number of skills """
    epi_hist: int  = 250
    """ epsiodes after which histogram plotted"""
    save_model_every: int = 250
    """ epsiodes after which model saved"""
    hidden_units: int = 300
    """hidden units for both Qnetwork and Discriminator"""
    episode_logging: int = 1
    """number of episodes after which episodic plots are plotted"""

def concat_state_latent(s, z, n_skills):
    z_one_hot = np.zeros(n_skills, dtype=np.float32)
    z_one_hot[z] = 1.0
    return np.concatenate([s, z_one_hot], axis=-1)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
            )
        else:
            env = gym.make(env_id)
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
            # sync_tensorboard=True,
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("-----------------------------------------------------------------------------------------------------------------------------------------------")
    print("Using device:", device)

   
    env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)()

    q_network = QNetwork(env , args.n_skills , args.hidden_units).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(env , args.n_skills).to(device)
    target_network.load_state_dict(q_network.state_dict())

    discriminator = Discriminator(env.observation_space.shape[0], args.n_skills , args.hidden_units).to(device)
    discriminator_opt = optim.Adam(discriminator.parameters(), lr=args.learning_rate)
    cross_ent_loss = torch.nn.CrossEntropyLoss()


    obs_shape = env.observation_space.shape[0] + args.n_skills
    augmented_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    rb = ReplayBuffer(
        args.buffer_size,
        augmented_obs_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    obs_dim = np.prod(env.observation_space.shape)
    action_space = env.action_space
    start_time = time.time()
    global_step = 0
    running_logq_zs = 0
    skill_reward_dict = defaultdict(list)
    episode = 0


    while global_step < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        z = np.random.choice(args.n_skills)
        state, _ = env.reset(seed=args.seed + episode)
        state = concat_state_latent(state, z, args.n_skills)
        episode_reward = 0
        logq_zses = []

        for steps in range(args.max_timesteps):

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            if random.random() < epsilon:
                action = action_space.sample()
            else:
                q_values = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=1).item()

            next_state, reward, done, _, info = env.step(action)
            next_state_aug = concat_state_latent(next_state, z, args.n_skills)
            rb.add(
                np.array([state]),
                np.array([next_state_aug]),
                np.array([action]),
                np.array([0.0]),
                np.array([done]),
                [info]
            )

            state = next_state_aug
            episode_reward += reward
            global_step += 1

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                # Update Q_network
                loss, old_val = train_dqn(q_network, target_network, discriminator, data, device, args, global_step , optimizer)

                # Update discriminator
                zs = data.observations[:, -args.n_skills:].argmax(dim=1)  # extract skills from one-hot
                disc_loss = train_discriminator(discriminator, data, zs, device , discriminator_opt)
                logq_zses.append(disc_loss)
                


                # if global_step % 100 == 0:
                #     writer.add_scalar("losses_by_global_step/td_loss", loss.item(), global_step)
                #     writer.add_scalar("losses_by_global_step/q_values", old_val.mean().item(), global_step)
                #     writer.add_scalar("charts_by_global_step/SPS", int(global_step / (time.time() - start_time)), global_step)
                    
                # update target network
                if global_step % args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )
                if done:
                    break

        episode+=1
                
        if(len(logq_zses)==0):
            average_logq_zs = 0
        else:
            average_logq_zs = sum(logq_zses) / len(logq_zses)
        skill_reward_dict[z].append(episode_reward)  # store reward under correct skill

        if running_logq_zs == 0:
            running_logq_zs = average_logq_zs
        else:
            running_logq_zs = 0.99 *running_logq_zs + 0.01 * average_logq_zs
        
        if(global_step > args.learning_starts and episode % args.episode_logging == 0):   
            if(episode % 10 == 0):
                print(f"Episodic Return for {episode} with {steps} number of steps  is {episode_reward}")
            writer.add_scalar("charts_by_episode/episodic_return", episode_reward, episode)
            writer.add_scalar("charts_by_episode/episodic_logq_zs", average_logq_zs, episode)
            writer.add_scalar("charts_by_episode/Running logq(z|s)", running_logq_zs, episode)
            writer.add_scalar("charts_by_episode/SPS", int(global_step / (time.time() - start_time)), episode)
            writer.add_scalar("losses_by_episode/td_loss", loss.item(), episode)
            writer.add_scalar("losses_by_episode/q_values", old_val.mean().item(), episode)

            wandb.log({
                "episodic/episodic_return": float(episode_reward),
                "episodic/episodic_logq_zs": float(average_logq_zs.item() if hasattr(average_logq_zs, 'item') else average_logq_zs),
                "episodic/Running logq(z|s)": float(running_logq_zs.item() if hasattr(running_logq_zs, 'item') else running_logq_zs),
                "episodic/td_loss": float(loss.item()),
                "episodic/q_values": float(old_val.mean().item()),
                # "episode": int(episode)
            } , step = int(episode))
          




        if (global_step > args.learning_starts and episode % args.epi_hist == 0):
            for skill_id in range(args.n_skills):
                raw_rewards = skill_reward_dict[skill_id]
                print(f"[DEBUG] skill {skill_id}, rewards type: {type(raw_rewards)}, content: {raw_rewards[:5]}")
                rewards = np.array(raw_rewards, dtype=np.float32)
                if len(rewards) > 1:
                    writer.add_histogram(
                        f"skills/skill_{skill_id}_reward_distribution_every_{args.epi_hist}_episodes",
                        rewards,
                        episode
                    )
                    wandb.log({
                        f"skills/skill_{skill_id}_reward_distribution": wandb.Histogram(rewards)
                    }, step=episode)
            skill_reward_dict.clear()



        if (global_step > args.learning_starts and episode % args.save_model_every == 0):
            model_dir = f"runs/checkpoints/{run_name}"
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                "q_network_state_dict": q_network.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "episode": episode
            }, os.path.join(model_dir, f"latest.pth"))

    env.close()
    writer.close()
