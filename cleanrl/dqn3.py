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
from cleanrl.diayn.models import Discriminator, QNetwork , SFNetwork
from cleanrl.diayn.utils import train_dqn, train_discriminator , test_dqn ,  merge_batches , train_sf
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit


from collections import defaultdict



@dataclass
class Args:
    exp_name: str = "FastAdaption"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "FastAdaption"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "LunarLander-v2"
    """the id of the environment"""
    total_timesteps: int = 700000
    """"total timesteps"""
    # max_episodes: int = 5001
    # """ number of episodes """
    max_timesteps: int = 1000
    """timesteps per episode"""
  
    lr: float = 4e-4
    """the learning rate of the sfnet optimizer"""

    num_env: int = 1

    skill: int = 5
    n_skills : int = 25
    sf_dim = 32


    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
 
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1
    """the target network update rate"""
    sf_target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
  
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
   
 
    episode_logging: int = 1
    """number of episodes after which episodic plots are plotted"""
    gradient_freq: int  = 1000
    """ gradient logging after given numer of .backward calls()"""
    train_frequency: int = 4
    """the frequency of training"""

    model_path1: str = "runs/checkpoints/diayn/LunarLander-v2__diayn__1__2025-04-25_22-19-35__1745599775/latest.pth"
    model_path2: str = "runs/checkpoints/maml/LunarLander-v2__MAML_SF__1__2025-04-28_00-34-15__1745780655/latest.pth" 
    # model_path2: str = ""
    
    ddqn: bool = True
    '''whether to use ddqn'''





def make_env(env_id, seed, idx, capture_video, run_name , max_timesteps):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = TimeLimit(env, max_timesteps)
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
            )
        else:
            env = gym.make(env_id)
            env = TimeLimit(env, max_timesteps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


def concat_state_latent(s, z, n_skills):
    z_one_hot = np.zeros(n_skills, dtype=np.float32)
    z_one_hot[z] = 1.0
    return np.concatenate([s, z_one_hot], axis=-1)




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

   
    env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name , args.max_timesteps)()
    args.n_actions = env.action_space.n

    q_network = QNetwork(env , args.n_skills).to(device)
    discriminator = Discriminator(env.observation_space.shape[0], args.n_skills ).to(device)
    
    checkpoint1 = torch.load(args.model_path1)
    q_network.load_state_dict(checkpoint1["q_network_state_dict"])
    discriminator.load_state_dict(checkpoint1["discriminator_state_dict"])
    

    sf_network = SFNetwork(env.observation_space.shape[0], env.action_space.n, args.sf_dim).to(device)
    sf_target_network = SFNetwork(env.observation_space.shape[0], env.action_space.n, args.sf_dim).to(device)
    

    if(args.model_path2!=""):
        checkpoint2 = torch.load(args.model_path2)
        sf_network.load_state_dict(checkpoint2["sfmeta_network_state_dict"])


    optimizer = optim.Adam(sf_network.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    sf_target_network.load_state_dict(sf_network.state_dict())

   

    if args.track:
        wandb.watch(
            models=[q_network],
            log="all",          # can also use "gradients" or "parameters"
            log_freq=args.gradient_freq,     # every 1000 backward() calls
            log_graph=False
        )


  

    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )


    action_space = env.action_space
    start_time = time.time()
    global_step = 0
    episode = 0
    w = discriminator.q.weight[args.skill].detach().to(device)
    w = w / (torch.norm(w) + 1e-8)
    z = args.skill


    while global_step < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0


        for steps in range(args.max_timesteps+5):

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            if random.random() < epsilon:
                action = action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1, state_dim)
                actions = []

                for a in range(env.action_space.n):
                    action_onehot = torch.zeros(env.action_space.n, device=device)
                    action_onehot[a] = 1.0
                    action_onehot = action_onehot.unsqueeze(0)  # shape (1, action_dim)
                    
                    q_val = sf_network(state_tensor, action_onehot, w)  # output shape (1,)
                    actions.append(q_val.item())  # collect scalar value

                # Choose action with maximum sfᵀw
                action = int(np.argmax(actions))

            next_state, reward, termination, truncation, info = env.step(action)
            rb.add(
                np.array([state]),
                np.array([next_state]),
                np.array([action]),
                np.array([reward]),
                np.array([termination]),
                [info]
            )
    
            state = next_state
            episode_reward += reward
            global_step += 1

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                
                if(global_step % args.train_frequency == 0):
                    data = rb.sample(args.batch_size)
                 

                    td_loss, old_val ,  td_target , target_loss  = train_sf(sf_network, sf_target_network, data, device, args, global_step , optimizer  , z , discriminator , w , q_network)
            
                    
                # update target network
                if global_step % args.sf_target_network_frequency == 0:
                    for sf_target_network_param, sf_network_param in zip(sf_target_network.parameters(), sf_network.parameters()):
                        sf_target_network_param.data.copy_(
                            args.tau * sf_network_param.data + (1.0 - args.tau) * sf_target_network_param.data
                        )



            # if (global_step > args.learning_starts and global_step % args.step_hist_save == 0):
                
                # model_dir = f"runs/checkpoints/diayn/{run_name}"
                # os.makedirs(model_dir, exist_ok=True)
                # torch.save({
                #     "q_network_state_dict": q_network.state_dict(),
                #     "discriminator_state_dict": discriminator.state_dict(),
                #     "episode": episode
                # }, os.path.join(model_dir, f"latest.pth"))       
                # skill_reward_dict.clear()

            if(global_step % 100 == 0):
                print(f"global steps {global_step}")
            if termination or truncation:
                break

        episode+=1
        

        
        if(global_step > args.learning_starts and episode % args.episode_logging == 0): 
            wandb.log({
                "episodic/td_loss": float(td_loss.item()),
                "episodic/q_values": float(old_val.mean().item()),
                "episodic/global_steps": float(global_step),
                "episodic/target_loss" : float(target_loss.item()),
                "episodic/td_target": float(td_target.mean().item()),
            } , step = int(episode))
    
    env.close()
    writer.close()