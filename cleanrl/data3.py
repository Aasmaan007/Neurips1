import os
import random
import time
from dataclasses import dataclass
import tyro

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import TimeLimit
import wandb
import pickle

from cleanrl.diayn.models import Discriminator, QNetwork

@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    env_id: str = "Acrobot-v1"
    max_timesteps: int = 1000
    total_timesteps: int = 1000000
    # skill_timesteps: int = 392157
    n_skills_total: int = 25
    n_skills_selected: int = 6
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.50
    pos_dup_factor: int = 60
    model_path_disc: str = "runs/checkpoints/diayn/Acrobot-v1__diayn__1__2025-05-19_23-20-06__1747677006/latest.pth"
    model_path_qnet: str = "runs/checkpoints/qtargetmaml/Acrobot-v1__q_online__1__2025-05-20_01-46-39__1747685799/latest.pth"
    wandb_project_name: str = "unified_data_collection"
    wandb_entity: str = None
    track: bool = True
    exp_name: str = "unified_collection"
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
    return min(start_e + slope * t, end_e)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    args = tyro.cli(Args)
    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}_{args.seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}__{timestamp}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
        )
        wandb.define_metric("episode")
        wandb.define_metric("episodic/*", step_metric="episode")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = make_env(args.env_id, args.seed, args.max_timesteps)

    q_net = QNetwork(env, args.n_skills_selected).to(device)
    discriminator = Discriminator(env.observation_space.shape[0], args.n_skills_total).to(device)
    checkpoint_disc = torch.load(args.model_path_disc)
    checkpoint_qnet = torch.load(args.model_path_qnet)
    q_net.load_state_dict(checkpoint_qnet["q_network_state_dict"])
    discriminator.load_state_dict(checkpoint_disc["discriminator_state_dict"])

    allowed_skills = [2, 6, 8, 10, 21, 23]
    model_idx_to_true_skill = {i: s for i, s in enumerate(allowed_skills)}
    true_skill_to_model_idx = {s: i for i, s in enumerate(allowed_skills)}  #22 ->5

    phi_training_data = []
    maml_training_data = []
    per_skill_states = {z: [] for z in allowed_skills}
    pos_only_buffer = []
    task_regression_data = []  # (s_next, reward) tuples
    task_regression_data2 = []

    # offline_q_buffer = []  # new buffer to be saved for offline Q training

    episode = 0
    global_step = 0

    # for z in allowed_skills:
    #     skill_steps = 0
    #     episode_skill = 0
    #     z_ind = true_skill_to_model_idx[z]
    while global_step < args.total_timesteps:
        z = np.random.choice(allowed_skills)
        z_ind = true_skill_to_model_idx[z]
        obs, _ = env.reset(seed=args.seed + z_ind * 1000 + episode)
        obs_aug = concat_state_latent(obs, z_ind, args.n_skills_selected)
        episode += 1
        for t in range(args.max_timesteps+5):
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.tensor(obs_aug, dtype=torch.float32).unsqueeze(0).to(device))
                    action = torch.argmax(q_vals, dim=1).item()


            next_obs, reward, terminated, truncated, _ = env.step(action)
            task_regression_data.append((next_obs.copy(), reward))
            with torch.no_grad():
                    next_obs_aug = concat_state_latent(next_obs, z_ind, args.n_skills_selected)
                    q_vals_next = q_net(torch.tensor(next_obs_aug, dtype=torch.float32).unsqueeze(0).to(device))
                    action_next = torch.argmax(q_vals_next, dim=1).item()
            task_regression_data2.append((obs.copy() , action , reward , next_obs.copy(), action_next , terminated))

            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = discriminator(next_obs_tensor.unsqueeze(0))
                q_zs = F.softmax(logits, dim=-1)
                logq_zs = torch.log(torch.clamp(q_zs, min=1e-6))
                logpz = torch.tensor(1.0 / args.n_skills_total + 1e-6).log().to(device)

            for z_idx in allowed_skills:
                r = (logq_zs[0, z_idx] - logpz).item()
                w = discriminator.q.weight[z_idx].detach().cpu().numpy()
                w = w / (np.linalg.norm(w) + 1e-8)
                phi_training_data.append((next_obs.copy(), r, w))
                # offline_q_buffer.append((obs.copy(), action, r, next_obs.copy(), skill_to_model_idx[z_idx], terminated))

                if z_idx == z and r > 0:
                    for _ in range(args.pos_dup_factor):
                        pos_only_buffer.append((next_obs.copy(), r, w))
                        # offline_q_buffer.append((obs.copy(), action, r, next_obs.copy(), skill_to_model_idx[z_idx], terminated))

            maml_training_data.append(next_obs.copy())
            per_skill_states[z].append(next_obs.copy())

            obs = next_obs
            obs_aug = concat_state_latent(obs, z_ind, args.n_skills_selected)
            global_step +=1 

            if(global_step % 1000 == 0):
                print(f"Global steps {global_step} completed")

            if terminated or truncated:
                break

        if(args.track):
            wandb.log({
                "episodic/len_SF_data": float(len(phi_training_data)),
                "episodic/len_MAML_data": float(len(maml_training_data)),
                 "episodic/len_task_regression_data": float(len(task_regression_data)),
                "episodic/global_steps": float(global_step),
                "episodic/pos_only_buffer": float(len(pos_only_buffer))
            },step = int(episode))
        
        

    env.close()
    model_dir = f"runs/data/{run_name}"
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "phi_training_data.pkl"), "wb") as f:
        pickle.dump(phi_training_data + pos_only_buffer, f)
    with open(os.path.join(model_dir, "maml_training_data.pkl"), "wb") as f:
        pickle.dump(maml_training_data, f)
    # with open(os.path.join(model_dir, "offline_q_buffer.pkl"), "wb") as f:
    #     pickle.dump(offline_q_buffer, f)
    with open(os.path.join(model_dir, "task_regression_data.pkl"), "wb") as f:
        pickle.dump(task_regression_data, f)
    with open(os.path.join(model_dir, "task_regression_data2.pkl"), "wb") as f:
        pickle.dump(task_regression_data2, f)



    per_skill_dir = os.path.join(model_dir, "per_skill_states")
    os.makedirs(per_skill_dir, exist_ok=True)
    for z, states in per_skill_states.items():
        with open(os.path.join(per_skill_dir, f"skill_{z}.pkl"), "wb") as f:
            pickle.dump(states, f)

   
    # print(f"Saved offline Q buffer to {model_dir}/offline_q_buffer.pkl")
    print(f"Saved φ(s) training data to {model_dir}/phi_training_data.pkl")
    print(f"Saved MAML training data to {model_dir}/maml_training_data.pkl")
    print(f"Total positive reward duplicates added: {len(pos_only_buffer)}")