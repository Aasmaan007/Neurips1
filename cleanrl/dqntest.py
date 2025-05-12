# full cleanrl DQN with dot-product Q-values (q(s, a) = phi(s, a)^T w)
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 35
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "NEURIPS"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    env_id: str = "MountainCar-v0"
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
    w_path: str  = "runs/checkpoints/env_phi_task/MountainCar-v0__joint_phi_task__1__2025-05-13_01-35-43/latest.pth"
    model_path = "runs/checkpoints/maml/MountainCar-v0__MAML_SF__1__2025-05-13_00-16-21__1747075581/latest.pth"
    w_random: bool = False
    pretrained: bool = False

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
        state_dim = np.prod(env.single_observation_space.shape)
        action_dim = env.single_action_space.n
        self.input_dim = state_dim + action_dim
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 32),  # 16-dim embedding
        )

    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        return self.embedding(x)  # returns phi(s, a)
    
class TaskVector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))

    def forward(self, phi_next):
        w_norm = self.w / (torch.norm(self.w) + 1e-8)
        return torch.matmul(phi_next, w_norm)
    

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.seed}__wrandom-{args.w_random}__pretrained-{args.pretrained}__{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("stepwise/*", step_metric="gobal_step")      
        wandb.config.update(vars(args), allow_val_change=True)

    writer = SummaryWriter(f"runs/testingrandom/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    q_network = QNetwork(envs).to(device)


    if(args.pretrained):
        checkpoint2 = torch.load(args.model_path)
        sf_state_dict = checkpoint2["sfmeta_network_state_dict"]
        mapped_state_dict = {}
        mapped_state_dict["embedding.0.weight"] = sf_state_dict["l1.weight"]
        mapped_state_dict["embedding.0.bias"]   = sf_state_dict["l1.bias"]
        mapped_state_dict["embedding.2.weight"] = sf_state_dict["l2.weight"]
        mapped_state_dict["embedding.2.bias"]   = sf_state_dict["l2.bias"]
        mapped_state_dict["embedding.4.weight"] = sf_state_dict["l3.weight"]
        mapped_state_dict["embedding.4.bias"]   = sf_state_dict["l3.bias"]
        q_network.load_state_dict(mapped_state_dict)






    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    w = torch.randn(32).to(device)
    w = w / (w.norm() + 1e-8)
    task_vector = TaskVector(32).to(device)
    checkpoint1 = torch.load(args.w_path)
    if(not args.w_random):
        task_vector.load_state_dict(checkpoint1["task_vector"])
    w = (task_vector.w / (torch.norm(task_vector.w) + 1e-8)).detach()

    if args.track:
        wandb.watch(
            models=[q_network],
            log="all",          # can also use "gradients" or "parameters"
            log_freq=1000,     # every 1000 backward() calls
            log_graph=False
        )


    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device, handle_timeout_termination=False)
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                batch_size = obs_tensor.shape[0]
                action_dim = envs.single_action_space.n
                action_onehots = torch.eye(action_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
                obs_expanded = obs_tensor.unsqueeze(1).repeat(1, action_dim, 1).reshape(-1, obs_tensor.shape[1])
                action_expanded = action_onehots.reshape(-1, action_dim)
                phi_sa = q_network(obs_expanded, action_expanded).view(batch_size, action_dim, -1)
                qvals = torch.einsum("bad,d->ba", phi_sa, w)
                actions = torch.argmax(qvals, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    
                    if args.track:
                        wandb.log({
                            "stepwise/episodic_return":float(info["episode"]["r"]),
                        } , step = int(global_step))


        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                batch_size = data.observations.shape[0]
                action_dim = envs.single_action_space.n

                # Compute Q(s', a') target
                with torch.no_grad():
                    next_obs_exp = data.next_observations.unsqueeze(1).repeat(1, action_dim, 1).reshape(-1, data.next_observations.shape[1])
                    action_onehots_all = torch.eye(action_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, action_dim)
                    target_feats = target_network(next_obs_exp, action_onehots_all).view(batch_size, action_dim, -1)
                    qvals_next = torch.einsum("bad,d->ba", target_feats, w)
                    target_max = qvals_next.max(dim=1).values
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())

                # Current Q(s, a)
                action_onehot = torch.zeros((batch_size, action_dim), device=device)
                action_onehot.scatter_(1, data.actions, 1.0)
                current_feat = q_network(data.observations, action_onehot)
                q_pred = torch.einsum("bd,d->b", current_feat, w)

                loss = F.mse_loss(q_pred, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), global_step)
                    writer.add_scalar("losses/q_values", q_pred.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    
                    if args.track:
                        wandb.log({
                            "stepwise/q_values":float(q_pred.mean().item()),
                            "stepwise/td_loss":float(loss.item()),
                            # "stepwise/reward":float(data.rewards.flatten().mean().item()),

                        } , step = int(global_step))

            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()