import os, time, random, pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from cleanrl.diayn.models import FeatureNetwork, SFNetwork, QNetwork
import gymnasium as gym
import wandb
import tyro

@dataclass
class Args:
    exp_name: str = "joint_phi_task"
    env_data_path: str = "runs/data/Acrobot-v1__1__2025-05-16_07-54-59__pretrained-False /task_regression_data.pkl"
    env_id: str = "Acrobot-v1"
    sf_dim: int = 32
    batch_size: int = 1024
    sample_size: int = 1000000
    total_epochs: int = 3
    learning_rate_phi: float = 2.5e-4
    learning_rate_w: float = 6e-4
    task_lag: int = 1
    seed: int = 1
    cuda: bool = True
    wandb_project_name: str = "JointTraining"
    wandb_entity: str = None
    track: bool = True
    workers: int = 4
    dropout: float = 0.15
    model_path2: str = "runs/checkpoints/maml/Acrobot-v1__MAML_SF__1__2025-05-17_17-16-23__1747482383/500/latest.pth" 
    qnet_path: str =  "runs/checkpoints/qtargetmaml/Acrobot-v1__q_online__1__2025-05-17_02-28-14__1747429094/latest.pth"
    env_weight: float = 0.30
    diayn_weight: float = 0.70
    n_skills_selected: int = 21


class TaskVector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))

    def forward(self, phi_next):
        w_norm = self.w / (torch.norm(self.w) + 1e-8)
        return torch.matmul(phi_next, w_norm)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def load_task_data(path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    random.shuffle(raw)
    s , action , r_env , s_next, terminated  = zip(*raw)
    return (
        torch.tensor(s, dtype=torch.float32),
        torch.tensor(action, dtype=torch.long),
        torch.tensor(r_env, dtype=torch.float32),
        torch.tensor(s_next, dtype=torch.float32),
        torch.tensor(terminated, dtype=torch.bool)
    )

def concat_state_latent_batch(states, z, n_skills):
    """
    states: Tensor of shape [B, D] or [B, 1, D]
    z: Tensor of shape [B]
    Returns: [B, D + n_skills]
    """
    if states.dim() == 3:
        states = states.squeeze(1)  # remove the extra dimension
    z_onehot = F.one_hot(z.to(torch.long), num_classes=n_skills).float()
    return torch.cat([states, z_onehot], dim=-1)

def train():
    args = tyro.cli(Args)
    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    if args.track:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args), name=run_name)
        wandb.define_metric("epoch")
        wandb.define_metric("epochs/*", step_metric="epoch")

    print("Loading DIAYN and task regression data...")
    s, a, r_env_batch, s_next, terminated = load_task_data(args.env_data_path)

    dummy_env = gym.make(args.env_id)

    task_vector = TaskVector(args.sf_dim).to(device)
    qnet = QNetwork(dummy_env, args.n_skills_selected).to(device)


    checkpoint2 = torch.load(args.model_path2)
    sf_network = SFNetwork(dummy_env.observation_space.shape[0], dummy_env.action_space.n, args.sf_dim).to(device)
    sf_network.load_state_dict(checkpoint2["sfmeta_network_state_dict"])

    qnet.load_state_dict(torch.load(args.qnet_path)['q_network_state_dict'])
    qnet.eval()

    optimizer_w = optim.Adam(task_vector.parameters(), lr=args.learning_rate_w)

    task_dataset = TensorDataset(s, a, r_env_batch, s_next, terminated)
    task_loader = DataLoader(task_dataset, batch_size=args.batch_size,
                             sampler=RandomSampler(task_dataset, replacement=True, num_samples=args.sample_size),
                             num_workers=args.workers)

    if args.track:
        wandb.watch(task_vector, log="all", log_freq=1000)

    for epoch in range(1, args.total_epochs + 1):
        total_lunar_loss = 0

        for batch_idx, (s, a, r, snext, terminated) in enumerate(task_loader):
            s = s.to(device)
            a = a.to(device)
            r = r.to(device)
            snext = snext.to(device)
            terminated = terminated.to(device)

            if epoch % args.task_lag == 0:
                with torch.no_grad():
                    a_onehot = F.one_hot(a, num_classes=dummy_env.action_space.n).float().to(device)
                    skill_idx = 1
                    skill_vec = torch.full((snext.size(0),), skill_idx, dtype=torch.long, device=device)
                    s_aug = concat_state_latent_batch(snext, skill_vec, args.n_skills_selected)
                    q_values = qnet(s_aug)
                    a_next = torch.argmax(q_values, dim=1)
                    a_next_onehot = F.one_hot(a_next, num_classes=dummy_env.action_space.n).float().to(device)
                    s = s.squeeze(1)
                    snext = snext.squeeze(1)
                    a_onehot = a_onehot.squeeze(1)
                    # print(snext.shape)
                    # print(s.shape)
                    # print(a_next_onehot.shape)
                    # print(a_onehot.shape)
                    sf_sa = sf_network.sf_vector(s, a_onehot)
                    sf_snext_anext = sf_network.sf_vector(snext, a_next_onehot)
                    gamma = 0.99
                    phi_env = torch.where(terminated.unsqueeze(1), sf_sa, sf_sa - gamma * sf_snext_anext)

                pred_r_env_for_w = task_vector(phi_env)
                loss_w = F.mse_loss(pred_r_env_for_w, r)

                optimizer_w.zero_grad()
                loss_w.backward()
                optimizer_w.step()

                total_lunar_loss += loss_w.item() * s.size(0)

        avg_loss_lunar = total_lunar_loss / args.sample_size
        print(f" [Epoch {epoch}] - Lunar Loss: {avg_loss_lunar:.4f} - Pred R: {pred_r_env_for_w.mean().item()} - Correc R: {r.mean().item()}")

        if args.track:
            wandb.log({"epochs/env_mse_loss": avg_loss_lunar}, step=epoch)

        if epoch % 1 == 0:
            save_path = f"runs/checkpoints/env_phi_task/{run_name}"
            os.makedirs(save_path, exist_ok=True)
            torch.save({"task_vector": task_vector.state_dict()}, os.path.join(save_path, "latest.pth"))

    print("\u2705 Joint training complete.")

    # Final save
    # save_path = f"runs/checkpoints/env_phi_task/{run_name}"
    # os.makedirs(save_path, exist_ok=True)
    # torch.save({"task_vector": task_vector.state_dict()}, os.path.join(save_path, "latest.pth"))

if __name__ == "__main__":
    train()