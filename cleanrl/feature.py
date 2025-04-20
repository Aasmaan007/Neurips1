import os, time, random, pickle
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, random_split
import gymnasium as gym
import wandb
import tyro

from cleanrl.diayn.models import FeatureNetwork


@dataclass
class Args:
    exp_name: str = "phi_regression"
    data_path: str = "runs/data/phi_training_data.pkl"
    env_id: str = "LunarLander-v2"
    seed: int = 1
    cuda: bool = True
    total_epochs: int = 50000
    batch_size: int = 256
    val_split: float = 0.1
    sf_dim: int = 32
    learning_rate: float = 1e-3
    use_normalized: bool = False
    wandb_project_name: str = "FeatureNet_LunarLander_Train"
    wandb_entity: str = None
    track: bool = True
    gradient_freq: int = 1000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_dataset(data_path):
    with open(data_path, "rb") as f:
        raw = pickle.load(f)
    states, skill_idx, rewards, weights = zip(*raw)
    states = torch.tensor(np.stack(states), dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    weights = torch.tensor(np.stack(weights), dtype=torch.float32)
    return states, rewards, weights


def train():
    args = tyro.cli(Args)
    set_seed(args.seed)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name
        )
        wandb.define_metric("epoch")
        wandb.define_metric("epochs/*", step_metric="epoch")      
        wandb.config.update(vars(args), allow_val_change=True)


    print(f"Loading data from {args.data_path}")
    states, rewards, weights = load_dataset(args.data_path)
    dataset = TensorDataset(states, rewards, weights)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    dummy_env = gym.make(args.env_id)
    phi_net = FeatureNetwork(dummy_env, args.sf_dim).to(device)
    optimizer = optim.Adam(phi_net.parameters(), lr=args.learning_rate)

    if args.track:
        wandb.watch(phi_net, log="all", log_freq=args.gradient_freq)

    for epoch in range(1, args.total_epochs + 1):
        phi_net.train()
        total_train_loss = 0

        for s, r, w in train_loader:
            s, r, w = s.to(device), r.to(device), w.to(device)
            phi_raw, phi_norm = phi_net(s)
            phi_used = phi_norm if args.use_normalized else phi_raw
            pred = torch.sum(phi_used * w, dim=1)
            loss = F.mse_loss(pred, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * s.size(0)

        train_mse = total_train_loss / len(train_loader.dataset)
        train_rmse = np.sqrt(train_mse)

        # Validation
        phi_net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for s, r, w in val_loader:
                s, r, w = s.to(device), r.to(device), w.to(device)
                phi_raw, phi_norm = phi_net(s)
                phi_used = phi_norm if args.use_normalized else phi_raw
                pred = torch.sum(phi_used * w, dim=1)
                total_val_loss += F.mse_loss(pred, r, reduction='sum').item()

        val_mse = total_val_loss / len(val_loader.dataset)
        val_rmse = np.sqrt(val_mse)

        print(f"[Epoch {epoch}] Train MSE: {train_mse:.4f} | RMSE: {train_rmse:.4f} | "
            f"Val MSE: {val_mse:.4f} | RMSE: {val_rmse:.4f}")

        if args.track:
            wandb.log({
                "epochs/train/mse": float(train_mse),
                "epochs/train/rmse": float(train_rmse),
                "epochs/val/mse": float(val_mse),
                "epochs/val/rmse": float(val_rmse),
            } , step = int(epoch))


    model_dir = f"runs/checkpoint/train_phi/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        "phi_network" : phi_net.state_dict()
    }, os.path.join(model_dir , f"latest.pth"))
    

if __name__ == "__main__":
    train()
