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
    data_path: str = "runs/data/LunarLander-v2__unified_collection_1__2025-05-02_06-27-16__1746147436/phi_training_data.pkl"
    env_id: str = "LunarLander-v2"
    seed: int = 1
    cuda: bool = True
    total_epochs: int = 75
    batch_size: int = 1024
    val_split: float = 0.15
    sample_size: int = 2000000
    sf_dim: int = 32
    learning_rate: float = 2.5e-4
    use_normalized: bool = False
    wandb_project_name: str = "FeatureNet"
    wandb_entity: str = None
    track: bool = True
    gradient_freq: int = 1000
    workers: int = 4
    dropout: float = 0.10


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_dataset(data_path):
    with open(data_path, "rb") as f:
        raw = pickle.load(f)

    # Shuffle raw list while keeping (s, r, w) tuples intact
    random.shuffle(raw)

    # Convert to numpy arrays first with preallocation to reduce memory fragmentation
    num_samples = len(raw)
    state_dim = len(raw[0][0])
    sf_dim = len(raw[0][2])

    states_np = np.empty((num_samples, state_dim), dtype=np.float32)
    rewards_np = np.empty((num_samples,), dtype=np.float32)
    weights_np = np.empty((num_samples, sf_dim), dtype=np.float32)

    for i, (s, r, w) in enumerate(raw):
        states_np[i] = s
        rewards_np[i] = r
        weights_np[i] = w

    # Convert to torch tensors (just one time, no copies)
    states = torch.from_numpy(states_np)
    rewards = torch.from_numpy(rewards_np)
    weights = torch.from_numpy(weights_np)

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

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=args.sample_size),
        num_workers = args.workers  # <-- ADD THIS
    )

    val_loader = DataLoader(
        val_set,
        batch_size= args.batch_size,
        num_workers = args.workers  # <-- ADD THIS
    )



    dummy_env = gym.make(args.env_id)
    phi_net = FeatureNetwork(dummy_env, args.sf_dim , args.dropout).to(device)
    optimizer = optim.Adam(phi_net.parameters(), lr=args.learning_rate)

    if args.track:
        wandb.watch(phi_net, log="all", log_freq=args.gradient_freq)

    for epoch in range(1, args.total_epochs + 1):
        phi_net.train()
        total_train_loss = 0

        for s, r, w in train_loader:
            s, r, w = s.to(device), r.to(device), w.to(device)
            r_pred = phi_net(s , w)
            loss = F.mse_loss(r_pred, r)
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
                r_pred = phi_net(s , w)
                total_val_loss += F.mse_loss(r_pred, r, reduction='sum').item()

        val_mse = total_val_loss / len(val_loader.dataset)
        val_rmse = np.sqrt(val_mse)

        print(f"[Epoch {epoch}] Train MSE: {train_mse:.4f} |"
            f" Val MSE: {val_mse:.4f}")

        if args.track:
            wandb.log({
                "epochs/train/mse": float(train_mse),
                # "epochs/train/rmse": float(train_rmse),
                "epochs/val/mse": float(val_mse),
                # "epochs/val/rmse": float(val_rmse),
            } , step = int(epoch))

        if(epoch % 5 == 0):
            model_dir = f"runs/checkpoints/featurenet/{run_name}"
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                "phi_network" : phi_net.state_dict()
            }, os.path.join(model_dir , f"latest.pth"))
    
    model_dir = f"runs/checkpoints/featurenet/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        "phi_network" : phi_net.state_dict()
    }, os.path.join(model_dir , f"latest.pth"))
    

if __name__ == "__main__":
    train()