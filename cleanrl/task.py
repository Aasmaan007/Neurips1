import os, time, random, pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from cleanrl.diayn.models import FeatureNetwork , SFNetwork ,QNetwork
import gymnasium as gym
import wandb
import tyro

@dataclass
class Args:
    exp_name: str = "joint_phi_task"
    diayn_data_path: str = "runs/data/LunarLander-v2__unified_collection_1__2025-05-02_06-27-16__1746147436/phi_training_data.pkl"
    env_data_path: str = "runs/data/LunarLander-v2__5__2025-05-04_04-29-13__1746313153/task_regression_data.pkl"
    env_id: str = "LunarLander-v2"
    sf_dim: int = 32
    batch_size: int = 1024
    sample_size: int = 1000000
    total_epochs: int = 1000
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
    model_path1: str = "runs/checkpoints/featurenet/LunarLander-v2__phi_regression__1__2025-05-02_20-45-59/latest.pth"
    model_path2: str = "runs/checkpoints/maml/LunarLander-v2__MAML_SF__1__2025-05-02_00-50-05__1746127205/latest.pth" 
    qnet_path: str =  "runs/checkpoints/qtargetmaml/LunarLander-v2__q_online__1__2025-05-01_16-14-07__1746096247/latest.pth"
    env_weight: float = 0.30
    diayn_weight: float = 0.70
    n_skills_selected: int = 6


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

def load_phi_data(data_path):
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

def load_task_data(path):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    random.shuffle(raw)
    s , action , r_env , s_next, terminated  = zip(*raw)
    return (
        torch.tensor(s, dtype=torch.float32),          # s
        torch.tensor(action, dtype=torch.long),        # a
        torch.tensor(r_env, dtype=torch.float32),      # r
        torch.tensor(s_next, dtype=torch.float32),     # s'
        torch.tensor(terminated, dtype=torch.bool)     # done
    )

def concat_state_latent(s, z, n_skills):
    z_one_hot = np.zeros(n_skills, dtype=np.float32)
    z_one_hot[z] = 1.0
    return np.concatenate([s, z_one_hot], axis=-1)


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
    # s_next_phi, r_diayn, w_z = load_phi_data(args.diayn_data_path)
    # s_next_env, r_env = load_task_data(args.env_data_path)
    s, a, r_env, s_next,  terminated = load_task_data(args.env_data_path)


    # r_env_std = r_env.std().item()
    # r_diayn_std = r_diayn.std().item()
    # diayn_scale = r_env_std / (r_diayn_std + 1e-8)


    

    dummy_env = gym.make(args.env_id)
    input_dim = np.prod(dummy_env.observation_space.shape)

    phi_net = FeatureNetwork(dummy_env, args.sf_dim , args.dropout).to(device)
    task_vector = TaskVector(args.sf_dim).to(device)

    checkpoint1 = torch.load(args.model_path1)
    phi_net.load_state_dict(checkpoint1["phi_network"])

    checkpoint2 = torch.load(args.model_path2)
    sf_network = SFNetwork(dummy_env.observation_space.shape[0], dummy_env.action_space.n, args.sf_dim).to(device)
    sf_network.load_state_dict(checkpoint2["sfmeta_network_state_dict"])

    qnet = QNetwork(dummy_env , args.n_skills_selected)
    qnet.load_state_dict(torch.load(args.qnet_path)['q_network_state_dict'])

    # optimizer_phi = optim.Adam(phi_net.parameters(), lr=args.learning_rate_phi)
    optimizer_w = optim.Adam(task_vector.parameters(), lr=args.learning_rate_w)

    # diayn_dataset = TensorDataset(s_next_phi, r_diayn, w_z)
    # task_dataset = TensorDataset(s_next_env, r_env)
    task_dataset = TensorDataset(s, a, r_env , s_next, terminated)


    # diayn_loader = DataLoader(diayn_dataset, batch_size=args.batch_size,
    #                           sampler=RandomSampler(diayn_dataset, replacement=True, num_samples=args.sample_size),
    #                           num_workers=args.workers)

    task_loader = DataLoader(task_dataset, batch_size=args.batch_size,
                             sampler=RandomSampler(task_dataset, replacement=True, num_samples=args.sample_size),
                             num_workers=args.workers)

    if args.track:
        # wandb.watch(phi_net, log="all", log_freq=1000)
        wandb.watch(task_vector, log="all", log_freq=1000)

    env_weight = args.env_weight
    diayn_weight = args.diayn_weight

    for epoch in range(1, args.total_epochs + 1):
        phi_net.train()
        total_diayn_loss = 0
        total_lunar_loss = 0
        total_diayn_loss_raw = 0

        # for batch_idx, ((s_env, r_env_batch), (s_phi, r_diayn_batch, w_z_batch)) in enumerate(zip(task_loader, diayn_loader)):
        for batch_idx, (s, a, r , snext, terminated) in enumerate(task_loader):
            s = s.to(device)        # s'
            a = a.to(device)                # a
            r = r.to(device)
            snext = snext.to(device)                # s
            terminated = terminated.to(device)
            # s_phi = s_phi.to(device)
            # r_diayn_batch = r_diayn_batch.to(device)
            # w_z_batch = w_z_batch.to(device)

            # (1) w update: φ(s′)^T w ≈ r_env
            if(epoch % args.task_lag == 0):
                # with torch.no_grad():
                #     phi_env = phi_net.forward2(s_env)
                with torch.no_grad():
                    a_onehot = F.one_hot(a.to(torch.long), num_classes=dummy_env.action_space.n).float().to(device)
                    skill_idx = 1  # You can loop over skills if needed later

                    # Step 2: Generate batch-wise skill vector
                    skill_vec = torch.full((snext.size(0),), skill_idx, dtype=torch.long, device=device)

                    # Step 3: Use your helper to concatenate state with one-hot skill vector
                    s_aug_np = [concat_state_latent(snext.cpu().numpy(), z.item(), args.n_skills_selected) for s, z in zip(snext, skill_vec)]
                    s_aug = torch.tensor(s_aug_np, dtype=torch.float32, device=device)

                    # Step 4: Pass through Q-network to get greedy next action
                    q_values = qnet(s_aug)
                    a_next = torch.argmax(q_values, dim=1)
                    a_next_onehot = F.one_hot(a_next.to(torch.long), num_classes=dummy_env.action_space.n).float().to(device)
                    sf_sa = sf_network.sf_vector(s, a_onehot)                  # SF(s, a)
                    sf_snext_anext = sf_network.sf_vector(snext, a_next_onehot)  # SF(s′, a′)# SF(s', a')
                    gamma = 0.99
                    phi_env = torch.where(
                        terminated.unsqueeze(1),
                        sf_sa,
                        sf_sa - gamma * sf_snext_anext
                    )

                pred_r_env_for_w = task_vector(phi_env)
                loss_w = F.mse_loss(pred_r_env_for_w, r)

                optimizer_w.zero_grad()
                loss_w.backward()
                optimizer_w.step()

            # (2) φ update: dual loss
            # w_detached = (task_vector.w / (torch.norm(task_vector.w) + 1e-8)).detach()
            # w_expanded = w_detached.unsqueeze(0).expand(s_env.size(0), -1)  # Shape: [batch_size, sf_dim]
            # pred_r_env = phi_net(s_env, w_expanded)

            # pred_r_diayn = phi_net(s_phi, w_z_batch)
            # loss_env = F.mse_loss(pred_r_env, r_env_batch)
            # loss_diayn_raw = F.mse_loss(pred_r_diayn, r_diayn_batch)
            # # loss_diayn = diayn_scale**2 * loss_diayn_raw
            # loss_diayn = loss_diayn_raw

            # max_epoch_for_schedule = 250
            # # env_weight = min(0.95, 0.8 + 0.15 * (epoch / max_epoch_for_schedule))
            # # diayn_weight = 1.0 - env_weight

            # total_phi_loss = env_weight * loss_env + diayn_weight * loss_diayn
            # optimizer_phi.zero_grad()
            # total_phi_loss.backward()
            # optimizer_phi.step()

            # total_diayn_loss += loss_diayn.item() * s_phi.size(0)
            # total_diayn_loss_raw += loss_diayn_raw.item() * s_phi.size(0)
            # total_lunar_loss += loss_env.item() * s_env.size(0)
            total_lunar_loss += loss_w.item() * s_env.size(0)

        # avg_loss_diayn = total_diayn_loss / args.sample_size
        avg_loss_lunar = total_lunar_loss / args.sample_size
        # avg_loss_diayn_raw = total_diayn_loss_raw  /  args.sample_size
        
        # print(f"[Epoch {epoch}]  Scaled DIAYN Loss: {avg_loss_diayn:.4f} | (Raw Diayn Loss {avg_loss_diayn_raw:.4f}) | Lunar Loss: {avg_loss_lunar:.4f}")
        print(f" [Epoch {epoch}] - Lunar Loss: {avg_loss_lunar:.4f} - Pred R: {pred_r_env_for_w.mean().item()} - Correc R: {r.mean().item()}")

        if args.track:
            wandb.log({
                # "epochs/diayn_mse_loss_scaled": avg_loss_diayn,
                # "epochs/diayn_mse_loss_raw": avg_loss_diayn_raw,
                "epochs/env_mse_loss": avg_loss_lunar
            }, step=epoch)


        # if epoch % 5 == 0:
        #     save_path = f"runs/checkpoints/env_phi_task/{run_name}"
        #     os.makedirs(save_path, exist_ok=True)
        #     torch.save({
        #         # "phi_net": phi_net.state_dict(),
        #         "task_vector": task_vector.state_dict()
        #     }, os.path.join(save_path, "latest.pth"))

    print("✅ Joint training complete.")
    # save_path = f"runs/checkpoints/env_phi_task/{run_name}"
    # os.makedirs(save_path, exist_ok=True)
    # torch.save({
    #     # "phi_net": phi_net.state_dict(),
    #     "task_vector": task_vector.state_dict()
    # }, os.path.join(save_path, "latest.pth"))

if __name__ == "__main__":
    train()