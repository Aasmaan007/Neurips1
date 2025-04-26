import os
import time
import random
import pickle
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import gymnasium as gym
import wandb

from cleanrl.diayn.models import SFNetwork, Discriminator , QNetwork

@dataclass
class Args:
    exp_name: str = "maml_sf"
    seed: int = 1
    cuda: bool = True
    env_id: str = "LunarLander-v2"
    exp_name: str = "MAML"
    data_path: str = "runs/data/maml_training_data.pkl"
    model_path: str = "runs/checkpoints/latest.pth"
    sf_dim: int = 32
    n_skills: int = 25
    n_skills_epoch: int = 12
    n_actions: int = 4  # Set this according to env
    hidden_dim: int = 120
    inner_lr: float = 1e-2
    outer_lr: float = 3e-4
    num_epochs: int = 1000
    support_size: int = 64
    query_size: int = 64
    val_skill: int = 5
    wandb_project_name: str = "MAML_SF"
    wandb_entity: str = None
    track: bool = True
    multi_step_loss: bool = True  
    ''' toggle multi-step outer loss'''
    use_fixed_outer_loss_weights: bool = True
    multi_step_loss_num_epochs: int = 200  
    '''for deciding weights , epochs after which almost all weight to last loss  '''
    support_fraction: float = 0.7
    """total fraction of dataset which is support set"""
    num_steps: int = 3
    '''number of inner loop updates'''

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_all_pairs(states, n_actions):
    all_states, all_actions = [], []
    for s in states:
        for a in range(n_actions):
            all_states.append(s)
            one_hot = np.zeros(n_actions)
            one_hot[a] = 1.0
            all_actions.append(one_hot)
    return torch.tensor(np.stack(all_states), dtype=torch.float32), torch.tensor(np.stack(all_actions), dtype=torch.float32)

def partition_full_dataset(states, actions, support_fraction):
    total_samples = states.size(0)
    indices = torch.randperm(total_samples)

    num_support = int(total_samples * support_fraction)
    support_idx = indices[:num_support]
    query_idx = indices[num_support:]

    s_sup = states[support_idx]
    a_sup = actions[support_idx]
    s_que = states[query_idx]
    a_que = actions[query_idx]

    return (s_sup, a_sup), (s_que, a_que)


def concat_state_latent(s, z, n_skills):
    z_one_hot = np.zeros(n_skills, dtype=np.float32)
    z_one_hot[z] = 1.0
    return np.concatenate([s, z_one_hot], axis=-1)

def get_q_values(qnet, states, actions, z, n_skills, device):
    # states: (B, state_dim), actions: (B, action_dim one-hot)
    with torch.no_grad():
        states_np = states.detach().cpu().numpy()
        state_aug = np.array([concat_state_latent(s, z, n_skills) for s in states_np])
        state_aug = torch.tensor(state_aug, dtype=torch.float32).to(device)

        qvals = qnet(state_aug)  # shape: (B, num_actions)
        action_indices = torch.argmax(actions, dim=1).view(-1, 1)  # shape: (B, 1)
        q_selected = qvals.gather(1, action_indices).squeeze()  # shape: (B,)
        return q_selected




def maml_inner_loop(model, criterion, s_sup, a_sup, s_que, a_que,
                    q_sup, q_que, w_z, inner_lr, weights, num_steps,
                    step_weights=None):
    fast_weights = [w.clone() for w in weights]
    step_outer_losses = []

    for step in range(num_steps):
        q_pred_sup = model.argforward(s_sup, a_sup, fast_weights, w_z)
        innerloss = criterion(q_sup, q_pred_sup)

        grads = torch.autograd.grad(innerloss, fast_weights, create_graph=True)
        fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]

        q_pred_que = model.argforward(s_que, a_que, fast_weights, w_z)
        outer_loss = criterion(q_que, q_pred_que)
        step_outer_losses.append(outer_loss)

    if step_weights is not None:
        weighted_outer_loss = sum(w * l for w, l in zip(step_weights, step_outer_losses))
    else:
        weighted_outer_loss = step_outer_losses[-1]  # last step only (standard MAML)

    return innerloss, step_outer_losses, weighted_outer_loss

def get_per_step_loss_weights(args: Args, current_epoch: int):
    weights = np.ones(args.num_steps) * (1.0 / args.num_steps)
    
    if args.use_fixed_outer_loss_weights:
        return torch.tensor(weights, dtype=torch.float32)

    decay_rate = 1.0 / args.num_steps / args.multi_step_loss_num_epochs
    min_non_final = 0.03 / args.num_steps

    for i in range(args.num_steps - 1):
        weights[i] = max(weights[i] - (current_epoch * decay_rate), min_non_final)

    weights[-1] = min(1.0 - ((args.num_steps - 1) * min_non_final),
                      weights[-1] + (current_epoch * (args.num_steps - 1) * decay_rate))
    
    return torch.tensor(weights, dtype=torch.float32)


def train():
    args = tyro.cli(Args)
    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d_%H-%M-%S')}__{timestamp}"

    if args.track:

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
        )
        
        wandb.define_metric("epoch")
        wandb.define_metric("epochs/*", step_metric="epoch")      
        wandb.config.update(vars(args), allow_val_change=True)

    with open(args.data_path, "rb") as f:
        state_data = pickle.load(f)
    state_data = np.array(state_data)

    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    
    discriminator = Discriminator(state_dim, args.n_skills)
    discriminator.load_state_dict(torch.load(args.model_path)['discriminator_state_dict'])
    discriminator = discriminator.to(device)

    qnet = QNetwork(env , args.n_skills)
    qnet.load_state_dict(torch.load(args.model_path)['q_net_state_dict'])
    qnet = qnet.to(device)
    
    
    model = SFNetwork(state_dim, args.n_actions, sf_dim=args.sf_dim).to(device)
    meta_opt = optim.Adam(model.parameters(), lr=args.outer_lr)
    criterion = nn.MSELoss()

    states, actions = get_all_pairs(state_data, args.n_actions)
    (support_states, support_actions), (query_states, query_actions) = partition_full_dataset(states, actions, args.support_fraction)

    num_steps = args.num_steps
    # number of inner loop updates 


    for epoch in range(1, args.num_epochs + 1):
        metaloss_sum = 0
        innerloss_sum = 0
        weights=list(model.parameters())
        step_loss_sums = [0.0 for _ in range(args.num_steps)]
        step_weights = get_per_step_loss_weights(args, epoch) if args.multi_step_loss else None
        skills_this_epoch = random.sample([z for z in range(args.n_skills) if z!=args.val_skill], args.n_skills_epoch)
        
        for z in skills_this_epoch:
            
            if z == args.val_skill:
                continue

            w_z = discriminator.q.weight[z].detach().to(device)
            w_z = w_z / (np.linalg.norm(w_z) + 1e-8)
    
            support_indices = torch.randint(0, support_states.size(0), (args.support_size,))
            query_indices = torch.randint(0, query_states.size(0), (args.query_size,))

            s_sup = support_states[support_indices].to(device)
            a_sup = support_actions[support_indices].to(device)
            s_que = query_states[query_indices].to(device)
            a_que = query_actions[query_indices].to(device)


            q_sup = get_q_values(qnet, s_sup, a_sup, z, args.n_skills, device)
            q_que = get_q_values(qnet, s_que, a_que, z, args.n_skills, device)


            
            
            innerloss, step_losses, metaloss = maml_inner_loop(
                model, criterion, s_sup, a_sup, s_que, a_que,
                q_sup, q_que, w_z, args.inner_lr, weights,
                num_steps,
                step_weights=step_weights.to(device)
            )
            
            metaloss_sum += metaloss
            innerloss_sum += innerloss
            for i, step_loss in enumerate(step_losses):
                step_loss_sums[i] += step_loss

        
        metaloss_sum = metaloss_sum / (args.n_skills_epoch)
        innerloss_sum = innerloss_sum / (args.n_skills_epoch)
        step_loss_avgs = [loss / (args.n_skills_epoch) for loss in step_loss_sums]


        meta_opt.zero_grad(set_to_none=True)
        metagrads=torch.autograd.grad(metaloss_sum, weights)
        for w,g in zip(weights,metagrads):
            w.grad=g
        meta_opt.step()
        
        
        

        # Validation (on a held-out skill)
        w_z = discriminator.q.weight[args.val_skill].to(device)
        w_z = w_z / (np.linalg.norm(w_z) + 1e-8)

        support_indices = torch.randint(0, support_states.size(0), (args.support_size,))
        query_indices = torch.randint(0, query_states.size(0), (args.query_size,))

        s_sup = support_states[support_indices].to(device)
        a_sup = support_actions[support_indices].to(device)
        s_que = query_states[query_indices].to(device)
        a_que = query_actions[query_indices].to(device)

        q_sup = get_q_values(qnet, s_sup, a_sup, args.val_skill, args.n_skills, device)
        q_que = get_q_values(qnet, s_que, a_que, args.val_skill, args.n_skills, device)
        weights=list(model.parameters())

        valinnerloss , valstep_losses, valmetaloss = maml_inner_loop(
                model, criterion, s_sup, a_sup, s_que, a_que,
                q_sup, q_que, w_z, args.inner_lr, weights,
                num_steps,
                step_weights=step_weights.to(device)
            )

        if args.track:
            log_dict = {
                "epochs/train/mean_innerloss": float(innerloss_sum),
                "epochs/train/mean_metaloss": float(metaloss_sum),
                "epochs/train/root/mean_innerloss": float(torch.sqrt(innerloss_sum)),
                "epochs/train/root/mean_metaloss": float(torch.sqrt(metaloss_sum)),
                "epochs/val/innerloss": float(valinnerloss),
                "epochs/val/metaloss": float(valmetaloss),
                "epochs/val/root/innerloss": float(torch.sqrt(valinnerloss)),
                "epochs/val/root/metaloss": float(torch.sqrt(valmetaloss))
            }
                        # Add per-step training average losses
            for i, loss in enumerate(step_loss_avgs):
                log_dict[f"epochs/train/outer_loss_step_{i}"] = float(loss)
                log_dict[f"epochs/train/root/outer_loss_step_{i}"] = float(torch.sqrt(loss))

            # Add per-step validation losses
            for i, loss in enumerate(valstep_losses):
                log_dict[f"epochs/val/outer_loss_step_{i}"] = float(loss)
                log_dict[f"epochs/val/root/outer_loss_step_{i}"] = float(torch.sqrt(loss))

            # Final logging
            wandb.log(log_dict, step=int(epoch))


    model_dir = f"runs/checkpoints/maml/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
            "sfmeta_network_state_dict": model.state_dict(),
        }, os.path.join(model_dir, f"latest.pth"))


if __name__ == "__main__":
    train()
