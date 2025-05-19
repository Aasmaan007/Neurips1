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
    seed: int = 1
    cuda: bool = True
    env_id: str = "CartPole-v1"
    exp_name: str = "MAML_SF"
    data_path: str = "runs/data/CartPole-v1__unified_collection_1__2025-05-18_16-13-47__1747565027/maml_training_data.pkl"
    disc_path: str = "runs/checkpoints/diayn/CartPole-v1__diayn__1__2025-05-18_13-19-37__1747554577/latest.pth"
    qnet_path: str = "runs/checkpoints/qtargetmaml/CartPole-v1__q_online__1__2025-05-18_15-13-17__1747561397/latest.pth"
    sf_dim: int = 32
    n_skills_total: int = 25
    n_skills_selected: int = 6
    n_skills_epoch: int = 4
    n_actions: int = 2  # Set this according to env
    hidden_dim: int = 120
    inner_lr: float = 1e-3
    outer_lr: float = 2.5e-3
    num_epochs: int = 500000
    support_size: int = 128
    query_size: int = 64
    val_skill: int = 5
    wandb_project_name: str = "MAML_SF"
    wandb_entity: str = None
    track: bool = False
    multi_step_loss: bool = False
    ''' toggle multi-step outer loss'''
    use_fixed_outer_loss_weights: bool = True
    multi_step_loss_num_epochs: int = 600000  
    '''for deciding weights , epochs after which almost all weight to last loss  '''
    support_fraction: float = 0.5
    """total fraction of dataset which is support set"""
    num_steps: int = 1
    '''number of inner loop updates'''
    gradient_freq: int = 1
    '''every number of backward calls after which gradient logged'''
    max_param_change_fraction: float = 0.01
    '''parameter clip '''
    max_norm: float = 5.0
    '''gradient clipping'''
    model_path: str = "runs/checkpoints/maml//CartPole-v1__MAML_SF__1__2025-05-18_19-04-30__1747575270/latest.pth"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_fast_weights(model, fast_weights):
    """
    Load the fast weights into a model instance.
    """
    state_dict = model.state_dict()
    fast_weights_dict = {k: w for k, w in zip(state_dict.keys(), fast_weights)}
    state_dict.update(fast_weights_dict)
    model.load_state_dict(state_dict)

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
                    q_sup, q_que, w_z, inner_lr, weights, num_steps,max_param_change_fraction,
                    step_weights=None):
    fast_weights = [w.clone() for w in weights]
    step_outer_losses = []
    step_inner_losses = []
    q_pred_supp = []
    q_pred_query = []

    for step in range(num_steps):
        q_pred_sup = model.argforward(s_sup, a_sup, fast_weights, w_z)
        q_pred_supp.append(q_pred_sup.mean().item())
        innerloss = criterion(q_sup, q_pred_sup)
        step_inner_losses.append(innerloss)

        grads = torch.autograd.grad(innerloss, fast_weights, create_graph=True)

        # --- Compute norms
        total_grad_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
        total_param_norm = torch.sqrt(sum((w.detach() ** 2).sum() for w in fast_weights))

        # --- Compute maximum allowed step
        max_delta = max_param_change_fraction * total_param_norm
        proposed_update_norm = inner_lr * total_grad_norm

        # --- Rescale gradients if step would be too big
        if proposed_update_norm > max_delta:
            scale = max_delta / (proposed_update_norm + 1e-8)
            grads = [g * scale for g in grads]
            fast_weights = [w - g for w, g in zip(fast_weights, grads)]
        else :
            fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]

        q_pred_que = model.argforward(s_que, a_que, fast_weights, w_z)
        q_pred_query.append(q_pred_que.mean().item())
        outer_loss = criterion(q_que, q_pred_que)
        step_outer_losses.append(outer_loss)

    if step_weights is not None:
        weighted_outer_loss = sum(w * l for w, l in zip(step_weights, step_outer_losses))
    else:
        weighted_outer_loss = step_outer_losses[-1]  # last step only (standard MAML)

    # q_pred_sup_sum = q_pred_sup_sum / num_steps

    return step_inner_losses, step_outer_losses, weighted_outer_loss , q_pred_supp , q_pred_query , fast_weights

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
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")      
        wandb.config.update(vars(args), allow_val_change=True)

    with open(args.data_path, "rb") as f:
        state_data = pickle.load(f)
        np.random.shuffle(state_data)
    state_data = np.array(state_data)
    np.random.shuffle(state_data)

    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    
    discriminator = Discriminator(state_dim, args.n_skills_total)
    discriminator.load_state_dict(torch.load(args.disc_path)['discriminator_state_dict'])
    discriminator = discriminator.to(device)

    qnet = QNetwork(env , args.n_skills_selected)
    qnet.load_state_dict(torch.load(args.qnet_path)['q_network_state_dict'])
    qnet = qnet.to(device)
    
    
    model = SFNetwork(state_dim, args.n_actions, sf_dim=args.sf_dim)
    model.load_state_dict(torch.load(args.model_path)['sfmeta_network_state_dict'])
    model = model.to(device)
    criterion = nn.MSELoss()

    dummy_state  = torch.zeros(1, state_dim, device=device)
    dummy_action = torch.zeros(1, args.n_actions, device=device)
    dummy_task   = torch.zeros(args.sf_dim,   device=device)
    _ = model(dummy_state, dummy_action, dummy_task)

    if(args.track):
        wandb.watch(
            models = [model],
            log = "all",
            log_freq = args.gradient_freq,
            log_graph = False
        )

    states, actions = get_all_pairs(state_data, args.n_actions)
    (support_states, support_actions), (query_states, query_actions) = partition_full_dataset(states, actions, args.support_fraction)

    num_steps = args.num_steps
    # number of inner loop updates 
    allowed_skills = [2, 5, 8, 10, 12, 16]
    true_skill_to_model_idx = {s: i for i, s in enumerate(allowed_skills)}  #22 ->5
    

    for z in allowed_skills:
        z_ind =  true_skill_to_model_idx[z]
       
        weights=list(model.parameters())
        step_inner_losses_sums = [0.0 for _ in range(args.num_steps)]
        step_outer_losses_sums = [0.0 for _ in range(args.num_steps)]
        qsupport_sums = [0.0 for _ in range(args.num_steps)]
        qquery_sums = [0.0 for _ in range(args.num_steps)]

        step_weights = get_per_step_loss_weights(args, 1) if args.multi_step_loss else None
        # skills_this_epoch = random.sample([z for z in range(args.n_skills) if z!=args.val_skill], args.n_skills_epoch)
        # skills_this_epoch = random.sample([z for z in allowed_skills if z!=args.val_skill], args.n_skills_epoch)
        # skills_this_epoch = [6]
      

        w_z = discriminator.q.weight[z].detach().to(device)
        w_z = w_z / (torch.norm(w_z) + 1e-8)

        support_indices = torch.randint(0, support_states.size(0), (args.support_size,))
        query_indices = torch.randint(0, query_states.size(0), (args.query_size,))

        s_sup = support_states[support_indices].to(device)
        a_sup = support_actions[support_indices].to(device)
        s_que = query_states[query_indices].to(device)
        a_que = query_actions[query_indices].to(device)


        q_sup = get_q_values(qnet, s_sup, a_sup, z_ind, args.n_skills_selected, device)
        q_que = get_q_values(qnet, s_que, a_que, z_ind, args.n_skills_selected, device)


        
        
        step_inner_losses, step_outer_losses, metaloss, q_pred_supp , q_pred_query , fast_weights = maml_inner_loop(
            model, criterion, s_sup, a_sup, s_que, a_que,
            q_sup, q_que, w_z, args.inner_lr, weights,
            num_steps, args.max_param_change_fraction,
            step_weights=step_weights.to(device) if step_weights is not None else None
        )
        sf_model = SFNetwork(state_dim, args.n_actions, sf_dim=args.sf_dim).to(device)
        # Load the fast weights
        load_fast_weights(sf_model, fast_weights)

        print(f"metaloss for skill {z} is {metaloss}")
        state_dict = sf_model.state_dict()
        fast_weights_dict = {k: w for k, w in zip(state_dict.keys(), fast_weights)}
        model_dir = f"runs/checkpoints/sfmetaadapt/{z}/{run_name}"
        os.makedirs(model_dir, exist_ok=True)
        torch.save({
                "sfmeta_network_state_dict": sf_model.state_dict()
            }, os.path.join(model_dir, f"latest.pth"))


if __name__ == "__main__":
    train()