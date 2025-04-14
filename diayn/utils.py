import torch
import torch.nn.functional as F
from torch.optim import Adam
from stable_baselines3.common.buffers import ReplayBuffer

def train_dqn(q_network, target_network, discriminator, data, device, args, global_step , optimizer):

    # Extract state and next state from the replay buffer
    states = data.observations
    next_states = data.next_observations
    zs = data.observations[:, -args.n_skills:].argmax(dim=1)  # infer skill index from one-hot

    # ----------------------------
    # Compute intrinsic reward
    logits = discriminator(next_states[:, :discriminator.input_dim])  # only state, no skill
    logq_zs = F.log_softmax(logits, dim=-1)
    logq_z = logq_zs[range(args.batch_size), zs]
    logpz = torch.tensor(1.0 / args.n_skills + 1e-6).log().to(device)
    intrinsic_rewards = (logq_z - logpz).detach()
    if(global_step % 500 == 0):
        print("Intrinsic rewards (first 5):", intrinsic_rewards[:5].cpu().numpy())


    # Calculate Q-values and loss
    with torch.no_grad():
        target_max, _ = target_network(next_states).max(dim=1)
        td_target = intrinsic_rewards + args.gamma * target_max * (1 - data.dones.flatten())

    old_val = q_network(states).gather(1, data.actions).squeeze()
    loss = F.mse_loss(td_target, old_val)

    # Optimize Q-network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss , old_val , intrinsic_rewards


def test_dqn(target_network, discriminator, data, device, args):

    # Extract state and next state from the replay buffer
    states = data.observations
    next_states = data.next_observations
    zs = data.observations[:, -args.n_skills:].argmax(dim=1)  # infer skill index from one-hot

    # ----------------------------
    # Compute intrinsic reward
    logits = discriminator(next_states[:, :discriminator.input_dim])  # only state, no skill
    logq_zs = F.log_softmax(logits, dim=-1)
    logq_z = logq_zs[range(5), zs]
    logpz = torch.tensor(1.0 / args.n_skills + 1e-6).log().to(device)
    intrinsic_rewards = (logq_z - logpz).detach()
  


    # Calculate Q-values and loss
    with torch.no_grad():
        target_max, _ = target_network(next_states).max(dim=1)
        td_target = intrinsic_rewards + args.gamma * target_max * (1 - data.dones.flatten())


    return td_target


def train_discriminator(discriminator, data, zs, device , discriminator_opt):
    # Train the discriminator to predict the skills
    discriminator_logits = discriminator(data.observations[:, :discriminator.input_dim])  # Only use the state (no skill)
    cross_ent_loss = torch.nn.CrossEntropyLoss()
    disc_loss = cross_ent_loss(discriminator_logits, zs)

    # Optimize discriminator
    discriminator_opt.zero_grad()
    disc_loss.backward()
    discriminator_opt.step()

    return -disc_loss
