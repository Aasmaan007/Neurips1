import torch
import torch.nn.functional as F
from torch.optim import Adam
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers import ReplayBufferSamples


def merge_batches(a: ReplayBufferSamples, b: ReplayBufferSamples) -> ReplayBufferSamples:
    """
    Concatenate two ReplayBufferSamples along the batch dimension (dim=0)
    and return a new ReplayBufferSamples object.
    """
    return ReplayBufferSamples(
        observations      = torch.cat([a.observations,      b.observations],      dim=0),
        next_observations = torch.cat([a.next_observations, b.next_observations], dim=0),
        actions           = torch.cat([a.actions,           b.actions],           dim=0),
        rewards           = torch.cat([a.rewards,           b.rewards],           dim=0),
        dones             = torch.cat([a.dones,             b.dones],             dim=0),
        # infos             = a.infos + b.infos               # list concat
    )

def train_dqn(q_network, target_network, discriminator, data, device, args, global_step , optimizer):

    # Extract state and next state from the replay buffer
    states = data.observations
    next_states = data.next_observations
    zs = data.observations[:, -args.n_skills:].argmax(dim=1)  # infer skill index from one-hot

    # ----------------------------
    # Compute intrinsic reward
    logits = discriminator(next_states[:, :discriminator.input_dim])  # only state, no skill
    q_zs = F.softmax(logits , dim = -1)
    q_zs_clamped = torch.clamp(q_zs,min = 1e-6)
    logq_zs = torch.log(q_zs_clamped)
    logq_z = logq_zs[range(args.batch_size), zs]
    logpz = torch.tensor(1.0 / args.n_skills + 1e-6).log().to(device)
    intrinsic_rewards = (logq_z - logpz).detach()
    # intrinsic_rewards = data.rewards.flatten() 
    # if(global_step % 500 == 0):
    # #     print("Intrinsic rewards (first 5):", intrinsic_rewards[:5].cpu().numpy())
    # if(global_step % 500 == 0):
    #     print("logq_z (first 5):", logq_z[:5].detach().cpu().numpy())
    # Calculate Q-values and loss
    with torch.no_grad():
        # Step 1: Use the online network to choose the best next action
        next_q_values_online = q_network(next_states)
        next_actions = next_q_values_online.argmax(dim=1, keepdim=True)  # shape [batch_size, 1]

        # Step 2: Use the target network to evaluate the value of that action
        next_q_values_target = target_network(next_states)
        target_q_values = next_q_values_target.gather(1, next_actions).squeeze()

        # TD target with bootstrapping
        td_target = intrinsic_rewards + args.gamma * target_q_values * (1 - data.dones.flatten())
        bootstrapping = args.gamma * target_q_values * (1 - data.dones.flatten())

    old_val = q_network(states).gather(1, data.actions).squeeze()
    loss = F.mse_loss(td_target, old_val)

    # Optimize Q-network
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(q_network.parameters() , max_norm = args.grad_norm)
    optimizer.step()

    return loss , old_val , intrinsic_rewards , logq_z.detach() , td_target , bootstrapping



def train_dqn2(q_network, target_network, data, device, args, global_step , optimizer):

    # Extract state and next state from the replay buffer
    states = data.observations
    next_states = data.next_observations

    with torch.no_grad():
        target_max, _ = target_network(next_states).max(dim=1)
        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
        bootstrapping = args.gamma * target_max * (1 - data.dones.flatten())

    old_val = q_network(states).gather(1, data.actions).squeeze()
    loss = F.mse_loss(td_target, old_val)

    # Optimize Q-network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss , old_val 

def test_dqn(target_network, discriminator, data, device, q_network, args , optimizer):

    # Extract state and next state from the replay buffer
    states = data.observations
    next_states = data.next_observations
    zs = data.observations[:, -args.n_skills:].argmax(dim=1)  # infer skill index from one-hot

    # ----------------------------
    # Compute intrinsic reward
    logits = discriminator(next_states[:, :discriminator.input_dim])  # only state, no skill
    q_zs = F.softmax(logits , dim = -1)
    q_zs_clamped = torch.clamp(q_zs,min = 1e-6)
    logq_zs = torch.log(q_zs_clamped)
    logq_z = logq_zs[range(args.batch_size_terminal_log), zs]
    logpz = torch.tensor(1.0 / args.n_skills + 1e-6).log().to(device)
    intrinsic_rewards = (logq_z - logpz).detach()
  
   

    # # Calculate Q-values and loss
    with torch.no_grad():
        next_q_values_online = q_network(next_states)
        next_actions = next_q_values_online.argmax(dim=1, keepdim=True)  # shape [batch_size, 1]

        # Step 2: Use the target network to evaluate the value of that action
        next_q_values_target = target_network(next_states)
        target_q_values = next_q_values_target.gather(1, next_actions).squeeze()

        # TD target with bootstrapping
        td_target = intrinsic_rewards + args.gamma * target_q_values * (1 - data.dones.flatten())
        bootstrapping = args.gamma * target_q_values * (1 - data.dones.flatten())
        old_val = q_network(states).gather(1, data.actions).squeeze()

    loss = F.mse_loss(td_target, old_val)

    # # Optimize Q-network
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # return bootstrapping , intrinsic_rewards , logq_z.detach() , td_target
    return bootstrapping , intrinsic_rewards , old_val , loss


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
