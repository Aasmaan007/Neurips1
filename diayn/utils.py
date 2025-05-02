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

def train_sf(sf_network, sf_target_network, data, device, args, global_step, optimizer,  w):
    """
    Train SF Network using Bellman backup and intrinsic rewards based on discriminator.
    Fully batched version (no slow for-loop over actions).
    """
    states = data.observations
    actions = data.actions
    next_states = data.next_observations
    dones = data.dones.flatten()
    rewards = data.rewards

    batch_size = states.shape[0]

    

    # ----------------------------
    # Step 2: Find best next action using batched evaluation
    with torch.no_grad():
        next_states_expanded = next_states.unsqueeze(1).repeat(1, args.n_actions, 1)  # (batch, n_actions, state_dim)

        action_onehots = torch.eye(args.n_actions, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_actions, n_actions)

        next_states_flat = next_states_expanded.reshape(batch_size * args.n_actions, -1)  # (batch*n_actions, state_dim)
        action_onehots_flat = action_onehots.reshape(batch_size * args.n_actions, -1)  # (batch*n_actions, action_dim)

        q_vals_flat = sf_network(next_states_flat, action_onehots_flat, w)  # (batch*n_actions,)
        q_vals = q_vals_flat.view(batch_size, args.n_actions)  # (batch, n_actions)

        next_actions = q_vals.argmax(dim=1)  # (batch,)

    # ----------------------------
    # Step 3: Evaluate next best action value using target network
    next_action_onehot = torch.zeros((batch_size, args.n_actions), device=device)
    next_action_onehot.scatter_(1, next_actions.unsqueeze(1), 1.0)

    with torch.no_grad():
        target_q_values = sf_target_network(next_states, next_action_onehot, w).squeeze()

    td_target = rewards + args.gamma * target_q_values * (1 - dones)

    # ----------------------------
    # Step 4: Evaluate current Q-values
    action_onehot = torch.zeros((batch_size, args.n_actions), device=device)
    action_indices = actions.long()
    action_onehot.scatter_(1, action_indices, 1.0)

    current_q_values = sf_network(states, action_onehot, w).squeeze()
    # ----------------------------
    # Step 5: Loss and optimization
    loss = F.mse_loss(current_q_values, td_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    return loss, current_q_values, td_target

def train_dqn_online(q_network, target_network, discriminator, data, device, args, global_step, optimizer, model_idx_to_true_skill):
    states = data.observations
    next_states = data.next_observations

    # Step 1: Get internal skill idx used for one-hot encoding
    model_zs = data.observations[:, -args.n_skills_selected:].argmax(dim=1)  # [0–5]

    # Step 2: Map internal skill idx to true skill ID
    true_zs = torch.tensor([model_idx_to_true_skill[z.item()] for z in model_zs], dtype=torch.long, device=device)

    # Step 3: Compute intrinsic rewards
    logits = discriminator(next_states[:, :discriminator.input_dim])  # state only
    q_zs = F.softmax(logits, dim=-1).clamp(min=1e-6)
    logq_zs = torch.log(q_zs)
    logq_z = logq_zs[range(args.batch_size), true_zs]
    logpz = torch.tensor(1.0 / args.n_skills_total + 1e-6).log().to(device)  # Since discriminator trained on 25 skills
    intrinsic_rewards = (logq_z - logpz).detach()

    # TD Target
    with torch.no_grad():
        next_actions = q_network(next_states).argmax(dim=1, keepdim=True)
        next_q_values_target = target_network(next_states)
        target_q_values = next_q_values_target.gather(1, next_actions).squeeze()
        td_target = intrinsic_rewards + args.gamma * target_q_values * (1 - data.dones.flatten())

    old_val = q_network(states).gather(1, data.actions).squeeze()
    loss = F.mse_loss(td_target, old_val)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, old_val, intrinsic_rewards, logq_z.detach(), td_target, target_q_values

def phi_only(net, x):
    x1 = net.activation(net.fc1(x))
    x2 = net.activation(net.fc2(x1))
    x_res = x1 + x2
    x3 = net.activation(net.fc3(x_res))
    return net.fc4(x3)  # Return φ(s)

def train_w(task_vector , phi_net , data , optimizer_w):
    states = data.observations
    actions = data.actions
    next_states = data.next_observations
    rewards = data.rewards

    with torch.no_grad():
        phi_env = phi_net.forward2(next_states)

    pred_r_env_for_w = task_vector(phi_env)
    loss_w = F.mse_loss(pred_r_env_for_w, rewards)

    optimizer_w.zero_grad()
    loss_w.backward()
    optimizer_w.step()

    return loss_w