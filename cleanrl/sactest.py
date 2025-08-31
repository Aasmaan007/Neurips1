# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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
from torch.utils.tensorboard import SummaryWriter

#from cleanrl_utils.buffers import ReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class Args:
   exp_name: str = os.path.basename(__file__)[: -len(".py")]
   """the name of this experiment"""
   seed: int = 3
   """seed of the experiment"""
   torch_deterministic: bool = True
   """if toggled, torch.backends.cudnn.deterministic=False"""
   cuda: bool = True
   """if toggled, cuda will be enabled by default"""
   track: bool = False
   """if toggled, this experiment will be tracked with Weights and Biases"""
   wandb_project_name: str = "cleanRL"
   """the wandb's project name"""
   wandb_entity: str = None
   """the entity (team) of wandb's project"""
   capture_video: bool = False
   """whether to capture videos of the agent performances (check out videos folder)"""

   # Algorithm specific arguments
   env_id: str = "Hopper-v4"
   """the environment id of the task"""
   total_timesteps: int = 1000000
   """total timesteps of the experiments"""
   num_envs: int = 1
   """the number of parallel game environments"""
   buffer_size: int = int(1e6)
   """the replay memory buffer size"""
   gamma: float = 0.99
   """the discount factor gamma"""
   tau: float = 0.005
   """target smoothing coefficient (default: 0.005)"""
   batch_size: int = 256
   """the batch size of sample from the reply memory"""
   learning_starts: int = 5e3
   """timestep to start learning"""
   policy_lr: float = 3e-4
   """the learning rate of the policy network optimizer"""
   q_lr: float = 1e-3
   """the learning rate of the Q network network optimizer"""
   policy_frequency: int = 2
   """the frequency of training policy (delayed)"""
   target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
   """the frequency of updates for the target nerworks"""
   alpha: float = 0.2
   """Entropy regularization coefficient."""
   autotune: bool = False
   """automatic tuning of the entropy coefficient"""
       # KL/meta infusion hyperparams
   k_action_samples: int = 4
   """K actions sampled per state to estimate KL term (lower variance)"""
   std_ema_alpha: float = 0.02
   """EMA alpha for running std estimates of Qmin and Qmeta (0 < alpha <= 1)"""
   lambda_kl_init: float = 0.5
   """Initial weight (lambda) for KL(pi || Boltzmann(Qmeta))"""
   lambda_kl_final_frac: float = 0.3
   """Fraction of total_timesteps over which lambda_kl is annealed to zero.
      i.e. lambda reaches zero at t = lambda_kl_final_frac * total_timesteps"""
   kl_percentile_low: float = 0.05
   kl_percentile_high: float = 0.95
   """Per-state percentiles: discard Qmeta samples outside [low,high]"""
   w_path: str  = "runs/checkpoints/env_phi_task/Hopper-v4__joint_phi_task__1__2025-08-20_01-59-52/latest.pth"
   model_path = "runs/checkpoints/maml/Hopper-v4__MAML_SF__1__2025-08-19_23-42-42__1755627162/latest.pth"
   w_random: bool = False
   pretrained: bool = True



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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = np.prod(env.single_observation_space.shape)
        action_dim = np.prod(env.single_action_space.shape)
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

class SoftQNetwork(nn.Module):
   def __init__(self, env):
       super().__init__()
       self.fc1 = nn.Linear(
           np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
           256,
       )
       self.fc2 = nn.Linear(256, 256)
       self.fc3 = nn.Linear(256, 1)

   def forward(self, x, a):
       x = torch.cat([x, a], 1)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
   def __init__(self, env):
       super().__init__()
       self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
       self.fc2 = nn.Linear(256, 256)
       self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
       self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
       # action rescaling
       self.register_buffer(
           "action_scale",
           torch.tensor(
               (env.single_action_space.high - env.single_action_space.low) / 2.0,
               dtype=torch.float32,
           ),
       )
       self.register_buffer(
           "action_bias",
           torch.tensor(
               (env.single_action_space.high + env.single_action_space.low) / 2.0,
               dtype=torch.float32,
           ),
       )

   def forward(self, x):
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       mean = self.fc_mean(x)
       log_std = self.fc_logstd(x)
       log_std = torch.tanh(log_std)
       log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

       return mean, log_std

   def get_action(self, x):
       mean, log_std = self(x)
       std = log_std.exp()
       normal = torch.distributions.Normal(mean, std)
       x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
       y_t = torch.tanh(x_t)
       action = y_t * self.action_scale + self.action_bias
       log_prob = normal.log_prob(x_t)
       # Enforcing Action Bound
       log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
       log_prob = log_prob.sum(1, keepdim=True)
       mean = torch.tanh(mean) * self.action_scale + self.action_bias
       return action, log_prob, mean


if __name__ == "__main__":

   args = tyro.cli(Args)
   # running (EMA) std estimates for Qmin and Qmeta; start small but nonzero to avoid div0
   ema_qmin_std = 1.0
   ema_qmeta_std = 1.0
   eps_small = 1e-6
   K = args.k_action_samples
   ema_alpha = args.std_ema_alpha
   lambda_init = args.lambda_kl_init
   lambda_final_t = int(args.lambda_kl_final_frac * args.total_timesteps)
   p_low = args.kl_percentile_low
   p_high = args.kl_percentile_high

   run_name = f"{args.env_id}{args.exp_name}{args.seed}{int(time.time())}"
   if args.track:
       import wandb

       wandb.init(
           project=args.wandb_project_name,
           entity=args.wandb_entity,
           sync_tensorboard=True,
           config=vars(args),
           name=run_name,
           monitor_gym=True,
           save_code=True,
       )
   writer = SummaryWriter(f"runs/cont/{run_name}")
   writer.add_text(
       "hyperparameters",
       "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
   )

   # TRY NOT TO MODIFY: seeding
   random.seed(args.seed)
   np.random.seed(args.seed)
   torch.manual_seed(args.seed)
   torch.backends.cudnn.deterministic = args.torch_deterministic

   device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

   # env setup
   envs = gym.vector.SyncVectorEnv(
       [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
   )
   assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

   max_action = float(envs.single_action_space.high[0])

   actor = Actor(envs).to(device)
   qf1 = SoftQNetwork(envs).to(device)
   qf2 = SoftQNetwork(envs).to(device)
   qf1_target = SoftQNetwork(envs).to(device)
   qf2_target = SoftQNetwork(envs).to(device)
   qf1_target.load_state_dict(qf1.state_dict())
   qf2_target.load_state_dict(qf2.state_dict())
   q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
   actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

   qmeta_net = QNetwork(envs).to(device)
    

   if(args.pretrained):
        checkpoint2 = torch.load(args.model_path, map_location="cuda:0")
        sf_state_dict = checkpoint2["sfmeta_network_state_dict"]
        mapped_state_dict = {}
        mapped_state_dict["embedding.0.weight"] = sf_state_dict["l1.weight"]
        mapped_state_dict["embedding.0.bias"]   = sf_state_dict["l1.bias"]
        mapped_state_dict["embedding.2.weight"] = sf_state_dict["l2.weight"]
        mapped_state_dict["embedding.2.bias"]   = sf_state_dict["l2.bias"]
        mapped_state_dict["embedding.4.weight"] = sf_state_dict["l3.weight"]
        mapped_state_dict["embedding.4.bias"]   = sf_state_dict["l3.bias"]
        qmeta_net.load_state_dict(mapped_state_dict)


   w = torch.randn(32).to(device)
   w = w / (w.norm() + 1e-8)
   task_vector = TaskVector(32).to(device)
   checkpoint1 = torch.load(args.w_path)
   if(not args.w_random):
        task_vector.load_state_dict(checkpoint1["task_vector"])
   w = (task_vector.w / (torch.norm(task_vector.w) + 1e-8)).detach()
    


   # Automatic entropy tuning
   if args.autotune:
       target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
       log_alpha = torch.zeros(1, requires_grad=True, device=device)
       alpha = log_alpha.exp().item()
       a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
   else:
       alpha = args.alpha

   envs.single_observation_space.dtype = np.float32
   rb = ReplayBuffer(
       args.buffer_size,
       envs.single_observation_space,
       envs.single_action_space,
       device,
       n_envs=args.num_envs,
       handle_timeout_termination=False,
   )
   start_time = time.time()

   # TRY NOT TO MODIFY: start the game
   obs, _ = envs.reset(seed=args.seed)
   for global_step in range(args.total_timesteps):
       # ALGO LOGIC: put action logic here
       if global_step < args.learning_starts:
           actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
       else:
           actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
           actions = actions.detach().cpu().numpy()

       # TRY NOT TO MODIFY: execute the game and log data.
       next_obs, rewards, terminations, truncations, infos = envs.step(actions)

       # TRY NOT TO MODIFY: record rewards for plotting purposes
       if "final_info" in infos:
           for info in infos["final_info"]:
               if info is not None:
                   print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                   writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                   writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                   break

       # TRY NOT TO MODIFY: save data to reply buffer; handle final_observation
       real_next_obs = next_obs.copy()
       for idx, trunc in enumerate(truncations):
           if trunc:
               real_next_obs[idx] = infos["final_observation"][idx]
       rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

       # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
       obs = next_obs

       # ALGO LOGIC: training.
       if global_step > args.learning_starts:
           data = rb.sample(args.batch_size)
           with torch.no_grad():
               next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
               qf1_next_target = qf1_target(data.next_observations, next_state_actions)
               qf2_next_target = qf2_target(data.next_observations, next_state_actions)
               min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
               next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

           qf1_a_values = qf1(data.observations, data.actions).view(-1)
           qf2_a_values = qf2(data.observations, data.actions).view(-1)
           qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
           qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
           qf_loss = qf1_loss + qf2_loss

           # optimize the model
           q_optimizer.zero_grad()
           qf_loss.backward()
           q_optimizer.step()

           if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
               for _ in range(args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                  
           # ------- start replacement block -------
                   # data.observations: [B, obs_dim]
                   B = data.observations.shape[0]
                   device = data.observations.device

                   # 1) Expand states to get K samples per state
                   obs_rep = data.observations.repeat_interleave(K, dim=0)  # [B*K, obs_dim]

                   # 2) Get reparam actions + corrected log_pi for all samples
                   pi_rep, log_pi_rep, _ = actor.get_action(obs_rep)       # shapes [B*K, act_dim], [B*K, 1]

                   # 3) Q estimates for actor samples (used in SAC actor loss)
                   qf1_pi_rep = qf1(obs_rep, pi_rep).view(-1)               # [B*K]
                   qf2_pi_rep = qf2(obs_rep, pi_rep).view(-1)               # [B*K]
                   min_q_pi_rep = torch.min(qf1_pi_rep, qf2_pi_rep)         # [B*K]

                   # 4) Compute qmeta at these samples — frozen (no grad)
                   with torch.no_grad():
                       qmeta_rep = qmeta_net(obs_rep, pi_rep)      # [B*K]; qmeta_net is your meta Qnet (must exist)
                       qmeta_rep = torch.einsum("bd,d->b", qmeta_rep, w).view(-1)
                       

                   # 5) Reshape per state to compute per-state statistics: [B, K]
                   qmeta_rs = qmeta_rep.view(B, K)
                   qmin_rs = min_q_pi_rep.view(B, K)

                   # 6) Compute batch-level robust stds (per-state then aggregate) WITHOUT grads
                   with torch.no_grad():
                       # per-state std (use population std to be stable), avoid unbiased estimator for stability
                       perstate_qmeta_std = qmeta_rs.std(dim=1, unbiased=False)   # [B]
                       perstate_qmin_std  = qmin_rs.std(dim=1, unbiased=False)    # [B]

                       # aggregate into batch-level stat; use median to be robust to state variability
                       batch_qmeta_std = perstate_qmeta_std.median().item()
                       batch_qmin_std  = perstate_qmin_std.median().item()

                       # update EMAs
                       ema_qmeta_std = (1.0 - ema_alpha) * ema_qmeta_std + ema_alpha * (batch_qmeta_std + eps_small)
                       ema_qmin_std  = (1.0 - ema_alpha) * ema_qmin_std  + ema_alpha * (batch_qmin_std + eps_small)

                   # 7) scale qmeta to qmin scale (avoid division by zero)
                   scale = (ema_qmin_std / (ema_qmeta_std + eps_small))
                   qmeta_scaled_rep = qmeta_rep * scale   # [B*K]

                   # 8) percentile-based discard (per-state)
                   # compute per-state percentiles (no grad)
                   with torch.no_grad():
                       qmeta_rs_scaled = qmeta_scaled_rep.view(B, K)   # [B, K]
                       p_lo = qmeta_rs_scaled.quantile(p_low, dim=1, keepdim=True)   # [B,1]
                       p_hi = qmeta_rs_scaled.quantile(p_high, dim=1, keepdim=True)  # [B,1]
                       # mask: True if within [p_lo, p_hi]
                       keep_mask_rs = (qmeta_rs_scaled >= p_lo) & (qmeta_rs_scaled <= p_hi)  # [B,K]

                   # flatten mask to [B*K]
                   keep_mask = keep_mask_rs.view(-1)  # boolean

                   # 9) KL term estimate: E_{s,a~pi}[ log_pi - qmeta_scaled ] averaged over kept samples
                   log_pi_flat = log_pi_rep.view(-1)              # [B*K]
                   qmeta_scaled_flat = qmeta_scaled_rep.view(-1)  # [B*K]

                   if keep_mask.any():
                       kl_term = (log_pi_flat[keep_mask] - qmeta_scaled_flat[keep_mask]).mean()
                   else:
                       kl_term = torch.tensor(0.0, device=device)

                   # 10) SAC actor loss (computed on all samples as usual)
                   sac_actor_loss = ((alpha * log_pi_flat) - min_q_pi_rep).mean()

                   # 11) Anneal lambda linearly to zero at lambda_final_t
                   if global_step >= lambda_final_t:
                       lambda_t = 0.0
                   else:
                       lambda_t = lambda_init * (1.0 - (global_step / float(max(1, lambda_final_t))))

                   # 12) Final actor loss = sac_actor_loss + lambda_t * kl_term
                   actor_loss = sac_actor_loss + (lambda_t * kl_term)

                   # 13) backward + step (same as before)
                   actor_optimizer.zero_grad()
                   actor_loss.backward()
                   actor_optimizer.step()
                   # ------- end replacement block -------
       
                  
                   if args.autotune:
                       with torch.no_grad():
                           _, log_pi, _ = actor.get_action(data.observations)
                       alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                       a_optimizer.zero_grad()
                       alpha_loss.backward()
                       a_optimizer.step()
                       alpha = log_alpha.exp().item()

           # update the target networks
           if global_step % args.target_network_frequency == 0:
               for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                   target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
               for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                   target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

           if global_step % 100 == 0:
               writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
               writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
               writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
               writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
               writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
               writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
               writer.add_scalar("losses/sac_actor_loss", sac_actor_loss.item(), global_step)
               writer.add_scalar("losses/alpha", alpha, global_step)
               writer.add_scalar("losses/kl_term", kl_term.item(), global_step)
               writer.add_scalar("meta/lambda_kl", lambda_t, global_step)
               writer.add_scalar("meta/ema_qmin_std", ema_qmin_std, global_step)
               writer.add_scalar("meta/ema_qmeta_std", ema_qmeta_std, global_step)
               writer.add_scalar("meta/scale_qmeta_to_qmin", scale, global_step)
               writer.add_scalar("meta/lambda_t", lambda_t, global_step)
               writer.add_scalar("meta/frac_kept", keep_mask.float().mean().item(), global_step)
               writer.add_scalar("meta/qmeta_rep", qmeta_rep.mean().item(), global_step)
               writer.add_scalar("meta/qmeta_scaled", qmeta_scaled_flat.mean().item(), global_step)
               writer.add_scalar("meta/qmeta_scaled_masked", qmeta_scaled_flat[keep_mask].mean().item(), global_step)
               writer.add_scalar("meta/qmin_pi", min_q_pi_rep.mean().item(), global_step)
               print("SPS:", int(global_step / (time.time() - start_time)))
               writer.add_scalar(
                   "charts/SPS",
                   int(global_step / (time.time() - start_time)),
                   global_step,
               )
               if args.autotune:
                   writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                

           if global_step % 1000 == 0:
               writer.add_scalar("intrinsic_rewards", data.rewards.mean(), global_step)

   envs.close()
   writer.close()