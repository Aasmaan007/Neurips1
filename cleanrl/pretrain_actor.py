# docs and experiment results: https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# Stage 1: pretrain actor against meta-Q (KL to Boltz(Q_meta))
# Stage 2: PURE SAC (no KL term, no meta compute)

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
from torch.utils.data import TensorDataset, DataLoader

# from cleanrl_utils.buffers import ReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from cleanrl.diayn.models_cont import Discriminator

import pickle


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 14
    torch_deterministic: bool = True
    cuda: bool = True
    cuda_device: str = "cuda"   # e.g. "cuda", "cuda:0", "cuda:1"
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False

    # SAC
    env_id: str = "Hopper-v4"
    total_timesteps: int = 1_000_000
    num_envs: int = 1
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = False

    # --- meta pretrain (Stage 1) ---
    pretrain: bool = True
    pretrain_collect_steps: int = 5000      # random-acting steps to populate buffer
    pretrain_steps: int = 20000             # actor-only updates vs meta
    pretrain_batch_size: int = 256
    pretrain_K: int = 4                     # actions per state to estimate KL
    pretrain_p_lo: float = 0.05
    pretrain_p_hi: float = 0.95
    pretrain_lr: float = 3e-4               # actor LR during pretrain
    std_ema_alpha: float = 0.02             # EMA for calibration stats
    log_every_pretrain: int = 500

    # Paths / pretrained
    w_path: str = "runs/checkpoints/env_phi_task/Hopper-v4__joint_phi_task__1__2025-08-20_01-59-52/latest.pth"
    model_path: str = "runs/checkpoints/maml/Hopper-v4__MAML_SF__1__2025-08-28_00-08-21__1756319901/latest.pth"
    disc_path: str = "runs/checkpoints/qtargetmaml/Hopper-v4__q_online__1__2025-08-19_12-44-30__1755587670/latest.pth"
    data_path: str = "runs/data/Hopper-v4__unified_collection_1__2025-08-19_12-03-05__1755585185/maml_training_data.pkl"
    qnet_path: str = "runs/checkpoints/qtargetmaml/Hopper-v4__q_online__1__2025-08-19_00-13-10__1755542590/latest.pth"
    w_random: bool = False
    pretrained: bool = True
    n_skills_total: int = 25

torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # needed by deterministic_algorithms on CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("high")  # keeps FP32, disables TF32 drift


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


def get_all_pairs(states):
    all_states= []
    all_actions = []
    for s,a in states:
        all_states.append(s)
        all_actions.append(a)
    return torch.tensor(np.stack(all_states), dtype=torch.float32), torch.tensor(np.stack(all_actions), dtype=torch.float32)

# --------- Networks ----------
class QNetwork(nn.Module):
    """Meta-Q embedding network producing phi(s,a) in R^32."""
    def __init__(self, env):
        super().__init__()
        state_dim = int(np.prod(env.single_observation_space.shape))
        action_dim = int(np.prod(env.single_action_space.shape))
        self.input_dim = state_dim + action_dim
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 32),
        )

    def forward(self, state, action_cont):
        x = torch.cat([state, action_cont], dim=-1)
        return self.embedding(x)  # [B, 32]
    
class Qonlinenet(nn.Module):
    def __init__(self, env, nskills: int):
        super().__init__()
        base_obs_dim = int(np.prod(env.observation_space.shape))
        obs_dim = base_obs_dim + nskills
        act_dim = int(np.prod(env.action_space.shape))
        hidden = 256
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, state, action, extra_vec):
        # state is assumed already concatenated with one-hot skill
        x = torch.cat([state, extra_vec, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            int(np.array(env.single_observation_space.shape).prod())
            + int(np.prod(env.single_action_space.shape)),
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
        self.fc1 = nn.Linear(int(np.array(env.single_observation_space.shape).prod()), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, int(np.prod(env.single_action_space.shape)))
        self.fc_logstd = nn.Linear(256, int(np.prod(env.single_action_space.shape)))
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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# ------- helpers for meta KL calibration (Stage 1 only) -------
@torch.no_grad()
def _affine_match(qmeta_rep, min_q_rep, B, K, ema_state, ema_alpha=0.02, eps_small=1e-6):
    """
    Compute an EMA'd affine mapping (scale, shift) so that:
      qmeta_scaled = scale * qmeta_rep + shift
    roughly matches the location/scale of min_q across states, using robust (median) aggregation.
    """
    qmeta_rs = qmeta_rep.view(B, K)
    qmin_rs = min_q_rep.view(B, K)

    per_mu_meta = qmeta_rs.mean(dim=1)
    per_sd_meta = qmeta_rs.std(dim=1, unbiased=False)
    per_mu_min = qmin_rs.mean(dim=1)
    per_sd_min = qmin_rs.std(dim=1, unbiased=False)

    batch_mu_meta = per_mu_meta.median().item()
    batch_sd_meta = per_sd_meta.median().item()
    batch_mu_min = per_mu_min.median().item()
    batch_sd_min = per_sd_min.median().item()

    if ema_state["mu_meta"] is None:
        ema_state["mu_meta"] = batch_mu_meta
        ema_state["sd_meta"] = batch_sd_meta + eps_small
        ema_state["mu_min"] = batch_mu_min
        ema_state["sd_min"] = batch_sd_min + eps_small
    else:
        ema_state["mu_meta"] = (1 - ema_alpha) * ema_state["mu_meta"] + ema_alpha * batch_mu_meta
        ema_state["sd_meta"] = (1 - ema_alpha) * ema_state["sd_meta"] + ema_alpha * (batch_sd_meta + eps_small)
        ema_state["mu_min"] = (1 - ema_alpha) * ema_state["mu_min"] + ema_alpha * batch_mu_min
        ema_state["sd_min"] = (1 - ema_alpha) * ema_state["sd_min"] + ema_alpha * (batch_sd_min + eps_small)

    scale = ema_state["sd_min"] / (ema_state["sd_meta"] + eps_small)
    shift = ema_state["mu_min"] - scale * ema_state["mu_meta"]
    return scale, shift


def meta_kl_loss(actor, qf1, qf2, qonline_net, w, obs_batch, K, p_lo, p_hi, ema_state, ema_alpha=0.02):
    """
    KL(π || Boltz(Q_meta)) estimate using K actions/state.
    Returns (kl_term, frac_kept, scale_used)
    """
    B = obs_batch.shape[0]
    device = obs_batch.device

    obs_rep = obs_batch.repeat_interleave(K, dim=0)  # [B*K, obs_dim]
    pi_rep, log_pi_rep, _ = actor.get_action(obs_rep)  # [B*K, act], [B*K,1]
    log_pi_flat = log_pi_rep.view(-1)

    with torch.no_grad():
        q1 = qf1(obs_rep, pi_rep).view(-1)
        q2 = qf2(obs_rep, pi_rep).view(-1)
        qmin = torch.min(q1, q2)  # [B*K]

        #qmeta_emb = qmeta_net(obs_rep, pi_rep)  # [B*K, D]
        #qmeta_rep = torch.einsum("bd,d->b", qmeta_emb, w).view(-1)  # [B*K]
        qmeta_rep  = qonline_net(obs_rep, pi_rep, extra_vec.repeat_interleave(K, dim=0)).view(-1)

        scale, shift = _affine_match(qmeta_rep, qmin, B, K, ema_state, ema_alpha)
        qmeta_scaled = qmeta_rep * 1 #scale + shift

        plo = torch.quantile(qmeta_scaled, p_lo)
        phi = torch.quantile(qmeta_scaled, p_hi)
        keep_mask = (qmeta_scaled >= plo) & (qmeta_scaled <= phi)

    if keep_mask.any():
        kl_term = (log_pi_flat[keep_mask] - qmeta_scaled[keep_mask]).mean()
    else:
        kl_term = torch.tensor(0.0, device=device)

    return kl_term, keep_mask.float().mean().item(), float(scale)


def cem_improve_action(state, a_D, qmeta_net, lambda_penalty=1.0,
                       N=64, K=6, T=3, sigma0=0.2, action_low=-1.0, action_high=1.0):
    """
    state: [obs_dim] tensor
    a_D: [act_dim] tensor (dataset action)
    qmeta_net: function(state_batch, action_batch) -> Q values
    """

    act_dim = a_D.shape[-1]
    mean = a_D.clone()
    cov = (sigma0 ** 2) * torch.eye(act_dim, device=state.device)

    for t in range(T):
        # 1. Sample N candidates
        mvn = torch.distributions.MultivariateNormal(mean, cov)
        actions = mvn.sample((N,))
        actions = torch.clamp(actions, action_low, action_high)

        # 2. Evaluate penalized Q
        states = state.unsqueeze(0).repeat(N, 1)
        q_vals = qmeta_net(states, actions).squeeze(-1)  # [N]
        penalties = lambda_penalty * ((actions - a_D)**2).sum(dim=-1)
        scores = q_vals - penalties

        # 3. Select top-K
        topk_idx = torch.topk(scores, K, dim=0).indices
        elites = actions[topk_idx]

        # 4. Update mean/cov
        mean = elites.mean(dim=0)
        cov = torch.from_numpy(np.cov(elites.cpu().numpy().T)).float().to(state.device)
        # add small epsilon to avoid singular
        cov += 1e-6 * torch.eye(act_dim, device=state.device)

    return mean.detach()  # improved action



# ============================ MAIN ============================
if __name__ == "__main__":
    args = tyro.cli(Args)
    K = args.pretrain_K
    # Logging setup
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_pre{int(args.pretrain)}_{int(time.time())}"
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
    writer = SummaryWriter(f"runs/cont1/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    with open(args.data_path, "rb") as f:
        # state_data = pickle.load(f)
        # np.random.shuffle(state_data)
        state_data = pickle.load(f)
        np.random.shuffle(state_data)
    # state_data = np.array(state_data)
    # np.random.shuffle(state_data)
    state_data = np.array(state_data)
    np.random.shuffle(state_data)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.cuda.manual_seed_all(args.seed)  # <-- IMPORTANT for all CUDA devices

    device = torch.device(args.cuda_device if torch.cuda.is_available() and args.cuda else "cpu")

    # Env
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    envs.single_observation_space.dtype = np.float32
    max_action = float(envs.single_action_space.high[0])

    # Models
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

    # Discriminator -> meta vector w
    state_dim = int(np.array(envs.single_observation_space.shape).prod())
    discriminator = Discriminator(state_dim, args.n_skills_total)
    disc_ckpt = torch.load(args.disc_path, map_location="cpu")
    discriminator.load_state_dict(disc_ckpt['disc_state_dict'])
    #discriminator.load_state_dict(torch.load(args.disc_path, map_location="cpu")["disc_state_dict"])
    discriminator = discriminator.to(device)


    extra_vec = torch.tensor([[1, 0, 0, 0, 0, 0]] * args.batch_size,  device=device)
    qonline_net = Qonlinenet(envs, 6).to(device)

    if(args.pretrained):
        checkpoint3 = torch.load(args.qnet_path, map_location="cpu")
        qnet_state_dict = checkpoint3["q_network_state_dict"]
        mapped_state_dict = {}
        mapped_state_dict["fc1.weight"] = qnet_state_dict["fc1.weight"]
        mapped_state_dict["fc1.bias"]   = qnet_state_dict["fc1.bias"]
        mapped_state_dict["fc2.weight"] = qnet_state_dict["fc2.weight"]
        mapped_state_dict["fc2.bias"]   = qnet_state_dict["fc2.bias"]
        mapped_state_dict["fc3.weight"] = qnet_state_dict["fc3.weight"]
        mapped_state_dict["fc3.bias"]   = qnet_state_dict["fc3.bias"]
        qonline_net.load_state_dict(mapped_state_dict)
    qonline_net = qonline_net.to(device)

    # map pretrained meta model if available
    if args.pretrained:
        checkpoint2 = torch.load(args.model_path, map_location="cpu")
        sf_state_dict = checkpoint2["sfmeta_network_state_dict"]
        mapped_state_dict = {
            "embedding.0.weight": sf_state_dict["l1.weight"],
            "embedding.0.bias": sf_state_dict["l1.bias"],
            "embedding.2.weight": sf_state_dict["l2.weight"],
            "embedding.2.bias": sf_state_dict["l2.bias"],
            "embedding.4.weight": sf_state_dict["l3.weight"],
            "embedding.4.bias": sf_state_dict["l3.bias"],
        }
        qmeta_net.load_state_dict(mapped_state_dict)
    qmeta_net = qmeta_net.to(device)

    w = discriminator.q.weight[1].detach().to(device)
    w = w / (w.norm() + 1e-8)

    # Alpha autotune
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    # ========= Stage 1: Meta pretrain (actor-only, no critic/alpha updates) =========
    ema_state = {"mu_meta": None, "sd_meta": None, "mu_min": None, "sd_min": None}
    obs, _ = envs.reset(seed=args.seed)
    envs.action_space.seed(args.seed)
    states,actions = get_all_pairs(state_data)
    train_ds = TensorDataset(states.cpu())
    train_loader = DataLoader(
    train_ds,
    batch_size=args.pretrain_batch_size,   # e.g., 256
    shuffle=True,                          # reshuffle each epoch
    drop_last=True,                        # cleaner batch sizes
    pin_memory=True
)
    
    num_epochs = 10            # <-- set how many passes you want
    grad_clip = 1.0
    K = args.pretrain_K

    #qf1.eval(); qf2.eval()     # critics used only for calibration
    actor.train()

    if args.pretrain:
        # print("==> Meta pretrain: collecting random transitions...")
        # for t in range(args.pretrain_collect_steps):
        #     actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        #     next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        #     real_next_obs = next_obs.copy()
        #     for idx, trunc in enumerate(truncations):
        #         if trunc:
        #             real_next_obs[idx] = infos["final_observation"][idx]
        #     rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        #     obs = next_obs

        print("==> Meta pretrain: optimizing actor to match Boltzmann(Q_meta)...")
        pretrain_actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.pretrain_lr)
        global_train_step = 0
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            for (batch_states_cpu,) in train_loader:
                batch_states = batch_states_cpu.to(device, non_blocking=True)

                kl_term, frac_kept, scale_used = meta_kl_loss(
                    actor=actor, qf1=qf1, qf2=qf2, qonline_net=qonline_net, w=w,
                    obs_batch=batch_states, K=K,
                    p_lo=args.pretrain_p_lo, p_hi=args.pretrain_p_hi,
                    ema_state=ema_state, ema_alpha=args.std_ema_alpha,
                )
                loss = kl_term

                pretrain_actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                pretrain_actor_optimizer.step()

                epoch_loss += loss.item()
                if (global_train_step % args.log_every_pretrain) == 0:
                    writer.add_scalar("pretrain/step_loss", loss.item(), global_train_step)
                    writer.add_scalar("pretrain/frac_kept", frac_kept, global_train_step)
                    writer.add_scalar("pretrain/scale", scale_used, global_train_step)

                global_train_step += 1

            epoch_loss /= max(1, len(train_loader))
            writer.add_scalar("pretrain/epoch_loss", epoch_loss, epoch)
            print(f"[pretrain] epoch {epoch:02d} | train_KL {epoch_loss:.4f}")

        # for step in range(args.pretrain_steps):
        #     num_states=states.shape[0]
        #     indices = torch.randint(0, num_states, (args.batch_size,), device=device)
        #     sampled_states = states[indices].to(device)
        #     #data = rb.sample(args.pretrain_batch_size)
        #     kl_term, frac_kept, scale_used = meta_kl_loss(
        #         actor=actor,
        #         qf1=qf1,
        #         qf2=qf2,
        #         qmeta_net=qmeta_net,
        #         w=w,
        #         obs_batch=sampled_states,
        #         K=args.pretrain_K,
        #         p_lo=args.pretrain_p_lo,
        #         p_hi=args.pretrain_p_hi,
        #         ema_state=ema_state,
        #         ema_alpha=args.std_ema_alpha,
        #     )
        #     actor_loss_pre = kl_term  # minimize KL(π || Boltz(Q_meta))

        #     pretrain_actor_optimizer.zero_grad()
        #     actor_loss_pre.backward()
        #     pretrain_actor_optimizer.step()

        #     if (step % args.log_every_pretrain) == 0:
        #         writer.add_scalar("pretrain/actor_loss_meta", actor_loss_pre.item(), step)
        #         writer.add_scalar("pretrain/frac_kept", frac_kept, step)
        #         writer.add_scalar("pretrain/scale", scale_used, step)
        #         writer.add_scalar("pretrain/ema_sd_meta", float(ema_state["sd_meta"]), step)
        #         writer.add_scalar("pretrain/ema_sd_min", float(ema_state["sd_min"]), step)

        # reset env for SAC proper
        obs, _ = envs.reset(seed=args.seed)
        print("==> Pretrain done. Proceed to PURE SAC...")
    
    model_dir = f"runs/checkpoints/pretrain_actor/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
            "pretrain_actor_state_dict": actor.state_dict(),
        }, os.path.join(model_dir, f"latest.pth"))
    
    envs.close()
    writer.close()

        # TRY NOT TO MODIFY: start the game
    obse, _ = envs.reset(seed=args.seed)
    
    for n in range(25):
        ep_return = 0
        terminations = False
        while not terminations:
            actions, _, _ = actor.get_action(torch.Tensor(obse).to(device))
            actions = actions.detach().cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            ep_return += rewards
            obse = next_obs
        print(f"eval episode {n} return {ep_return}")
        writer.add_scalar("pretrain/pretrained_ep_return", ep_return, n)




    # ========= Stage 2: PURE SAC (no KL, no meta compute) =========
    # start_time = time.time()

    # for global_step in range(args.total_timesteps):
    #     # action
    #     if global_step < args.learning_starts:
    #         actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    #     else:
    #         with torch.no_grad():
    #             actions, _, _ = actor.get_action(torch.tensor(obs, dtype=torch.float32, device=device))
    #             actions = actions.detach().cpu().numpy()

    #     # step
    #     next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    #     # episodic logs
    #     if "final_info" in infos:
    #         for info in infos["final_info"]:
    #             if info is not None:
    #                 print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
    #                 writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
    #                 writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
    #                 break

    #     # add to buffer (handle truncation)
    #     real_next_obs = next_obs.copy()
    #     for idx, trunc in enumerate(truncations):
    #         if trunc:
    #             real_next_obs[idx] = infos["final_observation"][idx]
    #     rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

    #     obs = next_obs

    #     # updates
    #     if global_step > args.learning_starts:
    #         data = rb.sample(args.batch_size)

    #         # --- Critic update ---
    #         with torch.no_grad():
    #             next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
    #             qf1_next_target = qf1_target(data.next_observations, next_state_actions)
    #             qf2_next_target = qf2_target(data.next_observations, next_state_actions)
    #             min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
    #             next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
    #                 min_qf_next_target
    #             ).view(-1)

    #         qf1_a_values = qf1(data.observations, data.actions).view(-1)
    #         qf2_a_values = qf2(data.observations, data.actions).view(-1)
    #         qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    #         qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    #         qf_loss = qf1_loss + qf2_loss

    #         q_optimizer.zero_grad()
    #         qf_loss.backward()
    #         q_optimizer.step()

    #         # --- Policy update (delayed) ---
    #         if global_step % args.policy_frequency == 0:
    #             for _ in range(args.policy_frequency):
    #                 pi, log_pi, _ = actor.get_action(data.observations)
    #                 qf1_pi = qf1(data.observations, pi)
    #                 qf2_pi = qf2(data.observations, pi)
    #                 min_q_pi = torch.min(qf1_pi, qf2_pi)
    #                 sac_actor_loss = (alpha * log_pi - min_q_pi).mean()

    #                 actor_optimizer.zero_grad()
    #                 sac_actor_loss.backward()
    #                 actor_optimizer.step()

    #             # alpha autotune
    #             if args.autotune:
    #                 with torch.no_grad():
    #                     _, log_pi_a, _ = actor.get_action(data.observations)
    #                 alpha_loss = (-log_alpha.exp() * (log_pi_a + target_entropy)).mean()
    #                 a_optimizer.zero_grad()
    #                 alpha_loss.backward()
    #                 a_optimizer.step()
    #                 alpha = log_alpha.exp().item()

    #         # target networks
    #         if global_step % args.target_network_frequency == 0:
    #             for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
    #                 target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    #             for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
    #                 target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    #         # logging
    #         if global_step % 100 == 0:
    #             writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
    #             writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
    #             writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
    #             writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
    #             writer.add_scalar("losses/qf_loss", (qf_loss.item() / 2.0), global_step)
    #             writer.add_scalar("losses/actor_loss", sac_actor_loss.item(), global_step)
    #             writer.add_scalar("losses/alpha", alpha, global_step)

    #             sps = int(global_step / (time.time() - start_time))
    #             print("SPS:", sps)
    #             writer.add_scalar("charts/SPS", sps, global_step)
    #             if args.autotune:
    #                 writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    #         if global_step % 1000 == 0:
    #             writer.add_scalar("intrinsic_rewards", float(data.rewards.mean().item()), global_step)

