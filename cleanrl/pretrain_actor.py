# docs and experiment results: https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# Stage 1: pretrain actor offline using CEM against meta-Q (qmeta_net + w)
# Stage 2: (commented) PURE SAC

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
    pretrain_collect_steps: int = 5000
    pretrain_steps: int = 20000
    pretrain_batch_size: int = 256
    pretrain_K: int = 4
    pretrain_p_lo: float = 0.05
    pretrain_p_hi: float = 0.95
    pretrain_lr: float = 3e-4
    std_ema_alpha: float = 0.02
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

    # ---------- CEM knobs ----------
    cem_lambda_penalty: float = 0.05   # trust penalty weight ||a - a_D||^2
    cem_N: int = 64                    # candidates per state
    cem_K: int = 6                     # elites per state
    cem_T: int = 3                     # CEM iterations
    cem_sigma0: float = 0.2            # initial std as fraction of action range

torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("high")


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
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# ======= Helpers for CEM distill =======
def atanh_safe(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    u = u.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(u) - torch.log1p(-u))

def actor_nll_on_actions(actor: Actor, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    mean, log_std = actor(states)
    std = log_std.exp()
    y = (actions - actor.action_bias) / (actor.action_scale + 1e-6)  # [-1,1]
    x = atanh_safe(y)  # pre-tanh
    normal = torch.distributions.Normal(mean, std)
    log_prob_x = normal.log_prob(x).sum(dim=1, keepdim=True)
    log_det = torch.log(actor.action_scale * (1 - y.pow(2)) + 1e-6).sum(dim=1, keepdim=True)
    log_prob_a = log_prob_x - log_det
    return -(log_prob_a.mean())

@torch.no_grad()
def cem_improve_actions_batch(
    states: torch.Tensor,
    a_D: torch.Tensor,
    qmeta_fn,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    lambda_penalty: float = 0.05,
    N: int = 64,
    K: int = 6,
    T: int = 3,
    sigma0: float = 0.2,
) -> torch.Tensor:
    device = states.device
    B, act_dim = a_D.shape
    act_range = (action_high - action_low).abs()
    mean = a_D.clone()
    std = torch.full_like(a_D, sigma0) * act_range
    for _ in range(T):
        eps = torch.randn(B, N, act_dim, device=device)
        A = mean.unsqueeze(1) + std.unsqueeze(1) * eps
        A = A.clamp(action_low.unsqueeze(1), action_high.unsqueeze(1))  # [B,N,act]
        S = states.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        A_flat = A.reshape(B * N, act_dim)
        Q = qmeta_fn(S, A_flat).view(B, N)  # [B,N]
        penalty = lambda_penalty * ((A - a_D.unsqueeze(1)).pow(2).sum(dim=-1))
        scores = Q - penalty
        topk_vals, topk_idx = scores.topk(K, dim=1)
        idx_exp = topk_idx.unsqueeze(-1).expand(B, K, act_dim)
        elites = torch.gather(A, dim=1, index=idx_exp)  # [B,K,act]
        mean = elites.mean(dim=1)
        std = elites.std(dim=1, unbiased=False) + 1e-6
    return mean  # a*


# ============================ MAIN ============================
if __name__ == "__main__":
    args = tyro.cli(Args)
    K = args.pretrain_K
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
        state_data = pickle.load(f)
        np.random.shuffle(state_data)
    state_data = np.array(state_data)
    np.random.shuffle(state_data)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.cuda.manual_seed_all(args.seed)

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

    # Meta net (embedding) + load weights
    qmeta_net = QNetwork(envs).to(device)
    if args.pretrained:
        checkpoint2 = torch.load(args.model_path, map_location="cpu")
        sf_state_dict = checkpoint2["sfmeta_network_state_dict"]
        mapped_state_dict = {
            "embedding.0.weight": sf_state_dict["l1.weight"],
            "embedding.0.bias":   sf_state_dict["l1.bias"],
            "embedding.2.weight": sf_state_dict["l2.weight"],
            "embedding.2.bias":   sf_state_dict["l2.bias"],
            "embedding.4.weight": sf_state_dict["l3.weight"],
            "embedding.4.bias":   sf_state_dict["l3.bias"],
        }
        qmeta_net.load_state_dict(mapped_state_dict)
    qmeta_net = qmeta_net.to(device)
    qmeta_net.eval()

    # Discriminator -> meta vector w
    state_dim = int(np.array(envs.single_observation_space.shape).prod())
    discriminator = Discriminator(state_dim, args.n_skills_total)
    disc_ckpt = torch.load(args.disc_path, map_location="cpu")
    discriminator.load_state_dict(disc_ckpt['disc_state_dict'])
    discriminator = discriminator.to(device)

    w = discriminator.q.weight[11].detach().to(device)
    w = w / (w.norm() + 1e-8)

    # Alpha autotune (unchanged)
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Replay buffer (unchanged)
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    # ========= Stage 1: Offline pretrain via CEM distillation (using qmeta_net + w) =========
    obs, _ = envs.reset(seed=args.seed)
    envs.action_space.seed(args.seed)
    states, actions = get_all_pairs(state_data)

    train_ds = TensorDataset(states.cpu(), actions.cpu())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.pretrain_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    
    num_epochs = 10
    grad_clip = 1.0
    actor.train()

    @torch.no_grad()
    def qmeta_fn(S, A):
        """Return q_meta(s,a) = <phi(s,a), w> using qmeta_net + w."""
        emb = qmeta_net(S, A)          # [B, 32]
        q = torch.einsum("bd,d->b", emb, w)  # [B]
        return q.unsqueeze(-1)         # [B,1]

    if args.pretrain:
        print("==> Offline pretrain: CEM improving dataset actions and distilling into actor (qmeta)...")
        pretrain_actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.pretrain_lr)
        global_train_step = 0

        # Action bounds tensors
        action_low_t  = torch.tensor(envs.single_action_space.low,  dtype=torch.float32, device=device).unsqueeze(0)
        action_high_t = torch.tensor(envs.single_action_space.high, dtype=torch.float32, device=device).unsqueeze(0)

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            for batch_states_cpu, batch_actions_cpu in train_loader:
                batch_states = batch_states_cpu.to(device, non_blocking=True)
                batch_actions_D = batch_actions_cpu.to(device, non_blocking=True)

                # CEM improvement around dataset actions (no grad)
                with torch.no_grad():
                    a_star = cem_improve_actions_batch(
                        states=batch_states,
                        a_D=batch_actions_D,
                        qmeta_fn=qmeta_fn,
                        action_low=action_low_t.expand(batch_states.size(0), -1),
                        action_high=action_high_t.expand(batch_states.size(0), -1),
                        lambda_penalty=args.cem_lambda_penalty,
                        N=args.cem_N,
                        K=args.cem_K,
                        T=args.cem_T,
                        sigma0=args.cem_sigma0,
                    )

                # Distill actor to a_star via NLL
                loss = actor_nll_on_actions(actor, batch_states, a_star)
                pretrain_actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                pretrain_actor_optimizer.step()

                epoch_loss += loss.item()
                if (global_train_step % args.log_every_pretrain) == 0:
                    writer.add_scalar("pretrain/step_nll", loss.item(), global_train_step)
                global_train_step += 1

            epoch_loss /= max(1, len(train_loader))
            writer.add_scalar("pretrain/epoch_nll", epoch_loss, epoch)
            print(f"[pretrain] epoch {epoch:02d} | NLL {epoch_loss:.4f}")

        # reset env for quick check
        obs, _ = envs.reset(seed=args.seed)
        print("==> Pretrain done. Quick eval before saving...")

    # Save actor
    model_dir = f"runs/checkpoints/pretrain_actor/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        "pretrain_actor_state_dict": actor.state_dict(),
    }, os.path.join(model_dir, f"latest.pth"))
    print(f"Saved pretrained actor to {os.path.join(model_dir, 'latest.pth')}")

    # ======= Quick EVAL BEFORE closing envs =======
    obse, _ = envs.reset(seed=args.seed)
    for n in range(25):
        ep_return = 0.0
        terminations = False
        truncations = False
        while not (terminations or truncations):
            with torch.no_grad():
                actions_np, _, _ = actor.get_action(torch.tensor(obse, dtype=torch.float32, device=device))
                actions_np = actions_np.detach().cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = envs.step(actions_np)
            ep_return += float(rewards[0])
            obse = next_obs
        print(f"eval episode {n} return {ep_return:.3f}")
        writer.add_scalar("pretrain/pretrained_ep_return", ep_return, n)

    envs.close()
    writer.close()

    # ========= Stage 2: PURE SAC (kept commented) =========
    # (unchanged big block omitted)
