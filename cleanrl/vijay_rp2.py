import gymnasium as gym
import numpy as np
import gym_windy_gridworlds
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import dataclass
import tyro
import wandb
from pprint import pformat
import random
from gymnasium.wrappers import TimeLimit


# =================== ARGS ===================
@dataclass
class Args:
    env_id: str = "WindyGridWorld-v0"
    code_for: str = "double_coupled"
    alpha: float = 0.0001
    beta: float = 0.01
    gamma: float = 0.99
    epsilon_start: float = 1
    epsilon_final: float = 0.05
    exploration_fraction: float = 0.3
    num_episodes: int = 5000
    l: int = 2
    log_base_dir: str = "runs/comp"
    eta: float = 0.5
    seed: int = 42

#full list:  [0.0001, 0.0005, 0.001, 0.005, 0.01]

sweep_config = {
    "method": "grid",
    "parameters": {
        "alpha": {"values": [0.0005]},
        "beta": {"values": [0.0001, 0.0005, 0.001, 0.005, 0.01]},
        "eta": {"values": [0.125, 0.25, 0.5, 1.0]},
        "exploration_fraction": {"values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]},
        "num_episodes": {"values": [500, 1000, 2000, 5000]}
    }
}

# =================== TRAINING LOOP ===================
def run_training(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    env = gym.make(args.env_id, render_mode="human")
    env = TimeLimit(env, 200)
    env.reset(seed=args.seed)   # seed the environment
    env.action_space.seed(args.seed)  # seed action sampling if used

    n_actions = env.action_space.n
    
    obs_dim = env.observation_space.shape[0]
    log_name = f"{args.code_for}_{args.env_id}_rbf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"{args.log_base_dir}/{log_name}")
    writer.add_text("Args", pformat(vars(args)), 0)

    # Create RBF centers (l x l grid in state space)
    pos_min, pos_max = env.observation_space.low[0], env.observation_space.high[0]
    vel_min, vel_max = env.observation_space.low[1], env.observation_space.high[1]

    pos_centers = np.linspace(pos_min + 0.5*((pos_max-pos_min)/args.l), pos_max - 0.5*((pos_max-pos_min)/args.l), args.l)
    vel_centers = np.linspace(vel_min + 0.5*((vel_max-vel_min)/args.l), vel_max - 0.5*((vel_max-vel_min)/args.l), args.l)
    pos_grid, vel_grid = np.meshgrid(pos_centers, vel_centers)
    rbf_centers = np.column_stack([pos_grid.ravel(), vel_grid.ravel()])

    sigma_pos = args.eta * 1.8
    sigma_vel = args.eta * 0.14

    # Feature vector: RBFs over state
    def phi(state):
        state = np.array(state).flatten()
        diffs = rbf_centers - state
        exponents = -((diffs[:, 0] ** 2) / (2 * sigma_pos ** 2) + (diffs[:, 1] ** 2) / (2 * sigma_vel ** 2))
        return np.exp(exponents).astype(np.float32)

    feature_dim = len(phi(env.reset()[0]))
    weights_u = np.zeros((n_actions, feature_dim), dtype=np.float32)
    weights_v = np.zeros((n_actions, feature_dim), dtype=np.float32)
    weights_w1 = np.zeros((n_actions, feature_dim), dtype=np.float32)
    weights_w2 = np.zeros((n_actions, feature_dim), dtype=np.float32)


    # ε-greedy schedule
    def get_epsilon(episode):
        progress = min(episode / (args.exploration_fraction * args.num_episodes), 1.0)
        return args.epsilon_start + (args.epsilon_final - args.epsilon_start) * progress

    # Q-value: dot product between action weight and RBF feature vector
    def q_value_u(state, action):
        return np.dot(weights_u[action], phi(state))

    def q_value_v(state, action):
        return np.dot(weights_v[action], phi(state))
    
    def q_value_w1(state, action):
        return np.dot(weights_w1[action], phi(state))

    def q_value_w2(state, action):
        return np.dot(weights_w2[action], phi(state))

    # Policy: ε-greedy
    def select_action(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)
        q_valsv = [q_value_v(state, a) for a in range(n_actions)]
        q_valsw2 = [q_value_w2(state, a) for a in range(n_actions)]
        q_mean = np.mean([q_valsv, q_valsw2], axis=0)
        return int(np.argmax(q_mean))

    for episode in range(args.num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        epsilon = get_epsilon(episode)

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward


            next_action1 = np.argmax([q_value_w1(next_state, a) for a in range(n_actions)])
            q_next1 = q_value_u(next_state, next_action1)
            td_target1 = reward + args.gamma * q_next1
            td_error1 = td_target1 - q_value_v(state, action)

            weights_u[action] += args.alpha * ((phi(state) * q_value_v(state, action)) - weights_u[action])
            weights_v[action] += args.beta * td_error1 * phi(state)


            next_action2 = np.argmax([q_value_u(next_state, a) for a in range(n_actions)])
            q_next2 = q_value_w1(next_state, next_action2)
            td_target2 = reward + args.gamma * q_next2
            td_error2 = td_target2 - q_value_w2(state, action)

            

            weights_w1[action] += args.alpha * ((phi(state) * q_value_w2(state, action)) - weights_w1[action])
            weights_w2[action] += args.beta * td_error2 * phi(state)  

            weights_u = np.clip(weights_u, -1e6, 1e6)
            weights_v = np.clip(weights_v, -1e6, 1e6)
            weights_w1 = np.clip(weights_w1, -1e6, 1e6)
            weights_w2 = np.clip(weights_w2, -1e6, 1e6)        
            

            state = next_state

        # Log to TensorBoard
        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        writer.add_scalar("weight_1_u", weights_u[0].mean(), episode)
        writer.add_scalar("weight_2_u", weights_u[1].mean(), episode)
        writer.add_scalar("weight_3_u", weights_u[2].mean(), episode)
        writer.add_scalar("weight_1_v", weights_v[0].mean(), episode)
        writer.add_scalar("weight_2_v", weights_v[1].mean(), episode)
        writer.add_scalar("weight_3_v", weights_v[2].mean(), episode)
        writer.add_scalar("weight_1_w1", weights_w1[0].mean(), episode)
        writer.add_scalar("weight_2_w1", weights_w1[1].mean(), episode)
        writer.add_scalar("weight_3_w1", weights_w1[2].mean(), episode)
        writer.add_scalar("weight_1_w2", weights_w2[0].mean(), episode)
        writer.add_scalar("weight_2_w2", weights_w2[1].mean(), episode)
        writer.add_scalar("weight_3_w2", weights_w2[2].mean(), episode)

        # Also log to wandb if active
        if wandb.run is not None:
            wandb.log({"Reward": total_reward, "Epsilon": epsilon})

        print(f"Episode {episode+1}/{args.num_episodes} | Total reward: {total_reward:.1f}")

    env.close()
    writer.close()

# =================== SWEEP ENTRY ===================
def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        args = Args(
            env_id="WindyGridWorld-v0",
            code_for="coupled_lfa",
            alpha=config.alpha,
            beta=config.beta,
            gamma=0.99,
            epsilon_start=1,
            epsilon_final=0.05,
            exploration_fraction=config.exploration_fraction,
            num_episodes=config.num_episodes,
            l=2,
            log_base_dir="runs/lfa",
            eta=config.eta,
            seed=42
        )
        run_training(args)

# =================== MAIN ===================
if __name__ == "__main__":
    import sys

    if "sweep" in sys.argv:
        wandb.login()
        sweep_id = wandb.sweep(sweep_config, project="double_coupled_windygridworldx2")
        wandb.agent(sweep_id, function=train_sweep)
    else:
        args = tyro.cli(Args)
        run_training(args)