import os
import pickle
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm

MAML_DATA_PATH = "runs/data/LunarLander__maml_training_data.pkl"
OUTPUT_PATH = "runs/data/LunarLander__maml_transitions.pkl"
ENV_ID = "LunarLander-v2"
N_ACTIONS = 4

def load_states(path):
    with open(path, "rb") as f:
        states = pickle.load(f)
        np.random.shuffle(states)
    return states

def main():
    print(f"Loading MAML states from {MAML_DATA_PATH}")
    states = load_states(MAML_DATA_PATH)
    env = gym.make(ENV_ID)

    r_list, s_next_list = [], [], [], []

    print("Generating transitions...")
    for s in tqdm(states):
        for a in range(N_ACTIONS):
            env.reset()
            env.unwrapped.state = s
            next_obs, reward, terminated, truncated, info = env.step(a)

            
            r_list.append(reward)
            s_next_list.append(next_obs)

    transitions = {
        "r": np.array(r_list, dtype=np.float32),
        "s_next": np.array(s_next_list, dtype=np.float32),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(transitions, f)

    print(f"Saved {len(r_list)} transitions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()