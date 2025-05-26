import os
import pandas as pd
import re
import numpy as np
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    data_dir: str = "acrobot_runsdata_sf"
    reward_threshold: float = -100
    patience: int = 10

def extract_seed_pretrained(filename):
    # Matches: LunarLander-v2__35__wrandom-False__pretrained-True__2025-05-05_02-37-10.csv
    match = re.search(r"v1__([0-9]+)__wrandom-[^_]+__pretrained-(True|False)", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1)), match.group(2) == "True"

def find_convergence_timestep(path, threshold, patience):
    df = pd.read_csv(path)
    rewards, steps = df["Value"], df["Step"]
    for i in range(len(rewards) - patience + 1):
        if all(rewards[i:i+patience] >= threshold):
            return steps[i]
    return 500000

def main(args: Args):
    pretrained = {}
    scratch = {}

    for f in os.listdir(args.data_dir):
        if not f.endswith(".csv"):
            continue
        try:
            seed, is_pretrained = extract_seed_pretrained(f)
        except ValueError as e:
            print(e)
            continue
        ts = find_convergence_timestep(os.path.join(args.data_dir, f), args.reward_threshold, args.patience)
        (pretrained if is_pretrained else scratch)[seed] = ts

    seeds = sorted(pretrained.keys())
    p_vals = [pretrained[s] for s in seeds]
    s_vals = [scratch[s] for s in seeds]
    diffs = [s - p for s, p in zip(s_vals, p_vals)]

    print("Seeds:", seeds)
    print("Pretrained timesteps:", p_vals)
    print("Scratch timesteps:", s_vals)
    print("Diffs (scratch - pretrained):", diffs)
    print("Mean diff:", np.mean(diffs))

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)