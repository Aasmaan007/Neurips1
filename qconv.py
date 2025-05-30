import os
import pandas as pd
import re
import numpy as np
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    data_dir: str = "runsdataq"
    reward_threshold: float = 200.0
    patience: int = 9

def extract_seed_pretrained(filename):
    match = re.search(r"_(\d+)_.*__pretrained-(True|False)", filename)
    return int(match.group(1)), match.group(2) == "True"

def find_convergence_timestep(path, threshold, patience):
    df = pd.read_csv(path)
    rewards, steps = df["Value"], df["Step"]
    for i in range(len(rewards) - patience + 1):
        if all(rewards[i:i+patience] >= threshold):
            return steps[i]
    return 500000

def main(args: Args):
    pretrained, scratch = {}, {}

    for f in os.listdir(args.data_dir):
        if not f.endswith(".csv"): continue
        seed, is_pretrained = extract_seed_pretrained(f)
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