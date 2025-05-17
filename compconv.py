import os
import pandas as pd
import re
import numpy as np
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    sf_dir: str = "acrobot_runsdata_sf"
    q_dir: str = "acrobot_runsdata_q"
    reward_threshold: float = -100
    patience: int = 9

# === SF EXTRACT ===
def extract_sf_seed_pretrained(filename):
    match = re.search(r"v1__([0-9]+)__wrandom-[^_]+__pretrained-(True|False)", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1)), match.group(2) == "True"

# === Q EXTRACT ===
def extract_q_seed_pretrained(filename):
    match = re.search(r"_(\d+)_.*__pretrained-(True|False)", filename)
    return int(match.group(1)), match.group(2) == "True"

# === Common: timestep extractor ===
def find_convergence_timestep(path, threshold, patience):
    df = pd.read_csv(path)
    rewards, steps = df["Value"], df["Step"]
    for i in range(len(rewards) - patience + 1):
        if all(rewards[i:i+patience] >= threshold):
            return steps[i]
    return 500000  # fallback

# === SF Loader ===
def load_sf_convergence(args: Args):
    pretrained = {}
    for f in os.listdir(args.sf_dir):
        if not f.endswith(".csv"): continue
        try:
            seed, is_pretrained = extract_sf_seed_pretrained(f)
        except ValueError: continue
        if not is_pretrained: continue
        ts = find_convergence_timestep(os.path.join(args.sf_dir, f), args.reward_threshold, args.patience)
        pretrained[seed] = ts
    return pretrained

# === Q Loader ===
def load_q_convergence(args: Args):
    pretrained, scratch = {}, {}
    for f in os.listdir(args.q_dir):
        if not f.endswith(".csv"): continue
        seed, is_pretrained = extract_q_seed_pretrained(f)
        ts = find_convergence_timestep(os.path.join(args.q_dir, f), args.reward_threshold, args.patience)
        (pretrained if is_pretrained else scratch)[seed] = ts
    return pretrained, scratch

# === MAIN ===
def main(args: Args):
    sf_pretrained = load_sf_convergence(args)
    q_pretrained, q_scratch = load_q_convergence(args)

    # Ensure all seeds align
    seeds = sorted(q_scratch.keys())  # assume scratch is baseline
    sf_pre = [sf_pretrained[s] for s in seeds]
    q_pre = [q_pretrained[s] for s in seeds]
    q_scr = [q_scratch[s] for s in seeds]

    print("Seeds:", seeds)
    print("SF Pretrained Timesteps: ", sf_pre)
    print("Q Pretrained Timesteps:  ", q_pre)
    print("Q Scratch Timesteps:     ", q_scr)

    diff_qpre = [s - p for s, p in zip(q_scr, q_pre)]
    diff_sfpre = [s - p for s, p in zip(q_scr, sf_pre)]

    print("\nDiff: Q Scratch - Q Pretrained:", diff_qpre)
    print("Mean:", np.mean(diff_qpre))

    print("\nDiff: Q Scratch - SF Pretrained:", diff_sfpre)
    print("Mean:", np.mean(diff_sfpre))

    print("Mean timesteps over which  MAML SF converged faster than MAML Qnetwork ", np.mean(diff_sfpre) - np.mean(diff_qpre))


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)