import os
import pandas as pd
import re
import numpy as np
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    sf_dir: str = "runsdatasf"
    q_dir: str = "runsdataq"
    env_eplen: int = 1000
    

# =========================
#  Extractors
# =========================
# === SF EXTRACT ===
def extract_sf_seed_pretrained(filename):
    match = re.search(r"v2__([0-9]+)__wrandom-[^_]+__pretrained-(True|False)", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1)), match.group(2) == "True"

# === Q EXTRACT ===
def extract_q_seed_pretrained(filename):
    match = re.search(r"_(\d+)_.*__pretrained-(True|False)", filename)
    return int(match.group(1)), match.group(2) == "True"

# =========================
#  CSV Loading + Caching
# =========================

def preload_all_csvs_in_dir(directory):
    """
    Loads all CSVs in a directory into memory ONCE.
    Returns dict: filename -> DataFrame
    """
    data = {}
    for fname in os.listdir(directory):
        if fname.endswith(".csv"):
            full_path = os.path.join(directory, fname)
            data[fname] = pd.read_csv(full_path)
    return data

# =========================
#  Convergence Finder
# =========================

def find_convergence_timestep_df(df, threshold, patience):
    rewards, steps = df["Value"], df["Step"]
    for i in range(len(rewards) - patience + 1):
        if all(rewards[i:i+patience] >= threshold):
            return steps[i]
    return 500000  # fallback

# =========================
#  Precomputation
# =========================

def precompute_all_convergence(all_csv_data, thresholds, patiences):
    """
    For every file and every (threshold, patience) pair,
    compute convergence ONCE and store in a dict.
    """
    results = {}
    for fname, df in all_csv_data.items():
        for threshold in thresholds:
            for patience in patiences:
                ts = find_convergence_timestep_df(df, threshold, patience)
                # if(fname == 'Hopper-v4__td3_continuous_action__35__False__1756819382.csv' and threshold == 1500 and patience == 1):
                #     print(f"Found q at {ts}")
                # if(fname == 'Hopper-v4__td3test__35__True__1756917949.csv' and threshold == 1500 and patience == 1):
                #     print(f"Found sf at {ts}")
                results[(fname, threshold, patience)] = ts
    return results

# =========================
#  Cached Loaders
# =========================

def load_sf_convergence_from_precomputed(precomputed_results, threshold, patience):
    pretrained = {}
    for fname in set(f for f, t, p in precomputed_results.keys()):
        try:
            seed, is_pretrained = extract_sf_seed_pretrained(fname)
        except ValueError:
            continue
        if not is_pretrained:
            continue
        ts = precomputed_results[(fname, threshold, patience)]
        pretrained[seed] = ts
    return pretrained

def load_q_convergence_from_precomputed(precomputed_results, threshold, patience):
    pretrained, scratch = {}, {}
    for fname in set(f for f, t, p in precomputed_results.keys()):
        try:
            seed, is_pretrained = extract_q_seed_pretrained(fname)
        except ValueError:
            continue
        ts = precomputed_results[(fname, threshold, patience)]
        (pretrained if is_pretrained else scratch)[seed] = ts
    return pretrained, scratch

# =========================
#  Main
# =========================

def main(args: Args):

    ALL_THRESHOLDS = range(190, 201)
    ALL_PATIENCES = range(5, 36,5)
    max_diff_threshold = 15000

    print("=== Preloading CSVs ===")
    sf_data = preload_all_csvs_in_dir(args.sf_dir)
    q_data  = preload_all_csvs_in_dir(args.q_dir)

    print("=== Precomputing convergence ===")
    sf_precomputed_10 = precompute_all_convergence(sf_data, ALL_THRESHOLDS, ALL_PATIENCES)
    q_precomputed_10  = precompute_all_convergence(q_data,  ALL_THRESHOLDS, ALL_PATIENCES)
  


    print("=== Starting grid search ===")

    best_result = 0

    for patience in ALL_PATIENCES:
        for threshold in ALL_THRESHOLDS:
            sf_pretrained_10 = load_sf_convergence_from_precomputed(sf_precomputed_10, threshold, patience)
            q_pretrained_10, q_scratch_10 = load_q_convergence_from_precomputed(q_precomputed_10, threshold, patience)
            seeds = sorted(q_scratch_10.keys())
            if not seeds:
                continue

            try:
                sf_pre_vals_10 = [sf_pretrained_10[s] for s in seeds]
            except KeyError:
                continue
            q_scratch_vals_10 = [q_scratch_10[s] for s in seeds]
            #print(q_scratch_vals_10, sf_pre_vals_10 )
            diff10 = [qs - sf for qs, sf in zip(q_scratch_vals_10, sf_pre_vals_10)]
            #print(diff10)
            positive_count = sum(1 for x in diff10 if x > 0)
            if(positive_count > 4 and np.mean(diff10)> 0):
                print(f"Found patience of {patience} and threshold of {threshold} with mean diff {np.mean(diff10)} as approx {np.mean(diff10)/args.env_eplen} episodes")
                best_result = np.mean(diff10) 

    print("\n=== DONE ===")



if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)