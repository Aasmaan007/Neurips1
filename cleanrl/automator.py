import subprocess
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

# Parameters
seeds = [81, 14, 3, 94, 35]
pretrained_options = [False]
max_parallel_jobs = 10  # Set this based on your system

# Build combinations
combinations = list(itertools.product(seeds, pretrained_options))

def run_combination(seed, pretrained):
    cmd = [
        "python", "-m", "cleanrl.cleanrl.dqn",
        f"--seed={seed}"
    ]

    # Add boolean flag correctly
    if pretrained:
        cmd.append("--pretrained")
    else:
        cmd.append("--no-pretrained")

    print(f"Starting: seed={seed}, pretrained={pretrained}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Finished: seed={seed}, pretrained={pretrained}")
    return (seed, pretrained, result.returncode, result.stdout, result.stderr)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = [executor.submit(run_combination, seed, pretrained) for seed, pretrained in combinations]
        
        for future in as_completed(futures):
            seed, pretrained, returncode, stdout, stderr = future.result()
            print(f"\n=== Result: seed={seed}, pretrained={pretrained} ===")
            print(f"Exit Code: {returncode}")
            if returncode != 0:
                print(f"Error:\n{stderr}")
            else:
                print(f"Output:\n{stdout[:300]}...")
