import random

def generate_seeds(global_seed: int, n: int = 5):
    rng = random.Random(global_seed)
    return [rng.randint(0, 100) for _ in range(n)]

# Example usage
global_seed = 42
seeds = generate_seeds(global_seed)
print(seeds)