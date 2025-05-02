import torch

# Load your checkpoint
checkpoint_path = "runs/checkpoints/qtargetmaml/LunarLander-v2__q_online__1__2025-05-01_16-14-07__1746096247/latest.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Extract weight shape
q_state_dict = checkpoint["q_network_state_dict"]
input_dim = q_state_dict["network.0.weight"].shape[1]
output_dim = q_state_dict["network.0.weight"].shape[0]

print(f"QNetwork first layer weight shape: [{output_dim}, {input_dim}]")
print(f"=> Detected input_dim: {input_dim} (should be state_dim + skill_dim)")