import numpy as np
import matplotlib.pyplot as plt

def get_per_step_loss_weights(num_steps, current_epoch, decay_epoch, use_fixed=True):
    weights = np.ones(num_steps) * (1.0 / num_steps)

    if use_fixed:
        return weights

    decay_rate = 1.0 / num_steps / decay_epoch
    min_non_final = 0.03 / num_steps

    for i in range(num_steps - 1):
        weights[i] = max(weights[i] - (current_epoch * decay_rate), min_non_final)

    weights[-1] = min(1.0 - ((num_steps - 1) * min_non_final),
                      weights[-1] + (current_epoch * (num_steps - 1) * decay_rate))
    
    return weights


def visualize_weight_progression(num_steps, max_epochs, decay_epochs  , use_fixed=False):
    step_weights_over_epochs = []

    for epoch in range(max_epochs + 1):
        weights = get_per_step_loss_weights(num_steps, epoch, decay_epochs, use_fixed)
        step_weights_over_epochs.append(weights)

    step_weights_over_epochs = np.array(step_weights_over_epochs)  # shape: (epochs, num_steps)

    plt.figure(figsize=(8, 5))
    for i in range(num_steps):
        plt.plot(range(max_epochs + 1), step_weights_over_epochs[:, i], label=f"Step {i+1}")

    plt.title(f"Weight Progression per Step (num_steps={num_steps}, fixed={use_fixed})")
    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Visualize for 2 and 3 steps
# visualize_weight_progression(num_steps=2, max_epochs=1000, use_fixed=False)
visualize_weight_progression(num_steps=3, max_epochs=10000, decay_epochs = 2000, use_fixed=False)
