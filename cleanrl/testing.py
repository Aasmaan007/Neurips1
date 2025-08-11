import gymnasium as gym

# Create the HalfCheetah environment
env = gym.make('HalfCheetah-v4', render_mode='rgb_array')

# Initialize the environment
state = env.reset()

print(env.action_space.shape[0])

# Close the environment after finishing
env.close()
