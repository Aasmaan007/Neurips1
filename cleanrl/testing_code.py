import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, FlatObsWrapper

# Create the environment
env = gym.make("MiniGrid-Unlock-v0")  # a simple environment

# Optional: Wrap to get image observations instead of symbolic
#env = RGBImgPartialObsWrapper(env)  # Get RGB image observations
            # Get rid of the 'mission' field


env = FullyObsWrapper(env)
env = ImgObsWrapper(env)
#env = FlatObsWrapper(env)


num_episodes = 5

obs = env.reset()
action = env.action_space.sample()
obs,reward,done,info,_ = env.step(action)
print(obs.reshape(-1).shape)



env.close()
