# wrappers.py
import gym
import numpy as np

class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenObsWrapper, self).__init__(env)
        flat_size = np.prod(env.observation_space.shape)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(flat_size,), dtype=np.uint8)

    def observation(self, obs):
        return obs.flatten()
