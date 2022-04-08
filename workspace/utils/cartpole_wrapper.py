import numpy as np
import math
import gym
from gym import logger



class CartPoleWrapper(gym.Wrapper):
    """
    Simple wrapper that changes to the reward to be relative to the pole angle.
    Also supplies a method to try out different actions without affecting the environment's state.
    :param env: instance of CartPole-v1 environment
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        """Step environment with specified action but change the reward depending on pole angle theta."""
        next_state, orig_rew, done, info = self.env.step(action)
        state = self.env.state
        rew = self.get_rew(state)

        return next_state, rew, done, info

    def get_rew(self, state):
        """Calculate reward."""
        scale = 1000
        x, x_dot, theta, theta_dot = state
        rew = (-(theta * theta_dot)) / (self.env.theta_threshold_radians * 3) # results in a value between 0 and 1
        return rew #* scale