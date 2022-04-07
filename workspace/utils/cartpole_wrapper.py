import gym
import numpy as np


class CartPoleWrapper(gym.Wrapper):
    """
    Simple wrapper that changes to the reward to be relative to the pole angle.
    :param env: instance of CartPole-v1 environment
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        """Step environment with specified action but change the reward depending on pole angle theta."""
        next_state, orig_rew, done, info = self.env.step(action)
        x, x_dot, theta, theta_dot = self.env.state
        rew = (np.abs(self.env.theta_threshold_radians) - np.abs(theta)) / np.abs(self.env.theta_threshold_radians)

        return next_state, rew, done, info