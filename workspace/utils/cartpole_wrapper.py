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

    def test_action(self, action):
        """Test action without affecting state. Naturally this function is based on CartPole.step"""
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.env.action_space.contains(action), err_msg
        assert self.env.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.env.state
        force = self.env.force_mag if action == 1 else -self.env.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.env.polemass_length * theta_dot**2 * sintheta
        ) / self.env.total_mass
        thetaacc = (self.env.gravity * sintheta - costheta * temp) / (
            self.env.length * (4.0 / 3.0 - self.env.masspole * costheta**2 / self.env.total_mass)
        )
        xacc = temp - self.env.polemass_length * thetaacc * costheta / self.env.total_mass

        if self.env.kinematics_integrator == "euler":
            x = x + self.env.tau * x_dot
            x_dot = x_dot + self.env.tau * xacc
            theta = theta + self.env.tau * theta_dot
            theta_dot = theta_dot + self.env.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.env.tau * xacc
            x = x + self.env.tau * x_dot
            theta_dot = theta_dot + self.env.tau * thetaacc
            theta = theta + self.env.tau * theta_dot

        # don't set env's state just save it in a variable
        state = (x, x_dot, theta, theta_dot)

        # looks at state variable instead of env.state
        done = bool(
            state[0] < -self.env.x_threshold
            or state[0] > self.env.x_threshold
            or state[2] < -self.env.theta_threshold_radians
            or state[2] > self.env.theta_threshold_radians
        )

        # obviously also uses the new way of calculating the reward
        if not done:
            reward = self.get_rew(state)
        elif self.env.steps_beyond_done is None:
            # Pole just fell!
            self.env.steps_beyond_done = 0
            reward = self.get_rew(state)
        else:
            # no changes during testing
            # if self.env.steps_beyond_done == 0:
            #     logger.warn(
            #         "You are calling 'step()' even though this "
            #         "environment has already returned done = True. You "
            #         "should always call 'reset()' once you receive 'done = "
            #         "True' -- any further steps are undefined behavior."
            #     )
            # self.env.steps_beyond_done += 1
            reward = 0.0

        return np.array(state, dtype=np.float32), reward, done, {}