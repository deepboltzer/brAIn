import numpy as np
from audioop import maxpp
from base_attack import BaseAttack


class StrategicallyTimedAttack(BaseAttack):
    """Class for strategically timed adversarial attack."""

    def __init__(self, env, model, attack, epsilon, beta):
        super().__init__(env, model, attack, epsilon)
        self.beta = beta

    def perform_attack(self, render=False):
        """Performs an adversarial attack based on self.attack on one episode."""
        self.reset_attack()
        self.reset_env()

        while not self.data.last_done:

            orig_obs = self.data.last_obs
            if self.c(orig_obs) >= self.beta:
                adv_sample = self.craft_sample(orig_obs)
                perturbed_act, _states = self.predict(adv_sample)
                self.perform_step(perturbed_act)
            else:
                orig_act, _states = self.predict(orig_obs)
                self.perform_step(orig_act)
            if render:
                self.render()

    def c(self, state):
        """
        Action preference function. 
        The higher the value the more this action is preferred over the other.
        Therefore the higher the value the more efficient is an attack in this timestep.
        :param orig_act: original action on unperturbed observation
        :param perturbed_act: action chosen on adversarial sample
        """
        act_prob = self.predict_action_probabilities(state)

        max_act = np.max(act_prob)
        min_act = np.min(act_prob)

        return max_act - min_act
