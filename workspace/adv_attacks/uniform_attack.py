import torch

from base_attack import BaseAttack


class UniformAttack(BaseAttack):
    """Class for uniform adversarial attack."""

    def __init__(self, env, model, attack, epsilon):
        super().__init__(env, model, attack, epsilon)

    def perform_attack(self):
        """Performs an adversarial attack on one episode."""
        self.reset_attack()
        self.reset_env()

        while not self.data.last_done:
            orig_obs = self.data.last_obs
            orig_act, _states = self.predict(orig_obs)
            adv_sample, perturbation = self.craft_sample(orig_obs, orig_act)
            perturbed_act, _states = self.predict(adv_sample)
            self.perform_step(perturbed_act)
            self.perturbation_total += perturbation
            self.n_attacks += 1
