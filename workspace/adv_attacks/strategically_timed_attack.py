import torch

from base_attack import Base_Attack


class StrategicallyTimedAttack(Base_Attack):
    """Class for strategically timed adversarial attack."""

    def __init__(self, env, model, attack, beta):
        super().__init__(env, model, attack)
        self.beta = beta

    def perform_attack(self):
        """Performs an adversarial attack based on self.attack on one episode."""
        self.reset_attack()
        self.reset_env()

        while not self.data.last_done:
            
            orig_obs = self.data.last_obs
            orig_act, _states = self.predict(orig_obs)
            if self.c() >= self.beta:
                adv_sample = self.craft_sample(orig_obs, orig_act)
                perturbed_act, _states = self.predict(adv_sample)
                self.perform_step(perturbed_act)
            else:
                self.perform_step(orig_act)
    
    def c(self):
        pass
