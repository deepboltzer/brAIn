import torch
import copy

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
            adv_sample = self.craft_sample(orig_obs, orig_act)
            perturbed_act, _states = self.predict(adv_sample)
            if self.c(orig_act, perturbed_act) >= self.beta:
                self.perform_step(perturbed_act)
            else:
                self.perform_step(orig_act)
    
    def c(self, orig_act, perturbed_act):
        """
        Action preference function. 
        The higher the value the more this action is preferred over the other.
        Therefore the higher the value the more efficient is an attack in this timestep.
        :param orig_act: original action on unperturbed observation
        :param perturbed_act: action chosen on adversarial sample
        """
        env_copy1 = copy.deepcopy(self.env)
        env_copy2 = copy.deepcopy(self.env)
        _state, max_act, _done, _info = env_copy1.step(orig_act)
        _state, min_act, _done, _info = env_copy2.step(perturbed_act)

        return max_act - min_act
