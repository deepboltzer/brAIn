import torch

from base_attack import Base_Attack


class StrategicallyTimedAttack(Base_Attack):
    """Class for strategically timed adversarial attack."""

    def __init__(self, env, model, attack):
        super().__init__(env, model, attack)

    def perform_attack(self):
        """Performs an adversarial attack based on self.attack on one episode."""
        pass
