from abc import ABC, abstractmethod
import torch
from utils.data import AttackData


class BaseAttack(ABC):
    """
    Base class for adversarial attacks.
    :param env: environment
    :param attack: e.g. FGSM_Attack
    """

    def __init__(self, env, model, attack, epsilon=0.25, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.env = env
        self.model = model
        self.attack = attack

        self.epsilon = epsilon
        self.data = AttackData()

        self.reset_env()

        self.reward_total = 0 # total cumulative episode reward
        self.frames_count = 0 # current number of frames/timesteps in current episode
        self.n_attacks = 0 # number of attacks performed
        
    def reset_env(self):
        """Resets the environment and collects the observation."""
        self.data.set_last_obs(self.env.reset())

    def reset_attack(self):
        """Resets the episode and therefore all episode dependent variables."""
        self.reward_total, self.n_attacks = 0, 0

    def update_data(self, obs, act, rew, done, info):
        """Updates data dictionary accordingly."""
        self.data.set_last_obs(obs)
        self.data.set_last_act(act)
        self.data.set_last_rew(rew)
        self.data.set_last_done(done)
        self.data.set_last_info(info)

    def perform_step(self, act):
        """
        Perform an action chosen by the model and store next observation.
        Also checks if the episode finished and resets the environment.
        """
        obs, rew, done, info = self.env.step(act)
        self.update_data(obs, act, rew, done, info)
        self.reward_total += rew

        if self.data.last_done:
            self.reset_env()

    def predict(self, obs):
        """Chooses action based on model."""
        act, _states = self.model.predict(obs)
        self.data.set_last_act(act)
        return act

    def craft_sample(self, orig_act):
        """Craft adversarial sample using self.attack."""
        return self.attack.predict(orig_act, deterministic=True)

    @abstractmethod
    def perform_attack(self):
        """Performs an adversarial attack  on one episode."""
        raise NotImplementedError()
