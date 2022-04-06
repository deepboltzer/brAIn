import torch
import numpy as np
from abc import ABC, abstractmethod
from utils.data import AttackData


class BaseAttack(ABC):
    """
    Base class for adversarial attacks.
    :param env: environment
    :param model: target model
    :param attack: e.g. FGSM_Attack
    :param epsilon: scales the perturbation, e.g. 0.25
    """

    def __init__(self, env, model, attack, epsilon, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.env = env
        self.model = model
        self.attack = attack

        self.epsilon = epsilon
        self.data = AttackData()

        self.reset_env()

        self.reward_total = 0 # total cumulative episode reward
        self.frames_count = 0 # current number of frames/timesteps in current episode
        self.perturbation_total = 0 # total cumulative perturbation on observations
        self.n_attacks = 0 # number of attacks performed
        
    def reset_env(self):
        """Resets the environment and collects the observation."""
        obs = self.env.reset()
        self.data.last_obs = obs

    def reset_attack(self):
        """Resets the episode and therefore all episode dependent variables."""
        self.reward_total, self.perturbation_total, self.n_attacks = 0, 0, 0

    def update_data(self, obs, act, rew, done, info):
        """Updates data dictionary accordingly."""
        self.data.last_obs = obs
        self.data.last_act = act
        self.data.last_rew = rew
        self.data.last_done = done
        self.data.last_info = info

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

        self.frames_count += 1

    def predict(self, obs):
        """Chooses action based on model."""
        act, _states = self.model.predict(obs)
        self.data.last_act = act
        return act, _states

    def craft_sample(self, orig_obs, orig_act):
        """Craft adversarial sample using self.attack."""
        orig_adv_sample, _states = self.attack.predict(orig_act)

        # scale with epsilon
        orig_perturbation = orig_obs - orig_adv_sample
        scaled_perturbation = orig_perturbation * self.epsilon
        adv_sample = orig_obs - scaled_perturbation

        perturbation = np.abs(orig_obs - adv_sample)
        return adv_sample, perturbation

    @abstractmethod
    def perform_attack(self):
        """Performs an adversarial attack  on one episode."""
        raise NotImplementedError()
