from abc import abstractmethod
import torch


class Base_Attack():
    """
    Base class for adversarial attacks.
    :param env: environment
    :param attack: e.g. FGSM_Attack
    """

    def __init__(self, env, model, attack, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.env = env
        self.model = model
        self.attack = attack
        self.data = {'obs': {}, 'action': {}, 'rew': {}, 'done': {}, 'info': {}}

        self.reset_env()

        self.episode_count = 0 # current number of episodes
        self.reward_total = 0 # total cumulative episode reward
        self.frames_count = 0 # current number of frames/timesteps in current episode
        self.n_attacks = 0 # number of attacks performed
        
    def reset_env(self):
        """Resets the environment and collects the observation."""
        self.data['obs'] = self.env.reset()

    def reset_attack(self):
        """Resets the episode and therefore all episode dependent variables."""#
        self.episode_count, self.reward_total, self.n_attacks = 0, 0, 0

    def update_data(self, obs, action, rew, done, info):
        """Updates data dictionary accordingly."""
        self.data['obs'][self.frames_count] = obs
        self.data['action'][self.frames_count] = action
        self.data['rew'][self.frames_count] = rew
        self.data['done'][self.frames_count] = done
        self.data['info'][self.frames_count] = info

    def perform_step(self, action):
        """
        Perform an action chosen by the model and store next observation.
        Also checks if the episode finished and resets the environment.
        """
        obs, rew, done, info = self.env.step(action)
        self.update_data(obs, action, rew, done, info)
        self.reward_total += rew

        if self.data['done'][self.frames_count]:
            self.episode_count += 1
            self.reset_env()

    def predict(self):
        """Chooses action based on model."""
        action, _ = self.model.predict(self.data['obs'][self.frames_count])
        self.data['action'][self.frames_count] = action
        return action

    @abstractmethod
    def attack(self):
        """Performs an adversarial attack based on self.attack."""
        pass
