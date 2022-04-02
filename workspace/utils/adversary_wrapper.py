import gym


class AdversaryWrapper(gym.Wrapper):
    """
    Simple wrapper to train an adversary on a given model.
    :param env: target environment
    :param model: target model
    """
    def _init__(self, env):
        super().__init__(env)
        self.env = env

        self.action_space = self.env.observation_space
        self.observation_space = self.env.action_space

        self.last_state, self.last_rew, self.last_done, self.last_info = self.env.reset()
    
    def step(self, act):
        """Step in environment based on models prediction of adversary's action."""
        print(act)
        model_act, _states = self.model.predict(act)
        next_state, rew, done, info = self.env.step(model_act)

        self.last_action = act
        self.last_state = next_state
        self.last_rew = rew
        self.last_done = done
        self.last_info = info

        adv_obs = None
        adv_rew = -rew

        return adv_obs, adv_rew, done, info

    def set_model(self, model):
        self.model = model
