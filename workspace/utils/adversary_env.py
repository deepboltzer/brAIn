import gym

class AdversaryEnv(gym.Env):
    """
    Simple environment to train an adversary on a given model.
    :param env: target environment
    :param targ_model: target model
    """

    def __init__(self, env, targ_model):
        self.env = gym.make(env)
        self.targ_model = targ_model
        
        self.action_space = self.env.observation_space
        self.observation_space = self.env.action_space

        self.env.reset()

    def reset(self):
        self.env.reset()
        return self.observation_space.sample()

    def step(self, action):
        """Adversary supplies target model with observation and receives its action as new state."""
        targ_model_act, _states = self.targ_model.predict(action)

        # retreive reward as the inverted reward returned by original environment
        obs, rew, done, info = self.env.step(targ_model_act)
        rew *= -1 

        return targ_model_act, rew, done, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def seed(self, n=None):
        return self.env.seed(n)

    def close(self):
        return self.env.close()