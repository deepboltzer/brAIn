import gym
from gym import Env, Wrapper


multiagent_gym = 'PLACEHOLDER'
# static policy should be either 'default', 'array' or a function pointer
static_policy = 'default' # if 'array' you need to set opponents before use
agent_position = 0
number_of_agents = 2


"""
Needs adjustment to work with more than 2 agents
"""
class SingleMultiAgentEnv(Env):

    def __init__(self):
        self.env = gym.make(multiagent_gym)
        if (static_policy == 'default'):
            self.opponent = 'default'
            self.opponents = []
        elif (static_policy == 'array'):
            self.opponent = 'array'
            self.opponents = []
            for i in range(number_of_agents):
                self.opponents.append(0)
        else:
            # need some way to load policy
            self.opponent = static_policy()
            self.opponents = []

                
        self.num_agents = number_of_agents
        self.active_agent_n = agent_position
        self.action_space = self.env.action_space[agent_position]
        self.observation_space = self.env.observation_space[agent_position]

        self.obs = self.env.reset()
        self.act = self.env.action_space.sample()
        self.rew = []
        self.done = []
        self.info = []

    def step(self, action):
        n = self.active_agent_n
        if (self.opponent != 'default'):
            for i in range(self.num_agents):
                if (not i == n): # and (not self.done[i])
                    if (self.opponent != 'array'):
                        self.act[i], _states = self.opponent(self.obs[i])
                    else:
                        self.act[i], _states = self.opponents[i](self.obs[i])
        else:
            self.act = self.env.action_space.sample()
        self.act[n] = action

        self.obs, self.rew, self.done, self.info = self.env.step(self.act)

        return self.obs[n], self.rew[n], self.done[n], self.info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs[self.active_agent_n]

    def render(self, mode="human"):
        return self.env.render(mode)

    def seed(self, n=None):
        return self.env.seed(n)

    def close(self):
        return self.env.close()

    def set_opponent(self, value):
        self.opponent = value

    def set_opponent_n(self, value, n):
        assert(self.opponent == 'array')
        self.opponents[n] = value

    def set_active_agent_n(self, n):
        self.active_agent_n = n
        self.action_space = self.env.action_space[n]
        self.observation_space = self.env.observation_space[n]


    
def make_opponent(model):
    return lambda obs : model.predict(obs, deterministic=True)
        
        
    

"""
env = gym.make('ma_gym:Switch2-v0')

print(env.action_space)
print(env.observation_space)
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
env.close()
"""
