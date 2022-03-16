import gym
import EnvWrapper
import time

from stable_baselines3 import PPO, A2C

def make_opponent(model):
    return lambda obs : model.predict(obs, deterministic=True)

# Buld Models and environment setup
EnvWrapper.static_policy = 'array'
# PongDuel-v0 Checkers-v0
EnvWrapper.multiagent_gym = "ma_gym:PongDuel-v0"
env = EnvWrapper.SingleMultiAgentEnv()

models = []
for i in range(EnvWrapper.number_of_agents):
    env.set_active_agent_n(i)
    models.append(A2C("MlpPolicy", env, verbose=1))
    env.set_opponent_n(make_opponent(models[i]), i)
env.set_active_agent_n(0)


# Train the models
#env.set_opponent(models[1].predict)
for i in range(20):
    print(i)
    to_train = i%2
    to_next = (i+1)%2
    models[to_train].learn(total_timesteps=5000)
    env.set_opponent_n(make_opponent(models[to_train]), to_train)
    env.set_active_agent_n(to_next)

"""
# Train the models
#env.set_opponent(models[1].predict)
models[0].learn(total_timesteps=50000)
env.set_opponent_n(make_opponent(models[0]), 0)
env.set_active_agent_n(1)
models[1].learn(total_timesteps=50000)
env.set_opponent_n(make_opponent(models[1]), 0)
env.set_active_agent_n(0)
"""

# Visualize results
obs = env.reset()
for i in range(1000):
    action, _states = models[0].predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
    time.sleep(0.05)

env.close()
