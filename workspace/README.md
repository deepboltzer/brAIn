# Workspace

## CartPole-v1

1. `ppo_cartpole_v1_100K`

   - trained over 100000 timesteps
   - and then tested on 100 evaluation episodes
   - mean reward:
     ```
     mean_reward trained agent: 500.00 +/- 0.00
     ```

2. `ppo_cartpole_v1_steps`
   - agent was trained over 15000 timesteps
   - the model was saved (`/models/cartpole-v1`) and evaluated every 500 timesteps
   - evaluation was done over 100 evaluation episodes and can be found in the `/out/cartpole-v1/cartpole_v1_15000.csv`
   - check `visualize` for plots

## CartPole-v1k

A custom version of the CartPole environment was created. This version has an increased maximum episode timesteps of 1'000.

1. `ppo_cartpole_v1k_steps`
   - PPO agent was trained over 15000 timesteps
   - the model was saved (`/models/cartpole-v1k`) and evaluated every 500 timesteps
   - evaluation was done over 100 evaluation episodes and can be found in the `/out/cartpole-v1k/cartpole_v1k_15000.csv`
   - agent reaches maximum rewards in a comparable amount of timesteps as with `CartPole-v1`
   - parallel to the increase in mean rewards the standard deviation decreased over the course of the learning period
   - check `visualize` for plots
