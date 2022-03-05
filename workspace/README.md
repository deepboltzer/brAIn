# Workspace

## CartPole-v1

1. `ppo_cartpole_v1_100K`

   - trained over 100'000 timesteps
   - and then tested on 100 evaluation episodes
   - mean reward is shown in the output:
     ```
     mean_reward trained agent: 500.00 +/- 0.00
     ```

2. `ppo_cartpole_v1_steps`
   - agent was trained over 25'000 timesteps
   - the model was saved (`/models/cartpole-v1`) and evaluated every 5'000 timesteps
   - evaluation was done over 100 evaluation episodes and can be found in the output:
     ```
     mean_reward (agent trained over 5'000 timesteps): 290.61 +/- 101.96
     mean_reward (agent trained over 10'000 timesteps): 490.51 +/- 30.90
     mean_reward (agent trained over 15'000 timesteps): 500.00 +/- 0.00
     mean_reward (agent trained over 20'000 timesteps): 500.00 +/- 0.00
     mean_reward (agent trained over 25'000 timesteps): 500.00 +/- 0.00
     ```
   - parallel to the increase in mean rewards the standard deviation decreased over the course of the learning period

## CartPole-v1k

A custom version of the CartPole environment was created. This version has an increased maximum episode timesteps of 1'000.

1. `ppo_cartpole_v1k_steps`
   - PPO agent was trained over 20'000 timesteps
   - the model was saved (`/models/cartpole-v1k`) and evaluated every 1'000 timesteps
   - evaluation was done over 100 evaluation episodes and can be found in the output:
     ```
      mean_reward (agent trained over 1'000 timesteps): 330.41 +/- 270.76
      mean_reward (agent trained over 2'000 timesteps): 399.02 +/- 264.97
      mean_reward (agent trained over 3'000 timesteps): 349.32 +/- 229.78
      mean_reward (agent trained over 4'000 timesteps): 413.11 +/- 246.12
      mean_reward (agent trained over 5'000 timesteps): 421.51 +/- 228.71
      mean_reward (agent trained over 6'000 timesteps): 454.90 +/- 217.08
      mean_reward (agent trained over 7'000 timesteps): 475.73 +/- 256.68
      mean_reward (agent trained over 8'000 timesteps): 601.47 +/- 253.89
      mean_reward (agent trained over 9'000 timesteps): 764.44 +/- 250.46
      mean_reward (agent trained over 10'000 timesteps): 599.67 +/- 219.13
      mean_reward (agent trained over 11'000 timesteps): 823.81 +/- 212.27
      mean_reward (agent trained over 12'000 timesteps): 896.98 +/- 150.79
      mean_reward (agent trained over 13'000 timesteps): 902.53 +/- 154.34
      mean_reward (agent trained over 14'000 timesteps): 951.84 +/- 96.14
      mean_reward (agent trained over 15'000 timesteps): 942.05 +/- 108.92
      mean_reward (agent trained over 16'000 timesteps): 983.54 +/- 53.59
      mean_reward (agent trained over 17'000 timesteps): 1000.00 +/- 0.00
      mean_reward (agent trained over 18'000 timesteps): 1000.00 +/- 0.00
      mean_reward (agent trained over 19'000 timesteps): 1000.00 +/- 0.00
      mean_reward (agent trained over 20'000 timesteps): 1000.00 +/- 0.00
     ```
   - agent reaches maximum rewards in a comparable amount of timesteps as with `CartPole-v1`
   - parallel to the increase in mean rewards the standard deviation decreased over the course of the learning period
