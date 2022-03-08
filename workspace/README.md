# Workspace

## [CartPole](https://gym.openai.com/envs/CartPole-v1/)

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

Source: [cartpole.py](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)

Action Space

```
| Num | Action                 |
|-----|------------------------|
| 0   | Push cart to the left  |
| 1   | Push cart to the right |

 **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
```

Observation Space

```
| Num | Observation           | Min                  | Max                |
|-----|-----------------------|----------------------|--------------------|
| 0   | Cart Position         | -4.8                 | 4.8                |
| 1   | Cart Velocity         | -Inf                 | Inf                |
| 2   | Pole Angle            | ~ -0.418 rad (-24°)  | ~ 0.418 rad (24°)  |
| 3   | Pole Angular Velocity | -Inf                 | Inf                |

**Note:** While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
-  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates if the cart leaves the `(-2.4, 2.4)` range.
-  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
```

## PPO

We ran the Proximal Policy Optimization algorithm on two versions of the CartPole environment.

### CartPole-v1

`ppo_cartpole_v1`

- agent was trained over 15000 timesteps
- the model was saved (`/models/cartpole-v1`) and evaluated every 500 timesteps
- evaluation was done over 100 evaluation episodes and can be found in the `/out/cartpole-v1/cartpole_v1_15000.csv`
- check `/out/` for visualization

### CartPole-v1k

A custom version of the CartPole environment was created. This version has an increased maximum episode timesteps of 1'000.

`test_ppo_cartpole_v1k`

- PPO agent was trained over 15000 timesteps
- the model was saved (`/models/cartpole-v1k`) and evaluated every 500 timesteps
- evaluation was done over 100 evaluation episodes and can be found in the `/out/cartpole-v1k/cartpole_v1k_15000.csv`
- agent reaches maximum rewards in a comparable amount of timesteps as with `CartPole-v1`
- parallel to the increase in mean rewards the standard deviation decreased over the course of the learning period
- check `/out/` for visualization
