import numpy as np
import gym

"""
Uniform attack on a model in the CartPole-v1 environment.
"""


def perturb(env, obs, perturbation=(0.0, 0.0, 0.5, 0.5)):
    """Perturb the observation to hinder the agent.
    :param env: target environment
    :param obs: observation to craft an adversarial sample from
    :param perturbation: amount of perturbation to apply
    """

    # Threshholds
    # 2.4 for x
    # 12 * 2 * math.pi / 360 for theta
    x_threshold = env.x_threshold
    theta_threshold = env.theta_threshold_radians
    
    # Get the cart's x position and pole's angle 
    x = obs[0]
    theta = obs[2]

    # Calculate which variable is closer to termination condition
    x_danger = np.absolute(x / x_threshold)
    theta_danger = np.absolute(theta/ theta_threshold)

    # Actions are encoded as 
    # '0' -> push cart to the left
    # '1' -> push card to the right

    # Calculate the target action to minimize reward
    # The following favors theta over x in case of equal danger
    if x_danger > theta_danger:
        if x < 0:
            # If x is already to the left -> target action: move left
            # Therefore pretend x is more to the right and pole is angled more to the left
            x_perturbation = 1 * perturbation[0]
            x_dot_perturbation = 1 * perturbation[1]
            theta_perturbation = -1 * perturbation[2]
            theta_dot_perturbation = -1 * perturbation[3]
        else:
            x_perturbation = -1 * perturbation[0]
            x_dot_perturbation = -1 * perturbation[1]
            theta_perturbation = 1 * perturbation[2]
            theta_dot_perturbation = 1 * perturbation[3]
            
    else:
        if theta < 0:
            # Pole is angled to the left -> target action: move right
            # Therefore pretend x is more to the left and pole is angled more to the right
            x_perturbation = -1 * perturbation[0]
            x_dot_perturbation = -1 * perturbation[1]
            theta_perturbation = 1 * perturbation[2]
            theta_dot_perturbation = 1 * perturbation[3]
        else:
            x_perturbation = 1 * perturbation[0]
            x_dot_perturbation = 1 * perturbation[1]
            theta_perturbation = -1 * perturbation[2]
            theta_dot_perturbation = -1 * perturbation[3]

    # Generate adversarial sample to trick the agent to select target action
    x = obs[0] + x_perturbation
    x_dot = obs[1] + x_dot_perturbation
    theta = obs[2] + theta_perturbation
    theta_dot = obs[3] + theta_dot_perturbation
    
    # CartPole-v1's state: (x, x_dot, theta, theta_dot)
    state = (x, x_dot, theta, theta_dot)
    
    # Construct return value according to CartPole-v1 syntax
    adversarial_sample = np.array(state, dtype=np.float32)

    total_perturbation = np.absolute(obs - adversarial_sample)

    return adversarial_sample, total_perturbation, 1