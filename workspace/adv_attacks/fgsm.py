import numpy as np
import torch
import torch
from torch.nn import functional as F


class FGSMAttack():
    """
    FGSM attack as outlined in Goodfellow, et al. 
    Attacks using a scaled sign of the gradient of the model's loss function.
    """

    def __init__(self, model, epsilon=0.25):
        self.model = model
        self.epsilon = epsilon

    def perturb(self, obs, target):
        """Returns an adversarial sample using FGSM."""
        action, _states = self.model.predict(obs) 
        y = torch.tensor(np.array([float(action)])) # model's output
        target = torch.tensor(np.array([float(target)])) # cross entropy target
        
        loss = F.cross_entropy(y, target) # calculate loss cross_entropy(y_pred, y_true)
        grad = loss.backward() # cross entropy loss backward calculates the gradient of the current tensor
        signed_grad = np.sign(grad)
        eta = obs + self.epsilon * signed_grad # calculate eta according to FGSM by addign scaled signed gradient of loss

        return eta