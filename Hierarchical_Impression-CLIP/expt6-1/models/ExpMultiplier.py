import torch
import torch.nn as nn
import numpy as np

class ExpMultiplier(nn.Module):
    def __init__(self, initial_value=0.0):
        super(ExpMultiplier, self).__init__()
        self.t = nn.Parameter(torch.tensor(initial_value, requires_grad=True))
    def forward(self, x):
        return x * torch.exp(self.t)
    
class ExpMultiplierLogit(nn.Module):
    def __init__(self, initial_value=0.0):
        super(ExpMultiplier, self).__init__()
        self.t = nn.Parameter(torch.tensor(np.log(1/initial_value), requires_grad=True))
    def forward(self, x):
        return x * torch.log(torch.exp(self.t))