import torch
import torch.nn as nn
import numpy as np


class ExpMultiplier(nn.Module):
    def __init__(self, initial_value=0.07):
        super(ExpMultiplier, self).__init__()
        self.t = nn.Parameter(torch.tensor(initial_value, requires_grad=True))
    def forward(self, x):
        return x * torch.exp(self.t)
    
class ExpMultiplierLogit(nn.Module):    # CLIP本家ベースの実装
    def __init__(self, initial_value=0.07):
        super(ExpMultiplierLogit, self).__init__()
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1/initial_value)]))
    def forward(self, x):
        logit_scale_clamped = torch.clamp(self.logit_scale, 0, np.log(100))
        return x * logit_scale_clamped.exp()

# temperture = ExpMultiplierLogit(0.00001)
# a = torch.tensor([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9]])
# b = temperture(a)
# print(b)