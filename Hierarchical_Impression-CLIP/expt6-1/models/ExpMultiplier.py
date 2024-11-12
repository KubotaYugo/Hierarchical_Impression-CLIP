import torch
import torch.nn as nn


class ExpMultiplier(nn.Module):
    def __init__(self, initial_value=0.07):
        super(ExpMultiplier, self).__init__()
        self.t = nn.Parameter(torch.tensor(initial_value, requires_grad=True))
    def forward(self, x):
        return x * torch.exp(self.t)
    
class ExpMultiplierLogit(nn.Module):    # CLIP本家の実装
    def __init__(self, initial_value=0.07):
        super(ExpMultiplierLogit, self).__init__()
        self.t = nn.Parameter(torch.log(torch.tensor(1/initial_value)))
    def forward(self, x):
        t_clamp = torch.clamp(self.t, 0, 4.6052)
        return x * torch.exp(t_clamp)