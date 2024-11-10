import torch
import torch.nn as nn

class ExpMultiplier(nn.Module):
    def __init__(self, initial_value=0.0):
        super(ExpMultiplier, self).__init__()
        self.t = nn.Parameter(torch.tensor(initial_value, requires_grad=True))
    def forward(self, x):
        return x * torch.exp(self.t)