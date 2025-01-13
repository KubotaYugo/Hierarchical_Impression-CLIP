import torch
import numpy as np


class Temperature(torch.nn.Module):
    def __init__(self, initial_value=0.07):
        super(Temperature, self).__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor([np.log(1/initial_value)]))
    def forward(self, x):
        logit_scale_clamped = torch.clamp(self.logit_scale, 0, np.log(100))
        return x * logit_scale_clamped.exp()