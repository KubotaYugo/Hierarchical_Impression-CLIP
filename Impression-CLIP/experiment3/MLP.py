import torch.nn as nn
import torch.nn.functional as F

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.emb1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.emb2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.emb3 = nn.Sequential(nn.Linear(512, 512))
    def forward(self, x):
        x = self.emb1(x)
        x = self.emb2(x)
        x = self.emb3(x)
        x = F.normalize(x, dim=1)
        return x