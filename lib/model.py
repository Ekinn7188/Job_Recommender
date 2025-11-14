import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.FCL = nn.Linear(10,1, dtype=torch.float32)
    
    def forward(self, x : torch.Tesnor):
        x = self.FCL(x)

        return x
