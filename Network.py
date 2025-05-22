import torch 
import torch.nn as nn 
import torch.nn.functional as F 


# A very simple network for Q value estimation 
class Network(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim = 128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hid_dim), 
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), 
            nn.ReLU(), 
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)