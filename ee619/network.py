import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, out_features)
        )
    
    def forward(self, x: torch.tensor) -> np.ndarray:
        return self.net(x)

class Policy(Net):
    def __init__(self):
        super().__init__(
            in_features=24,
            out_features=6,
            hidden_layers=64
        )