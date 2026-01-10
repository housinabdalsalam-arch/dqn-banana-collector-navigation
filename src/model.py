import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    "Multi layer perceptron (Q) Network for aprroximating Q(s,a)"

    def __init__(self,state_size:int,action_size:int,seed:int,fc1_units:int=64, fc2_units:int=64):
        super().__init__()
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.out = nn.Linear(fc2_units, action_size)
        self.out = nn.Linear(fc2_units, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)




