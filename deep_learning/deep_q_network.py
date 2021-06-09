import torch
import torch.nn as nn
import torch.nn.functional as F


class BananaQNN(nn.Module):
    def __init__(
        self, state_size, action_size, seed, fc1_units=64, fc2_units=64
    ) -> None:
        """ the architecture for the policy model, inputs are parameterized, 
        picked number of nodes corresponding (I hope) to something that will
        divide well into decision boundaries for 4 actions"""
        super(BananaQNN, self).__init__()
        uniform = lambda tensor: nn.init.xavier_uniform(tensor)
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        uniform(self.fc1.weight)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        uniform(self.fc2.weight)
        self.fc3 = nn.Linear(fc2_units, action_size)
        uniform(self.fc3.weight)

    def forward(self, state):
        """ how a forward pass will fire through this network, the goal is to 
        learn a policy mapping from the input state to the action to be taken """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

