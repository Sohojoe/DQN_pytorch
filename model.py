import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        num_inputs = state_size
        self.in_layer = nn.Linear(num_inputs, 32)
        self.hidden_1 = nn.Linear(32, 32)
        self.out_layer = nn.Linear(32, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.in_layer(state))
        x = F.relu(self.hidden_1(x))
        x = self.out_layer(x)
        return x
