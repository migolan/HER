import torch
import torch.nn.functional as F


class SimpleDQN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleDQN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BFP_DQN(SimpleDQN):
    def __init__(self, state_dim, hidden_dim):
        super(BFP_DQN, self).__init__(state_dim*2, hidden_dim, state_dim)

    def forward(self, x):
        x = [s.augstate().float() for s in x]
        x = torch.stack(x)
        return super(BFP_DQN, self).forward(x)