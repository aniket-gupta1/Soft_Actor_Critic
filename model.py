import torch
import torch.nn as nn
from torch.distributions import Normal
from torchvision.models import mnasnet1_0

class QNetwork(nn.Module):
    def __init__(self, cfg):
        super(QNetwork, self).__init__()
        self.model = []

        input_dim = cfg.input_dim
        for h_unit in cfg.hidden_units:
            self.model.append(nn.Linear(input_dim, h_unit))
            self.model.append(nn.ReLU())
            input_dim = h_unit

        self.model.append(nn.Linear(input_dim, cfg.output_dim))
        # self.model.append(nn.)
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class ACNetwork(nn.Module):
    def __init__(self, cfg):
        super(ACNetwork, self).__init__()
        self.Q1 = QNetwork(cfg)
        self.Q2 = QNetwork(cfg)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.Q1(x), self.Q2(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class PolicyNetwork(nn.Module):
    def __init__(self, cfg):
        super(PolicyNetwork, self).__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.eps = 1e-6
        self.model = []

        input_dim = cfg.policy_input_dim
        for h_unit in cfg.policy_hidden_units:
            self.model.append(nn.Linear(input_dim, h_unit))
            self.model.append(nn.ReLU())
            input_dim = h_unit

        self.model.append(nn.Linear(input_dim, cfg.policy_output_dim))
        # self.model.append(nn.)
        self.model = nn.Sequential(*self.model)

    def forward(self, states):
        mean, log_std = torch.chunk(self.model(states), 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        std = log_std.exp()
        normals = Normal(mean, std)

        # Sample actions from the gaussian distribution
        x_actions = normals.rsample()
        actions = torch.tanh(x_actions)

        # Calculate entropies
        log_probs = normals.log_prob(x_actions) - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(mean)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
