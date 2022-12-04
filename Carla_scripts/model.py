import torch
import torch.nn as nn
from torch.distributions import Normal
from torchvision.models import mnasnet1_0

# Conv Networks
class QNetworkConv(nn.Module):
    def __init__(self, cfg):
        super(QNetworkConv, self).__init__()
        self.model = mnasnet1_0(weights='DEFAULT')
        self.model.layers = nn.Sequential(*list(self.model.layers.children())[1:])
        self.conv1 = nn.Conv2d(cfg.motion_batch, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.act = nn.ReLU()

        self.action_net = nn.Sequential(
            nn.Linear(cfg.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )

        self.fcn = nn.Sequential(nn.Linear(1256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, cfg.output_dim))

    def forward(self, states, actions):
        s = self.conv1(states)
        s = self.model(s)
        s = self.act(s)

        a = self.action_net(actions)
        x = torch.cat([s, a], dim=-1)
        x = self.fcn(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class ACNetworkConv(nn.Module):
    def __init__(self, cfg):
        super(ACNetworkConv, self).__init__()
        self.Q1 = QNetworkConv(cfg)
        self.Q2 = QNetworkConv(cfg)

    def forward(self, states, actions):
        return self.Q1(states, actions), self.Q2(states, actions)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class PolicyNetworkConv(nn.Module):
    def __init__(self, cfg):
        super(PolicyNetworkConv, self).__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.eps = 1e-6

        self.model = mnasnet1_0(weights='DEFAULT')
        self.model.layers = nn.Sequential(*list(self.model.layers.children())[1:])
        self.conv1 = nn.Conv2d(cfg.motion_batch, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.act = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, cfg.policy_output_dim)
        )

    def forward(self, states):
        s = self.conv1(states)
        s = self.model(s)
        s = self.act(s)
        s = self.fc(s)

        mean, log_std = torch.chunk(s, 2, dim=-1)
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
