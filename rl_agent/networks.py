"""networks module.

Handles core functionality and definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-06

def weights_init_(m):
    """Executes weights_init_ operations."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class SACActor(nn.Module):
    """SACActor class.

Provides state and behavior for SACActor."""

    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(SACActor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        """Executes forward operations."""
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return (mean, log_std)

    def sample(self, state):
        """Executes sample operations."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return (action, log_prob)

class SACCritic(nn.Module):
    """SACCritic class.

Provides state and behavior for SACCritic."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACCritic, self).__init__()
        self.q1_net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2_net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.apply(weights_init_)

    def forward(self, state, action):
        """Executes forward operations."""
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return (q1, q2)