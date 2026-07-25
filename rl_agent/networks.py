"""networks module.

Handles core functionality and definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-06

# Constants for State Normalization
MEAN_STATES = torch.tensor([3000.0, 110.0, 0.65, 0.25, 0.02])
STD_STATES = torch.tensor([1000.0, 40.0, 0.25, 0.1, 0.01])
MEAN_FEED = torch.tensor([10.0, 90.0, 0.55])
STD_FEED = torch.tensor([5.0, 20.0, 0.15])
MEAN_ACTS = torch.tensor([0.0, 0.0])
STD_ACTS = torch.tensor([1.0, 1.0])

MEAN_FRAME_CRITIC = torch.cat([MEAN_STATES, MEAN_FEED, MEAN_ACTS])
STD_FRAME_CRITIC = torch.cat([STD_STATES, STD_FEED, STD_ACTS])

MEAN_FRAME_ACTOR = torch.cat([MEAN_STATES[0:2], MEAN_FEED, MEAN_ACTS])
STD_FRAME_ACTOR = torch.cat([STD_STATES[0:2], STD_FEED, STD_ACTS])

def extract_actor_obs(batch_tensor):
    """Slices the fully-privileged 10-dim frame to the observable 7-dim frame for the Actor."""
    device = batch_tensor.device
    batch_size = batch_tensor.shape[0]
    k = batch_tensor.shape[-1] // 10
    
    reshaped = batch_tensor.view(batch_size, k, 10)
    indices = torch.tensor([0, 1, 5, 6, 7, 8, 9], device=device)
    sliced = torch.index_select(reshaped, dim=2, index=indices)
    return sliced.view(batch_size, k * 7)

def normalize_history(history_tensor, is_actor=False):
    """Normalizes the unrolled history buffer of states and actions."""
    device = history_tensor.device
    input_dim = history_tensor.shape[-1]
    
    if is_actor:
        k = input_dim // 7
        mean = MEAN_FRAME_ACTOR.to(device).repeat(k)
        std = STD_FRAME_ACTOR.to(device).repeat(k)
    else:
        k = input_dim // 10
        mean = MEAN_FRAME_CRITIC.to(device).repeat(k)
        std = STD_FRAME_CRITIC.to(device).repeat(k)
        
    return (history_tensor - mean) / std

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
        state = normalize_history(state, is_actor=True)
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
        state = normalize_history(state, is_actor=False)
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return (q1, q2)