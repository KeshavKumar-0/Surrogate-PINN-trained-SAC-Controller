"""pinn_model module.

Handles core functionality and definitions."""
import torch
import torch.nn as nn
MEAN_STATES = torch.tensor([3000.0, 110.0, 0.65, 0.25, 0.02])
STD_STATES = torch.tensor([1000.0, 40.0, 0.25, 0.1, 0.01])
MEAN_ACTS = torch.tensor([0.0, 5.005])
STD_ACTS = torch.tensor([20000.0, 4.995])
MEAN_FEED = torch.tensor([10.0, 90.0, 0.55])
STD_FEED = torch.tensor([5.0, 20.0, 0.15])

class PINNSurrogate(nn.Module):
    """PINNSurrogate class.

Provides state and behavior for PINNSurrogate."""

    def __init__(self, state_dim=5, action_dim=2, feed_dim=3, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feed_dim = feed_dim
        input_dim = state_dim + action_dim + feed_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, state_dim))

    def forward(self, state, action, feed):
        """Executes forward operations."""
        norm_s = (state - MEAN_STATES.to(state.device)) / STD_STATES.to(state.device)
        norm_a = (action - MEAN_ACTS.to(action.device)) / STD_ACTS.to(action.device)
        norm_f = (feed - MEAN_FEED.to(feed.device)) / STD_FEED.to(feed.device)
        x = torch.cat([norm_s, norm_a, norm_f], dim=-1)
        delta = self.net(x)
        return state + delta * STD_STATES.to(state.device)