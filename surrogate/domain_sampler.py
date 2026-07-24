"""domain_sampler module.

Handles core functionality and definitions."""
import torch
import numpy as np
from scipy.stats import qmc
BOUNDS_STATE = {'M': [1000.0, 5000.0], 'T': [70.0, 160.0], 'c_u': [0.4, 0.95], 'c_w': [0.05, 0.5], 'c_b': [0.0, 0.05]}
BOUNDS_ACTION = {'Q': [-20000.0, 20000.0], 'P': [0.01, 10.0]}
BOUNDS_FEED = {'F_in': [5.0, 15.0], 'T_in': [70.0, 110.0], 'c_u_in': [0.4, 0.7]}

class DomainSampler:
    """DomainSampler class.

Provides state and behavior for DomainSampler."""

    def __init__(self, seed=42):
        self.state_dim = len(BOUNDS_STATE)
        self.action_dim = len(BOUNDS_ACTION)
        self.feed_dim = len(BOUNDS_FEED)
        self.total_dim = self.state_dim + self.action_dim + self.feed_dim
        self.sampler = qmc.Sobol(d=self.total_dim, scramble=True, seed=seed)
        all_bounds = list(BOUNDS_STATE.values()) + list(BOUNDS_ACTION.values()) + list(BOUNDS_FEED.values())
        self.l_bounds = np.array([b[0] for b in all_bounds])
        self.u_bounds = np.array([b[1] for b in all_bounds])

    def sample_batch(self, batch_size_power_of_two=12):
        """Executes sample_batch operations."""
        n_samples = 2 ** batch_size_power_of_two
        sampler = qmc.Sobol(d=self.total_dim, scramble=True)
        unit_samples = sampler.random_base2(m=batch_size_power_of_two)
        physical_samples = qmc.scale(unit_samples, self.l_bounds, self.u_bounds)
        states = physical_samples[:, :self.state_dim]
        actions = physical_samples[:, self.state_dim:self.state_dim + self.action_dim]
        feeds = physical_samples[:, self.state_dim + self.action_dim:]
        state_tensor = torch.tensor(states, dtype=torch.float32, requires_grad=False)
        action_tensor = torch.tensor(actions, dtype=torch.float32, requires_grad=False)
        feed_tensor = torch.tensor(feeds, dtype=torch.float32, requires_grad=False)
        fractions = state_tensor[:, 2:5]
        fraction_sums = fractions.sum(dim=1, keepdim=True)
        state_tensor[:, 2:5] = fractions / fraction_sums
        return (state_tensor, action_tensor, feed_tensor)