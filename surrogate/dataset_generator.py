"""dataset_generator module.

Handles core functionality and definitions."""
import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_physics.dynamics import compute_odes
from surrogate.domain_sampler import DomainSampler

def rk4_step(state, action, feed_dict, dt=1.0, steps=10):
    """Executes rk4_step operations."""
    with torch.no_grad():
        h = dt / steps
        x = state.clone()
        for _ in range(steps):
            k1 = compute_odes(x, action, feed_dict)
            k2 = compute_odes(x + 0.5 * h * k1, action, feed_dict)
            k3 = compute_odes(x + 0.5 * h * k2, action, feed_dict)
            k4 = compute_odes(x + h * k3, action, feed_dict)
            x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

def generate_offline_dataset(num_samples=100000, save_path='checkpoints/offline_data.pt'):
    """Executes generate_offline_dataset operations."""
    if os.path.exists(save_path):
        print(f'Loading existing offline dataset from {save_path}...')
        return torch.load(save_path)
    print(f'Generating {num_samples} ground-truth transitions using RK4 solver...')
    sampler = DomainSampler()
    all_states = []
    all_actions = []
    all_feeds = []
    all_next_states = []
    power = 13
    chunk_size = 2 ** power
    num_chunks = int(np.ceil(num_samples / chunk_size))
    for i in range(num_chunks):
        state_k, action_k, feed_k = sampler.sample_batch(batch_size_power_of_two=power)
        feed_dict = {'F_in': feed_k[:, 0], 'T_in': feed_k[:, 1], 'c_u_in': feed_k[:, 2], 'c_w_in': 1.0 - feed_k[:, 2]}
        next_state_k = rk4_step(state_k, action_k, feed_dict, dt=1.0, steps=10)
        all_states.append(state_k)
        all_actions.append(action_k)
        all_feeds.append(feed_k)
        all_next_states.append(next_state_k)
        print(f'Generated chunk {i + 1}/{num_chunks}')
    states = torch.cat(all_states, dim=0)[:num_samples]
    actions = torch.cat(all_actions, dim=0)[:num_samples]
    feeds = torch.cat(all_feeds, dim=0)[:num_samples]
    next_states = torch.cat(all_next_states, dim=0)[:num_samples]
    dataset = {'states': states, 'actions': actions, 'feeds': feeds, 'next_states': next_states}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, save_path)
    print(f'Dataset saved to {save_path}')
    return dataset
if __name__ == '__main__':
    generate_offline_dataset(100000)