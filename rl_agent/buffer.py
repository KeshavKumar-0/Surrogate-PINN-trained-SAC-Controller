"""buffer module.

Handles core functionality and definitions."""
import numpy as np
import torch

class FrameStackBuffer:
    """FrameStackBuffer class.

Provides state and behavior for FrameStackBuffer."""

    def __init__(self, max_size=1000000, state_dim=8, action_dim=2, k=10):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Executes add operations."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Executes sample operations."""
        hist_dim = self.state_dim + self.action_dim
        batch_states = np.zeros((batch_size, self.k * hist_dim), dtype=np.float32)
        batch_next_states = np.zeros((batch_size, self.k * hist_dim), dtype=np.float32)
        valid_indices = []
        while len(valid_indices) < batch_size:
            idx = np.random.randint(self.k, self.size)
            if self.size == self.max_size:
                if idx >= self.ptr and idx - self.k < self.ptr:
                    continue
            if not np.any(self.done[idx - self.k + 1:idx]):
                valid_indices.append(idx)
        ind = np.array(valid_indices)
        for i, idx in enumerate(ind):
            frames_curr = []
            for t in range(self.k):
                curr_idx = idx - self.k + 1 + t
                s = self.state[curr_idx]
                if curr_idx > 0 and self.done[curr_idx - 1] == 0.0:
                    a = self.action[curr_idx - 1]
                else:
                    a = np.zeros(self.action_dim, dtype=np.float32)
                frames_curr.append(np.concatenate([s, a]))
            batch_states[i] = np.concatenate(frames_curr)
            frames_next = []
            for t in range(self.k):
                if t < self.k - 1:
                    curr_idx = idx - self.k + 2 + t
                    s = self.state[curr_idx]
                    if curr_idx > 0 and self.done[curr_idx - 1] == 0.0:
                        a = self.action[curr_idx - 1]
                    else:
                        a = np.zeros(self.action_dim, dtype=np.float32)
                else:
                    s = self.next_state[idx]
                    a = self.action[idx]
                frames_next.append(np.concatenate([s, a]))
            batch_next_states[i] = np.concatenate(frames_next)
        return (torch.FloatTensor(batch_states), torch.FloatTensor(self.action[ind]), torch.FloatTensor(self.reward[ind]), torch.FloatTensor(batch_next_states), torch.FloatTensor(self.done[ind]))