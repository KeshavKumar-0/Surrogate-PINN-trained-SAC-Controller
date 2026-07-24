"""actor_node module.

Handles core functionality and definitions."""
import os
import sys
import time
import io
import torch
import numpy as np
import zmq
import collections
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from communication.ipc_bus import get_context, create_weight_sub_socket, PORTS
from rl_agent.networks import SACActor

def create_control_rep_socket():
    """Executes create_control_rep_socket operations."""
    context = get_context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORTS['control']}")
    return socket

class OUNoise:
    """OUNoise class.

Provides state and behavior for OUNoise."""

    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        """Executes reset operations."""
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """Executes noise operations."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

def main():
    """Executes main operations."""
    k = 10
    state_dim = 8
    action_dim = 2
    hist_dim = state_dim + action_dim
    actor = SACActor(input_dim=k * hist_dim, action_dim=action_dim)
    actor.eval()
    rep_plant = create_control_rep_socket()
    sub_weights = create_weight_sub_socket()
    poller = zmq.Poller()
    poller.register(sub_weights, zmq.POLLIN)
    history_queue = collections.deque([np.zeros(hist_dim, dtype=np.float32)] * k, maxlen=k)
    exploration_noise = OUNoise(action_dim, sigma=0.15)
    step_count = 0
    try:
        while True:
            socks = dict(poller.poll(timeout=0))
            if sub_weights in socks and socks[sub_weights] == zmq.POLLIN:
                weight_bytes = sub_weights.recv()
                buffer = io.BytesIO(weight_bytes)
                actor.load_state_dict(torch.load(buffer, weights_only=True))
            state_bytes = rep_plant.recv()
            state_np = np.frombuffer(state_bytes, dtype=np.float32)
            if state_np[0] == 0.0:
                history_queue.clear()
                history_queue.extend([np.zeros(hist_dim, dtype=np.float32)] * k)
                exploration_noise.reset()
            prev_action = history_queue[-1][-action_dim:]
            current_frame = np.concatenate([state_np, prev_action])
            history_queue.append(current_frame)
            stacked_input = np.concatenate(list(history_queue))
            state_tensor = torch.FloatTensor(stacked_input).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _ = actor.sample(state_tensor)
            action_np = action_tensor.squeeze(0).numpy()
            action_np += exploration_noise.noise()
            action_np = np.clip(action_np, -1.0, 1.0)
            if step_count % 100 == 0:
                print(f'Actor Node | Step: {step_count} | Output Action: {action_np}')
                sys.stdout.flush()
            rep_plant.send(action_np.astype(np.float32).tobytes())
            step_count += 1
    except KeyboardInterrupt:
        rep_plant.close()
        sub_weights.close()
if __name__ == '__main__':
    main()