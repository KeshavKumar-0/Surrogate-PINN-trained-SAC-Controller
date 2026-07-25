"""learner module.

Handles core functionality and definitions."""
import os
import sys
import io
import signal
import zmq
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from communication.ipc_bus import get_context, create_pull_socket, create_weight_pub_socket
from rl_agent.networks import SACActor, SACCritic, extract_actor_obs
from rl_agent.buffer import FrameStackBuffer
GAMMA = 0.99
TAU = 0.005
LR = 0.0003
BATCH_SIZE = 256
UPDATE_FREQ = 10
GRAD_CLIP = 1.0
WARMUP_STEPS = 5000

class RunningNormalizer:
    """RunningNormalizer class.

Provides state and behavior for RunningNormalizer."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 1.0

    def update(self, x):
        """Executes update operations."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        """Executes std operations."""
        if self.n < 2:
            return 1.0
        return max(np.sqrt(self.M2 / (self.n - 1)), 1e-08)

    def normalize(self, x):
        """Executes normalize operations."""
        return (x - self.mean) / self.std

class SACLearner:
    """SACLearner class.

Provides state and behavior for SACLearner."""

    def __init__(self, state_dim=8, action_dim=2, k=10):
        self.hist_dim = k * (state_dim + action_dim)
        self.actor_hist_dim = k * (5 + action_dim)  # Observable states (M, T, F_in, T_in, c_u_in)
        self.action_dim = action_dim
        self.actor = SACActor(self.actor_hist_dim, action_dim)
        self.critic = SACCritic(self.hist_dim, action_dim)
        self.critic_target = SACCritic(self.hist_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR)
        self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=LR)
        self.buffer = FrameStackBuffer(max_size=500000, state_dim=state_dim, action_dim=action_dim, k=k)
        self.reward_normalizer = RunningNormalizer()
        self.total_steps = 0

    def update_parameters(self):
        """Executes update_parameters operations."""
        if self.buffer.size < max(BATCH_SIZE * 10, WARMUP_STEPS):
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample(BATCH_SIZE)
        mask_batch = 1.0 - done_batch
        reward_np = reward_batch.numpy().flatten()
        normalized_rewards = np.array([self.reward_normalizer.normalize(r) for r in reward_np], dtype=np.float32).reshape(-1, 1)
        reward_batch = torch.FloatTensor(normalized_rewards)
        with torch.no_grad():
            actor_next_state_batch = extract_actor_obs(next_state_batch)
            next_state_action, next_state_log_pi = self.actor.sample(actor_next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.log_alpha.exp() * next_state_log_pi
            next_q_value = reward_batch + mask_batch * GAMMA * min_qf_next_target
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.critic_optim.step()
        
        actor_state_batch = extract_actor_obs(state_batch)
        pi, log_pi = self.actor.sample(actor_state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (self.log_alpha.exp() * log_pi - min_qf_pi).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.actor_optim.step()
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    def add_transition(self, s, a, r, s_next, done):
        """Executes add_transition operations."""
        self.reward_normalizer.update(r)
        self.buffer.add(s, a, r, s_next, done)
        self.total_steps += 1

    def get_actor_weights(self):
        """Executes get_actor_weights operations."""
        buffer = io.BytesIO()
        torch.save(self.actor.state_dict(), buffer)
        return buffer.getvalue()

def main():
    """Executes main operations."""
    global learner
    learner = SACLearner(state_dim=8, action_dim=2, k=10)
    pull_socket = create_pull_socket()
    pub_socket = create_weight_pub_socket()

    def handle_sigterm(signum, frame):
        """Executes handle_sigterm operations."""
        os.makedirs('checkpoints', exist_ok=True)
        save_path = 'checkpoints/sac_actor_final.pt'
        torch.save(learner.actor.state_dict(), save_path)
        pull_socket.close()
        pub_socket.close()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    poller = zmq.Poller()
    poller.register(pull_socket, zmq.POLLIN)
    while True:
        while True:
            socks = dict(poller.poll(timeout=0))
            if pull_socket in socks and socks[pull_socket] == zmq.POLLIN:
                msg = pull_socket.recv()
                data = np.frombuffer(msg, dtype=np.float32)
                s = data[0:8]
                a = data[8:10]
                r = float(data[10])
                s_next = data[11:19]
                done = data[19]
                learner.add_transition(s, a, r, s_next, done)
            else:
                break
        if learner.buffer.size > max(BATCH_SIZE * 10, WARMUP_STEPS):
            for _ in range(UPDATE_FREQ):
                learner.update_parameters()
            pub_socket.send(learner.get_actor_weights())
            print(f'Learner update | Buffer size: {learner.buffer.size} | Total steps: {learner.total_steps}')
            sys.stdout.flush()
if __name__ == '__main__':
    main()