"""pinn_train module.

Handles core functionality and definitions."""
import os
import sys
import torch
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate.pinn_model import PINNSurrogate
from core_physics.dynamics import compute_odes
from surrogate.dataset_generator import generate_offline_dataset
DT = 1.0

def physics_data_loss(model, state_k, action_k, feed_k, true_next_state):
    """Executes physics_data_loss operations."""
    state_k_plus_1 = model(state_k, action_k, feed_k)
    feed_dict = {'F_in': feed_k[:, 0], 'T_in': feed_k[:, 1], 'c_u_in': feed_k[:, 2], 'c_w_in': 1.0 - feed_k[:, 2]}
    scale_vector = torch.tensor([1000.0, 100.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(state_k.device)
    data_loss = torch.mean(((state_k_plus_1 - true_next_state) / scale_vector) ** 2)
    d_state_dt = compute_odes(state_k_plus_1, action_k, feed_dict)
    residual = state_k_plus_1 - state_k - DT * d_state_dt
    normalized_residual = residual / scale_vector
    physics_residual_loss = torch.mean(normalized_residual ** 2)
    penalty_mass = torch.mean(torch.relu(-state_k_plus_1[:, 0])) * 1000.0
    penalty_temp = torch.mean(torch.relu(50.0 - state_k_plus_1[:, 1])) * 1000.0
    penalty_frac_u = torch.mean(torch.relu(-state_k_plus_1[:, 2])) * 1000.0
    penalty_frac_w = torch.mean(torch.relu(-state_k_plus_1[:, 3])) * 1000.0
    penalty_frac_b = torch.mean(torch.relu(-state_k_plus_1[:, 4])) * 1000.0
    total_penalty = penalty_mass + penalty_temp + penalty_frac_u + penalty_frac_w + penalty_frac_b
    physics_weight = (total_penalty.detach() / (physics_residual_loss.detach() + 1e-08)).clamp(min=1.0, max=100.0)
    total_loss = data_loss + 0.1 * physics_weight * physics_residual_loss + total_penalty
    return total_loss

def train_surrogate():
    """Executes train_surrogate operations."""
    print('Initializing PINN Supervised + Physics Training Engine...')
    model = PINNSurrogate()
    dataset = generate_offline_dataset(num_samples=100000)
    optimizer_adam = optim.Adam(model.parameters(), lr=0.0001)
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=50)
    epochs_adam = 10000
    epochs_lbfgs = 2000
    batch_size = 4096
    print('Phase 1: Adam Optimization')
    for epoch in range(epochs_adam):
        idx = torch.randperm(dataset['states'].shape[0])[:batch_size]
        state_k = dataset['states'][idx]
        action_k = dataset['actions'][idx]
        feed_k = dataset['feeds'][idx]
        true_next = dataset['next_states'][idx]
        optimizer_adam.zero_grad()
        loss = physics_data_loss(model, state_k, action_k, feed_k, true_next)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()
        if epoch % 500 == 0:
            print(f'Adam Epoch {epoch:5d} | Total Loss: {loss.item():.6e}')
    print('\nPhase 2: L-BFGS Optimization')
    for epoch in range(epochs_lbfgs):
        idx = torch.randperm(dataset['states'].shape[0])[:batch_size]
        state_k = dataset['states'][idx]
        action_k = dataset['actions'][idx]
        feed_k = dataset['feeds'][idx]
        true_next = dataset['next_states'][idx]

        def closure():
            """Executes closure operations."""
            optimizer_lbfgs.zero_grad()
            loss = physics_data_loss(model, state_k, action_k, feed_k, true_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            return loss
        loss = optimizer_lbfgs.step(closure)
        if epoch % 200 == 0:
            print(f'L-BFGS Epoch {epoch:4d} | Total Loss: {loss.item():.6e}')
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/pinn_weights_offline.pth'
    torch.save(model.state_dict(), save_path)
    print(f'\nPINN weights successfully serialized to {save_path}')
if __name__ == '__main__':
    train_surrogate()