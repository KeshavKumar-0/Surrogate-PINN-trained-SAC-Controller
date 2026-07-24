"""pinn_export module.

Handles core functionality and definitions."""
import os
import sys
import time
import json
import torch
import numpy as np
import zmq
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from communication.ipc_bus import create_push_socket, create_telemetry_pub_socket, get_context, poll_socket_with_timeout, PORTS
from surrogate.pinn_model import PINNSurrogate
Q_MIN, Q_MAX = (-20000.0, 20000.0)
P_MIN, P_MAX = (0.01, 10.0)

def latent_to_physical(action_latent):
    """Executes latent_to_physical operations."""
    Q = (action_latent[0] + 1.0) / 2.0 * (Q_MAX - Q_MIN) + Q_MIN
    P = (action_latent[1] + 1.0) / 2.0 * (P_MAX - P_MIN) + P_MIN
    return (Q, P)

def export_surrogate_to_jit():
    """Executes export_surrogate_to_jit operations."""
    model = PINNSurrogate()
    weights_path = 'checkpoints/pinn_weights_offline.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    dummy_state = torch.zeros(1, 5)
    dummy_action = torch.zeros(1, 2)
    dummy_feed = torch.zeros(1, 3)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (dummy_state, dummy_action, dummy_feed))
    os.makedirs('checkpoints', exist_ok=True)
    export_path = 'checkpoints/pinn_surrogate.pt'
    traced_model.save(export_path)
    return traced_model

def create_control_req_socket():
    """Executes create_control_req_socket operations."""
    context = get_context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{PORTS['control']}")
    return socket

def check_terminal(state):
    """Executes check_terminal operations."""
    M, T, c_u, c_w, c_b = state
    if T < 75.0 or T > 140.0 or c_b > 0.02 or (M < 500.0) or (M > 4500.0):
        return True
    return False

def calculate_reward(state, action_latent, prev_action_latent, feed, done):
    """Executes calculate_reward operations."""
    M, T, c_u, c_w, c_b = state
    F_in, T_in, c_u_in = feed
    if done:
        return -10.0
    revenue = min(max(0.0, (c_u - c_u_in) * 50.0), 20.0)
    Q_physical, _ = latent_to_physical(action_latent)
    cost_thermal = abs(Q_physical / 20000.0) * 1.0
    cost_pressure = abs(action_latent[1] - -0.5) * 2.0
    cost_slew = 0.0
    if prev_action_latent is not None:
        cost_slew = (abs(action_latent[0] - prev_action_latent[0]) + abs(action_latent[1] - prev_action_latent[1])) * 2.5
    opex = cost_thermal + cost_pressure + cost_slew
    target_tracking_bonus = 2.0 if 85.0 <= T <= 95.0 else 0.0
    if c_b <= 0.008:
        quality_penalty = 0.0
        target_tracking_bonus += 1.0
    else:
        quality_penalty = min((c_b - 0.008) / 0.012 * 5.0, 5.0)
    net_profit = revenue - opex - quality_penalty + target_tracking_bonus
    return 1.0 + net_profit

def reset_plant_state(episode_count):
    """Executes reset_plant_state operations."""
    difficulty = min(episode_count / 500.0, 1.0)
    F_nom, T_nom, c_nom = (10.0, 90.0, 0.55)
    F_in = np.random.uniform(F_nom - 5.0 * difficulty, F_nom + 5.0 * difficulty)
    T_in = np.random.uniform(T_nom - 20.0 * difficulty, T_nom + 20.0 * difficulty)
    c_u_in = np.random.uniform(c_nom - 0.15 * difficulty, c_nom + 0.15 * difficulty)
    state = torch.tensor([[2000.0, 90.0, c_u_in, 1.0 - c_u_in, 0.0]], dtype=torch.float32)
    feed = torch.tensor([[F_in, T_in, c_u_in]], dtype=torch.float32)
    return (state, feed)

def run_simulation_loop(traced_model):
    """Executes run_simulation_loop operations."""
    req_actor = create_control_req_socket()
    push_buffer = create_push_socket()
    pub_telemetry = create_telemetry_pub_socket()
    time.sleep(0.5)
    episode_count = 0
    current_state, current_feed = reset_plant_state(episode_count)
    prev_action_latent = None
    step_count = 0
    try:
        while True:
            state_flat = current_state.numpy().flatten()
            feed_flat = current_feed.numpy().flatten()
            obs_np = np.concatenate([state_flat, feed_flat])
            req_actor.send(obs_np.astype(np.float32).tobytes())
            if not poll_socket_with_timeout(req_actor, timeout_ms=2000):
                print('Warning: Actor node timeout. Reconnecting REQ socket...')
                req_actor.setsockopt(zmq.LINGER, 0)
                req_actor.close()
                req_actor = create_control_req_socket()
                continue
            action_bytes = req_actor.recv()
            action_latent = np.frombuffer(action_bytes, dtype=np.float32).copy()
            if np.any(np.isnan(action_latent)) or np.any(np.isinf(action_latent)):
                action_latent = np.array([0.0, -0.5], dtype=np.float32)
            action_latent = np.clip(action_latent, -1.0, 1.0)
            Q_physical, P_physical = latent_to_physical(action_latent)
            action_physical = np.array([Q_physical, P_physical], dtype=np.float32)
            action_tensor = torch.from_numpy(action_physical).float().unsqueeze(0)
            with torch.no_grad():
                next_state = traced_model(current_state, action_tensor, current_feed)
            next_state_flat = next_state[0].tolist()
            done = check_terminal(next_state_flat)
            reward = calculate_reward(next_state_flat, action_latent, prev_action_latent, feed_flat, done)
            next_obs_np = np.concatenate([next_state.numpy().flatten(), feed_flat])
            transition_arr = np.concatenate([obs_np, action_latent, np.array([reward], dtype=np.float32), next_obs_np, np.array([1.0 if done else 0.0], dtype=np.float32)])
            push_buffer.send(transition_arr.astype(np.float32).tobytes())
            if step_count % 10 == 0:
                metrics = {'source': 'PINN_Surrogate', 'mass': next_state_flat[0], 'temperature': next_state_flat[1], 'c_urea': next_state_flat[2], 'c_water': next_state_flat[3], 'c_biuret': next_state_flat[4], 'F_in': float(feed_flat[0]), 'T_in': float(feed_flat[1]), 'c_u_in': float(feed_flat[2]), 'input_heat': float(Q_physical), 'pressure': float(P_physical), 'reward': float(reward)}
                pub_telemetry.send_string(json.dumps(metrics))
            if step_count % 100 == 0:
                print(f'PINN Sim | Step: {step_count} | Ep: {episode_count} | Reward: {reward:.4f} | Mass: {next_state_flat[0]:.1f}')
                sys.stdout.flush()
            if done:
                episode_count += 1
                current_state, current_feed = reset_plant_state(episode_count)
                prev_action_latent = None
                req_actor.send(np.zeros(8, dtype=np.float32).tobytes())
                if poll_socket_with_timeout(req_actor, timeout_ms=2000):
                    req_actor.recv()
            else:
                current_state = next_state
                prev_action_latent = action_latent.copy()
                F_in = current_feed[0, 0].item()
                T_in = current_feed[0, 1].item()
                c_u_in = current_feed[0, 2].item()
                F_in += np.random.normal(0.0, 0.05)
                T_in += np.random.normal(0.0, 0.1)
                c_u_in += np.random.normal(0.0, 0.001)
                F_in = np.clip(F_in, 5.0, 15.0)
                T_in = np.clip(T_in, 70.0, 110.0)
                c_u_in = np.clip(c_u_in, 0.4, 0.7)
                current_feed = torch.tensor([[F_in, T_in, c_u_in]], dtype=torch.float32)
            step_count += 1
    except KeyboardInterrupt:
        pass
if __name__ == '__main__':
    jit_model = export_surrogate_to_jit()
    run_simulation_loop(jit_model)