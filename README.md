# Surrogate-PINN-SAC-Controller

An asynchronous, multiprocess Reinforcement Learning architecture designed for optimal control of a chemical Urea Reactor. This project replaces a computationally expensive numerical ODE solver with a fast **Hybrid Physics-Informed Neural Network (PINN) Surrogate**, which acts as a simulated, high-fidelity environment to train a **Soft Actor-Critic (SAC)** agent.

Traditional numerical solvers (like RK4) are too slow for generating the millions of transitions required for RL training. The PINN surrogate solves this by providing instantaneous state predictions. The RL agent then optimizes the continuous reactor control policy to minimize biuret formation while maximizing urea output and maintaining strict thermal bounds.

---

## 📁 Repository Structure

The codebase is modular, cleanly separating the physics engine, the neural network surrogate, the RL agent, and the inter-process communication layers.

```
├── communication/
│   └── ipc_bus.py               # ZeroMQ (ZMQ) configuration for asynchronous PUB/SUB and REQ/REP IPC.
├── core_physics/
│   ├── constants.py             # Physical constants (Activation energies, heat capacities, etc).
│   ├── dynamics.py              # Mathematical ODEs dictating reactor state changes (Mass, Temp, Concentrations).
│   └── thermodynamics.py        # Helper functions for temperature conversions and mixture properties.
├── rl_agent/
│   ├── actor_node.py            # SAC Actor process. Replies to the Plant with actions; pulls weights from Learner.
│   ├── buffer.py                # Replay Buffer storing (s, a, r, s') transitions for offline SAC training.
│   ├── learner.py               # SAC Learner process. Trains the Actor/Critic networks and broadcasts weights.
│   └── networks.py              # Neural network definitions for SAC (Actor and Critic), including state normalization.
├── surrogate/
│   ├── dataset_generator.py     # RK4 solver to generate highly accurate offline datasets for PINN training.
│   ├── domain_sampler.py        # Quasi-Monte Carlo (Sobol) sampler for generating physical bounds.
│   ├── pinn_export.py           # The Plant process. Runs the trained PINN, communicates with Actor/Learner.
│   ├── pinn_model.py            # Neural network architecture for the PINN (predicts state deltas).
│   └── pinn_train.py            # Supervised + Physics Loss training engine for the PINN surrogate.
├── telemetry/
│   └── monitor.py               # Background process that logs real-time operational data to CSV.
├── run_plant.py                 # The orchestrator. Spawns and manages all asynchronous processes.
├── requirements.txt             # Python package dependencies.
└── README.md                    # Project documentation.
```

---

## ⚙️ Hyperparameters

### SAC Agent (Reinforcement Learning)
- **Learning Rate**: `3e-4`
- **Discount Factor (Gamma)**: `0.99`
- **Soft Update (Tau)**: `0.005`
- **Batch Size**: `256`
- **History Length (k)**: `10` (The agent evaluates sequences of 10 consecutive frames).
- **Exploration Noise**: Ornstein-Uhlenbeck (OU) Noise with `sigma = 0.15` and `theta = 0.15` to ensure temporally correlated, smooth physical exploration.

### PINN Surrogate (Physics-Informed Training)
- **Architecture**: 3 Hidden Layers (128 units each) with `Tanh` activations.
- **Adam Optimization**: `10,000` Epochs, Learning Rate `1e-4`, Batch Size `4096`.
- **L-BFGS Optimization**: `2,000` Epochs, Learning Rate `0.1`, Fixed Dataset Batch `20,000`.
- **Loss Weights**: 
  - Data Loss (MSE against RK4) = `1.0`
  - Physics Residual Loss = `0.1` (Dynamically clamped)
  - Physical Bounds Penalty = `1000.0` (For enforcing positive mass/temperature).

---

## 🚀 Usage Guide

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the PINN Surrogate
Before running the RL agent, you must train the neural network to approximate the core physics ODEs:
```bash
python surrogate/pinn_train.py
```
This automatically generates a 100,000-sample RK4 offline dataset, trains the surrogate using both Adam and L-BFGS optimizers, and serializes the highly-accurate weights to `checkpoints/pinn_weights_offline.pth`.

### 3. Run the Asynchronous RL Architecture
Launch the orchestrator script. This boots all 4 background processes (Plant, Actor, Learner, Telemetry) across separate CPU cores and manages crash recovery:
```bash
python run_plant.py
```
*Note: Wait a few seconds for all ZeroMQ sockets to bind and connect.*

To safely terminate the system and trigger a graceful shutdown (which instantly saves the final SAC Actor weights to `checkpoints/sac_actor_final.pt`), press `Ctrl+C`.

---

## 📊 System Architecture

The architecture is entirely decoupled and asynchronous, utilizing **ZeroMQ (ZMQ)** for extremely fast Inter-Process Communication (IPC) across four distinct processes:

1. **Surrogate Plant**: Advances time based on actions from the Actor using the trained PINN, and pushes transition experiences to the Learner.
2. **Actor Node**: Receives observations and instantly replies with continuous latent control actions (using OU exploration noise).
3. **Learner**: Asynchronously trains the SAC networks and broadcasts the latest Actor weights to the Actor Node without blocking the simulation loop.
4. **Telemetry**: Subscribes to plant metrics and logs real-time operational data.
