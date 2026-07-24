# Urea Reactor Control System

An asynchronous, multiprocess Reinforcement Learning architecture for optimal control of a Urea Reactor. This project replaces a computationally expensive numerical ODE solver with a fast **Hybrid Physics-Informed Neural Network (PINN) Surrogate**, which then acts as a simulated environment to train a Soft Actor-Critic (SAC) reinforcement learning agent.

Because numerical solvers (like RK4) are too slow for millions of RL interactions, the PINN acts as a high-fidelity proxy. The RL agent optimizes the reactor control policy to minimize biuret formation while maximizing urea output and maintaining thermal bounds.

## System Architecture

The architecture is entirely decoupled and asynchronous. It utilizes **ZeroMQ (ZMQ)** for extremely fast Inter-Process Communication (IPC) across four distinct processes:

1. **Surrogate Plant (`surrogate/pinn_export.py`)**: Runs the pre-trained PINN model. It maintains the physical state of the reactor, advances time based on actions from the Actor, and pushes transition experiences `(s, a, r, s')` to the Learner.
2. **Actor Node (`rl_agent/actor_node.py`)**: Receives current observations from the Plant and instantly replies with continuous latent control actions (augmented with Ornstein-Uhlenbeck exploration noise). It periodically pulls updated neural network weights from the Learner.
3. **Learner (`rl_agent/learner.py`)**: Maintains a circular replay buffer of transitions. It asynchronously trains the SAC networks (Actor/Critic) and broadcasts the latest Actor weights to the Actor Node.
4. **Telemetry (`telemetry/monitor.py`)**: Subscribes to plant metrics and logs real-time operational data (mass, temperature, pressure, heat input, biuret fractions) to `logs/telemetry_history.csv`.

## The Hybrid PINN Surrogate

Standard PINNs trained purely on an implicit Euler residual can suffer from drifting inaccuracies or collapse into trivial solutions. To solve this, the PINN unsupervised training engine in this repository uses a **Hybrid Supervised Data + Physics Loss**:
- An offline RK4 dataset generator (`dataset_generator.py`) produces exact ground-truth transitions over the physical bounds of the domain using Quasi-Monte Carlo (Sobol) sampling.
- The training loop (`pinn_train.py`) evaluates an MSE "Data Loss" against the RK4 transitions alongside the implicit Euler "Physics Residual Loss", deeply anchoring the surrogate in mathematical reality.

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `torch`
- `numpy`
- `scipy`
- `pyzmq`

## Usage

### 1. Train the PINN Surrogate
Before running the RL agent, you must train the PINN to approximate the core physics ODEs:
```bash
python surrogate/pinn_train.py
```
This will automatically generate the RK4 offline dataset, train the surrogate using Adam and L-BFGS optimizers, and serialize the weights to `checkpoints/pinn_weights_offline.pth`.

### 2. Run the Asynchronous RL Architecture
Launch the orchestrator, which boots all 4 background processes (Plant, Actor, Learner, Telemetry) and manages crash recovery:
```bash
python run_plant.py
```
*Note: Wait a few seconds for all ZeroMQ sockets to bind and connect.*

To trigger a graceful shutdown and save the final SAC Actor weights (`checkpoints/sac_actor_final.pt`), press `Ctrl+C`.

## Logs
Output from the individual processes is piped to the `logs/` directory:
- `logs/monitor.log`
- `logs/learner.log`
- `logs/actor.log`
- `logs/plant.log`
