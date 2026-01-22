"""
Reinforcement Learning Pipeline.

This package contains the complete Reinforcement Learning pipeline for the WSmart-Route project.
It orchestrates the training, validation, and execution of deep learning agents designed to solve
Combinatorial Optimization problems (typically VRP/TSP variants).

Submodules:
- `core`: Fundamental RL algorithms (REINFORCE, PPO) and training infrastructure (Epoch Manager).
- `models`: Neural network architectures (Attention Model, GAT, etc.).
- `policies`: Heuristic and hybrid policies (e.g., Vectorized HGS) for baselining or hybrid execution.
- `hyperparameter_optimization`: Algorithms (DEHB) for tuning model hyperparameters.
- `meta`: Meta-learning strategies for dynamic objective weighting (Contextual Bandits, MORL).

Key Scripts:
- `train.py`: The main entry point for launching training experiments.
- `manager_train.py`: Training logic specific to the high-level "Manager" agent.
- `worker_train.py`: Training logic for the low-level "Worker" (routing) agent.
"""
