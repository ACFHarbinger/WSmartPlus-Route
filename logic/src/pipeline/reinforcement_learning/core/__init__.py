"""
Reinforcement Learning Core Algorithms.

This module implements the fundamental Deep Reinforcement Learning (DRL) algorithms, training loops,
and baseline mechanisms used to train the vehicle routing agents. It provides a unified framework
for Policy Gradient methods.

Key Algorithms:
- **REINFORCE (`reinforce.py`)**: The standard Monte Carlo Policy Gradient algorithm.
  - Class: `Reinforce`.
  - Supports baseline subtraction and entropy regularization.
- **PPO (`ppo.py`)**: Proximal Policy Optimization.
  - Class: `PPO`.
  - Implements the clipped objective function for more stable training updates.
- **DR-GRPO (`dr_grpo.py`)**: Group Relative Policy Optimization Done Right.
  - Class: `DR_GRPO`.
  - Specialized for group-wise relative advantages.
- **GSPO (`gspo.py`)**: Group Shared Policy Optimization.
  - Class: `GSPO`.
- **SAPO (`sapo.py`)**: Self-Adaptive Policy Optimization.
  - Class: `SAPO`.

Infrastructure:
- **Epoch Manager (`epoch.py`)**: The `EpochManager` class orchestrates the inner training loop for a single epoch.
  It manages:
    - Data loading and batching.
    - Model forward passes (Encoders/Decoders).
    - Loss computation (Actor and Critic).
    - Gradient backpropagation and optimization steps.
- **Baselines (`reinforce_baselines.py`)**: Implements various baselines to reduce variance in gradient estimation:
    - `WarmupBaseline` (exponential moving average).
    - `CriticBaseline` (learned state-value function).
    - `RolloutBaseline` (greedy rollout).
    - `NoBaseline`.

Usage:
    These classes are typically instantiated and driven by the `Trainer` in `logic/src/pipeline/train.py`.
"""
