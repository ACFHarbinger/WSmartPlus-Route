"""
Pipeline Module.

This package orchestrates the primary operational workflows of the WSmart-Route framework:
- **Features**: Training, evaluation, and testing entry points.
- **RL**: Reinforcement learning and imitation learning subpackages.
- **Simulator**: The underlying physics engine and environment management.

Attributes:
    features: Sub-package for train, eval, and test entry points.
    rl: Sub-package for RL algorithms, baselines, and HPO.
    simulations: Sub-package for the simulation engine and state machine.

Example:
    >>> from logic.src.pipeline.features.eval import run_evaluate_model
    >>> from logic.src.pipeline.simulations.simulator import Simulator
"""
