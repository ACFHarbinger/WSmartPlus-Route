"""
Hyperparameter Optimization (HPO) Module.

This module provides a suite of algorithms and utilities for optimizing the hyperparameters
of Reinforcement Learning models within the WSmart-Route framework. The primary optimization
engine is Differential Evolution Hyperband (DEHB), which efficiently searches large, mixed
hyperparameter spaces.

Key Components:
- **DEHB (`dehb.py`)**: The `DifferentialEvolutionHyperband` class combines the global search
  capabilities of Differential Evolution with the efficient resource allocation of Hyperband (Successive Halving).
  It supports:
    - Multi-fidelity optimization (budget-based).
    - Handling of continuous, integer, and categorical hyperparameters via `ConfigSpace`.
    - Parallel execution (synchronous or asynchronous).
    - Integration with Weights & Biases for logging.

- **Differential Evolution (`de.py`)**: Standalone implementations of Standard and Asynchronous
  Differential Evolution (`DifferentialEvolution`, `AsyncDifferentialEvolution`) serve as the
  evolutionary engine for DEHB and can also be used independently.

- **Configuration Management (`dehb_config_repo.py`)**: The `ConfigRepository` tracks the
  state/history of all configurations evaluated during the optimization process.

- **Bracket Management (`dehb_shb_manager.py`)**: The `SynchronousHalvingBracketManager`
  orchestrates the Successive Halving schedules within DEHB.

- **HPO Facade (`hpo.py`)**: Provides high-level entry points (`optimize_model`) and wrappers
  for various HPO libraries (Optuna, DEAP, etc.), although DEHB is the preferred internal method.

Usage:
    The module is typically accessed via the `hpo.py` script or by instantiating `DifferentialEvolutionHyperband`
    directly for custom optimization loops.
"""
