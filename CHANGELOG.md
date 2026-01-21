# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-21

### Added
- **New RL Pipeline (`logic/src/pipeline/rl/`)**: Modular, PyTorch Lightning-based architecture.
  - **Algorithms**: REINFORCE, PPO, SAPO, GSPO, DR-GRPO, POMO, SymNCO.
  - **Baselines**: Rollout (greedy), Warmup (exponential->greedy), POMO (mean-augmented), Critic (network).
  - **Meta-Learning**: Unified `MetaRLModule` with RNN, Contextual Bandit, and Pareto (MORL) strategies.
  - **Hierarchical RL**: `HRLModule` with Manager-Worker architecture.
  - **HPO**: Lightning-integrated Hyperparameter Optimization using Optuna and DEHB.
  - **Hybrid Policies**: `NeuralHeuristicHybrid` combining AM/TAM with ALNS/HGS.
- **Environment Layer (`logic/src/envs/`)**: RL4CO-compatible `VRPPEnv` and `WCVRPEnv` with diverse generators.
- **Data Layer**: Optimized `TensorDict`-based datasets and generators in `logic/src/data/`.

### Changed
- Refactored `logic/src/pipeline/reinforcement_learning/` into the new `logic/src/pipeline/rl/` package.
- Standardized all policy inputs/outputs to `TensorDict`.
- Unified training loop via `WSTrainer` (Lightning Wrapper).
- Updated `alns.py` and `hgs.py` to support `pd.DataFrame` coordinate inputs for distance calculation.

### Removed
- Legacy training scripts (`manager_train.py`, `worker_train.py`) in favor of `main.py` entry points.

## [Unreleased]

### Added
- `logic/src/dehb/` directory containing extracted DEHB library (internalized).
- `TESTING.md`: Comprehensive documentation on the project's testing strategy and organization.
- `DEPENDENCIES.md`: Detailed policy on dependency management and security.
- `logic/src/py.typed`: PEP 561 marker file for type hints support.

### Changed
- Moved DEHB implementation from `logic/src/pipeline/reinforcement_learning/hyperparameter_optimization/` to `logic/src/dehb/`.
- Updated all import statements in `hpo.py` and test files to reflect DEHB move.
- Updated `.gitignore` to exclude `.pytest_cache`, `.mypy_cache`, and `.ruff_cache`.
- Increased code coverage threshold to 60% with enforcement in `pyproject.toml`.

### Fixed
- Cleaned up 500+ `__pycache__` directories from the repository.
- Resolved DEHB import errors in the test suite by updating `@patch` decorators and import paths.
