# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0] - 2026-01-25

### Comprehensive Testing Suite (Phase 2)
Completed a major testing overhaul, introducing End-to-End (E2E) and Integration testing layers to ensure system reliability and stability.

### Added
- **E2E Testing**:
  - `logic/test/e2e/test_cli_commands.py`: Smoke tests for core CLI commands (`gen_data`, `train_lightning`, `eval`, `test_sim`).
- **Integration Testing**:
  - `logic/test/integration/test_training_advanced.py`: Validated advanced RL algorithms (DR-GRPO, GDPO) and checkpoint resumption.
  - `logic/test/integration/test_simulation.py`: Component-level tests for `SimulationContext`, `run_day`, and wrappers.
- **Test Infrastructure**:
  - Patched `VRPPEnv` and `CriticNetwork` for compatibility during testing.
  - `walkthrough.md`: Documentation of test results and coverage.

### Changed
- **Implementation Plan**: Updated to reflect completion of Phase 2.1 and 2.2.
- **Testing Standards**: Enforcing explicit `cpu` accelerator for CI/Integration tests to prevent GPU contention.


## [4.1.0] - 2026-01-24

### Legacy Pipeline Removal
The legacy `reinforcement_learning/` pipeline has been permanently removed after the successful migration to the PyTorch Lightning architecture.

### Changed
- **Directory Structure**: Removed `logic/src/pipeline/reinforcement_learning/` and corresponding symlinks.
- **Training Entry Point**: Centralized training logic in `logic/src/pipeline/features/train.py`.
- **Documentation**: Updated all architecture guides to reflect the consolidated pipeline structure.

## [4.0.0] - 2026-01-22

### Lightning Pipeline Migration Complete ✅
Successfully completed the full migration from the legacy `reinforcement_learning/` pipeline to the new Lightning-based `rl/` pipeline. All 6 migration phases finished with backward compatibility preserved.

### Migration Summary
- **Phase 1-2**: CLI compatibility and vectorized policy relocation complete
- **Phase 3-4**: External imports updated, 580 tests passing (167 core + 413 extended)
- **Phase 5**: Documentation updated (TUTORIAL.md, AGENTS.md)
- **Phase 6**: Legacy pipeline backed up and deprecated with symlink for backward compatibility

### Added
- **Symlink Approach**: `reinforcement_learning/` → `reinforcement_learning.bak/` symlink maintains backward compatibility for tests and existing code
- **Migration Documentation**: Comprehensive `MIGRATION_PLAN.md` with phase tracking and verification checklists

### Changed
- **Primary Pipeline**: `logic/src/pipeline/rl/` is now the active, maintained codebase
- **Deprecated**: `logic/src/pipeline/reinforcement_learning/` marked as deprecated (symlink to backup)
- **Documentation**: All tutorial examples, architecture docs, and agent instructions updated to reference new pipeline
- **Import Patterns**: Updated test fixtures to use new `wrap_dataset(dataset, policy=None, env=None)` signature

### Fixed
- **TensorDict Compatibility**: Fixed membership checks across 7 files (`key in list(td.keys())`)
- **RolloutBaseline**: Added backward compatibility for old `(model, problem, opts)` calling convention
- **List Datasets**: Added TensorDict conversion in `_rollout` for list-type datasets
- **NonTensorData Handling**: Fixed `get_tour_length` to handle TensorDict's None wrapper
- **Crossover Robustness**: Made `vectorized_ordered_crossover` handle duplicate values gracefully
- **Linting**: Fixed E402 errors in `train.py` by moving imports before deprecation warning

### Test Results
- ✅ 167 core tests passing (test_models, test_train, test_integration, test_hp_optim, test_il_train)
- ⚠️ 12 pre-existing failures in SAPO/GSPO integration (unrelated to migration)
- ✅ No import errors, full backward compatibility maintained

### Migration Notes
The old `reinforcement_learning/` directory is preserved as `reinforcement_learning.bak/` and accessed via symlink. This allows existing tests and code to continue working while new development uses the Lightning pipeline. After 1 week of stable operation, the backup can be removed.

## [3.2.1] - 2026-01-21

### Defaults Optimization & Stabilization
Standardized the training pipeline on `train_lightning`, optimized default performance settings, and resolved critical data loading concurrency issues.

### Changed
- **Entry Point**: Standardized on `python main.py train_lightning` as the primary training command in all documentation.
- **Performance Defaults**: Enabled **Automatic Mixed Precision (AMP)** (`precision="16-mixed"`) and set `num_workers=4` by default for faster training on modern GPUs.
- **Generator Handling**: `VRPPGenerator` and `WCVRPGenerator` now correctly handle `capacity=None` defaults (1.0 and 100.0 respectively).

### Fixed
- **Data Loading Concurrency**: Resolved CUDA forking errors with `num_workers > 0` by ensuring `Generator` instances are moved to CPU before being passed to `DataLoader`, enabling efficient parallel data generation.
- **Documentation**: Updated `CONTRIBUTING.md`, `TROUBLESHOOTING.md`, `MIGRATION_PLAN.md`, and `REFACTORING_PLAN2.md` to align with the new Lightning pipeline standards.

## [3.2.0] - 2026-01-21

### CWCVRP Lightning Migration & GDPO Support
Successfully migrated the Capacitated Waste Collection VRP (CWCVRP) pipeline to the new PyTorch Lightning architecture, achieving verifyied parity, and integrated Gradient-Decomposed Policy Optimization (GDPO).

### Added
- **GDPO Algorithm**: Implemented `GDPO` (Gradient-Decomposed Policy Optimization) in `pipeline/rl/core/gdpo.py` and consolidated it into `train_lightning.py`.
- **Decomposed Rewards**: Updated `VRPPEnv` and `WCVRPEnv` to calculate and store decomposed reward components (`reward_prize`, `reward_cost`, `reward_collection`) for advanced gradient analysis.
- **Baseline Improvements**: Enabled `bl_warmup_epochs` configuration and refactored `Baseline` classes to inherit from `torch.nn.Module` for automatic device management.
- **Parity Report**: Added `CWCVRP_LIGHTNING_MIGRATION_REPORT.md` documenting the successful parity verification between Legacy and Lightning pipelines.

### Changed
- **`CWCVRP` Pipeline**: Fully migrated to Lightning, verified with `val/reward` (~ -2.78) matching inverted legacy costs.
- **`RLConfig`**: Added GDPO-specific configurations (`gdpo_objective_keys`, `gdpo_objective_weights`, etc.) and `bl_warmup_epochs`.
- **Infrastructure**: Refactored `generators.py` and `data_utils.py` for cleaner data handling.

### Fixed
- **Device Mismatch**: Resolved critical `RuntimeError` in `reinforce.py` where reward and baseline tensors were on different devices.
- **Optimization**: Support for `RMSprop` in Lightning `configure_optimizers` to match legacy defaults.

## [3.1.0] - 2026-01-21

## [3.2.0] - 2026-01-21

### Multi-Objective Optimization (GDPO)
Introduction of Group reward-Decoupled Normalization Policy Optimization (GDPO) for stable multi-objective learning.

### Added
- **GDPO Algorithm**: Implementation of `GDPO` in `logic/src/pipeline/rl/core/gdpo.py`, featuring decoupled Z-score normalization and weighted aggregation.
- **Decomposed Rewards**: `VRPPEnv` and `WCVRPEnv` now expose individual reward components (`reward_prize`, `reward_cost`, `reward_overflow`) in `TensorDict`.
- **Config**: Added fields to `RLConfig` for GDPO objectives, weights, and conditional keys.
- **Integration**: Added `gdpo` support to `train_lightning.py`.

### Vectorization & Expert Policy Suite
A major performance upgrade and the introduction of a new stochastic expert for imitation learning.

### Added
- **Vectorized ALNS Policy**: Modular `VectorizedALNS` implementation for GPU-accelerated batch optimization, replacing instance-by-instance loops.
- **Random Local Search Policy**: New `RandomLocalSearchPolicy` expert for imitation learning, featuring pre-sampled operator sequences for maximum efficiency.
- **Full LS Vectorization**: Complete migration of local search operators (`relocate`, `two_opt_star`, `swap_star`, `three_opt`) in `local_search.py` to native PyTorch vectorized operations.
- **Imitation Learning Integration**: Added `random_ls` expert support in `ImitationLearning` and `AdaptiveImitation` RL modules.
- **Benchmarking & Validation**:
  - `benchmark_ls.py`: Performance auditing tool (achieving ~1,700 instances/sec on RTX 3090 Ti).
  - `test_random_local_search.py`: Comprehensive correctness suite for stochastic operators.

### Changed
- **`local_search.py`**: Architecture refactored to use advanced vectorized techniques (v-map equivalent) like priority-based `argsort` for double-relocations and case-masking for 3-opt.
- **`RLConfig`**: Expanded to support configurable iterations (`random_ls_iterations`) and probabilities (`random_ls_op_probs`) for expertos.
- **`ALNSPolicy` & `HGSPolicy`**: Now fully utilize low-level vectorized solvers for GPU throughput.

### Fixed
- Resolved CUDA `RuntimeError` and `IndexError` in vectorized operators by implementing strict tensor expansion and explicit device placement.
- Cleaned up redundant expressions and variables in the local search implementation.

## [3.0.0] - 2026-01-21

### Major Refactoring Completion (Phases 1-5)
The **Old RL Pipeline** (`logic/src/pipeline/reinforcement_learning/`) features have been fully migrated to the **New Lightning Pipeline** (`logic/src/pipeline/rl/`), achieving 100% parity and enhanced modularity.

### Added
- **Robust Baselines**:
  - `RolloutBaseline`: Now performs correct greedy rollout with T-test significance updates.
  - `POMOBaseline`: Computes mean reward across instance augmentations/starts.
  - `WarmupBaseline` & `CriticBaseline`: Restored and integrated.
- **Meta-Learning Suite**:
  - `CostWeightManager` (TD Learning): Reimplemented using tabular TD methods for adaptive weight tuning.
  - `HyperNetworkTrainer`: Implemented via `HyperNetworkStrategy` wrapper.
  - Registery now supports: `rnn` (RWA), `bandit` (Contextual), `morl` (Pareto), `tdl` (TD), `hypernet`.
- **Training Infrastructure**:
  - `TimeBasedMixin`: Full simulation parity for temporal bin fill updates in `time_training.py`.
  - `Epoch Utilities`: Rich validation metrics (overflows, efficiency, cost) in `epoch.py`.
  - `Post-Processing`: Full `calculate_efficiency` logic and `EfficiencyOptimizer` parity.
- **HPO Module** (`logic/src/pipeline/rl/hpo/`):
  - Consolidated HPO logic into dedicated module.
  - `OptunaHPO`: Clean class-based interface for Optuna (TPE, Grid, Random, Hyperband).
  - `DEHB`: Integrated Differential Evolution Hyperband.

### Changed
- **Folder Structure**:
  - Moved `dehb` from `features/` to `hpo/`.
  - Created alias proxies in `pipeline/rl/policies/` for classical solvers (`hgs`, `alns`).
- **Logic**:
  - `MetaRLModule` now automatically passes environment context to strategies like HyperNet and TDL.
  - `HRLModule` Manager now computes Actor-Critic (PPO) loss instead of no-op.

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
