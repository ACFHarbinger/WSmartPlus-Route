# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Created `CONTRIBUTING.md` with setup and style guidelines.
- Added `.pre-commit-config.yaml` for automated code quality checks.
- Created `justfile` for command automation (sync, test, lint, format, run, gui).
- Added GitHub Actions CI pipeline (`ci.yml`).
- Added `dependabot.yml` for automated dependency updates.
- Added `ARCHITECTURE.md` documenting system design and patterns.
- Integrated `check_docstrings.py` into `justfile` and `pre-commit` hooks.
- Audited and cleaned up `pyproject.toml` dependencies.
- Integrated `loguru` for structured system logging in the simulation pipeline.
- Added `tui` command to `main.py` enabling an interactive Terminal UI for configuration and execution.
- Added `[tool.ruff]` configuration to `pyproject.toml`.
- Added validation logic to TUI forms for immediate feedback.
- Added `HGSSolver` class to encapsulate HGS logic.
- Modularized ALNS into auxiliary files in `logic/src/policies/alns_aux/`.
- Modularized HGS into auxiliary files in `logic/src/policies/hgs_aux/`.
- Restored algorithm dispatch logic in `run_alns` and `run_hgs` to support multiple engines.

### Changed
- Refactored `run_day` in `logic/src/pipeline/simulator/day.py` to use `SimulationDayContext`.
- Updated `RunningState.handle` in `logic/src/pipeline/simulator/states.py` to support `SimulationDayContext`.
- Added comprehensive type hints and docstrings to `day.py`, `states.py`, `context.py`, `adaptive_large_neighborhood_search.py`, and `hybrid_genetic_search.py`.
- Strengthened `test_simulator.py` with `get_daily_results` edge cases and updated `test_policies.py` for new simulation signatures.
- Standardized return signatures of `ALNSSolver` and `HGSSolver` to `(routes, profit, cost)`.
- Updated ALNS adapters (`run_alns`, `run_alns_package`, `run_alns_ortools`) to return profit.
- Refactored `hybrid_genetic_search.py` and `adaptive_large_neighborhood_search.py` to use modular auxiliary components.
- Updated `AttentionDecoder` to support `scwcvrp` and `sdwcvrp` problems.
- Modified `StateSDWCVRP` to handle partial deliveries and standardized 3D tensor dimensions.
- Enhanced `WCContextEmbedder` and `VRPPContextEmbedder` with robust waste key handling.

### Fixed
- Resolved `ImportError` for `GridBase` by repairing the `wsmart_bin_analysis` git submodule.
- Fixed `TestDay` and `TestSimulation` mocks in `test_simulator.py`.
- Resolved policy name parsing collisions in `SimulationDayContext` test helper.
- Fixed `IndexError` in policy tests by ensuring full spatial data in mock fixtures.
- Fixed critical bug in `HGS.LinearSplit._split_limited` where limited fleet solutions returned empty routes.
- Fixed return value unpacking in `policy_lookahead_alns`.
- Resolved multiple `ImportError` issues across CLI (`logic/src/cli/__init__.py`) and GUI layers.
- Fixed circular import between `MainWindow` and `SimulationResultsWindow`.
- Corrected `ValueError` in simulation logging for single-sample parallel runs in `logic/src/pipeline/test.py`.
- Fixed `IndexError` and indexing logic in `StateSDWCVRP` update and masking.
- Fixed JSON serialization of `torch.device` objects in `logic/src/pipeline/train.py`.
- Corrected `SyntaxError` in `logic/test/__init__.py`.
