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

### Changed
- Refactored `run_day` in `logic/src/pipeline/simulator/day.py` to use `SimulationDayContext`.
- Updated `RunningState.handle` in `logic/src/pipeline/simulator/states.py` to support `SimulationDayContext`.
- Added comprehensive type hints and docstrings to `day.py`, `states.py`, `context.py`, `adaptive_large_neighborhood_search.py`, and `hybrid_genetic_search.py`.
- Strengthened `test_simulator.py` with `get_daily_results` edge cases and updated `test_policies.py` for new simulation signatures.
- Standardized return signatures of `ALNSSolver` and `HGSSolver` to `(routes, profit, cost)`.
- Updated ALNS adapters (`run_alns`, `run_alns_package`, `run_alns_ortools`) to return profit.
- Refactored `hybrid_genetic_search.py` to remove legacy functions and improve type safety.

### Fixed
- Resolved `ImportError` for `GridBase` by repairing the `wsmart_bin_analysis` git submodule.
- Fixed `TestDay` and `TestSimulation` mocks in `test_simulator.py`.
- Resolved policy name parsing collisions in `SimulationDayContext` test helper.
- Fixed `IndexError` in policy tests by ensuring full spatial data in mock fixtures.
- Fixed critical bug in `HGS.LinearSplit._split_limited` where limited fleet solutions returned empty routes.
- Fixed return value unpacking in `policy_lookahead_alns`.
