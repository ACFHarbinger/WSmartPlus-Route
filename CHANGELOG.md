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

### Changed
- Refactored `run_day` in `logic/src/pipeline/simulator/day.py` to use `SimulationDayContext`.
- Updated `RunningState.handle` in `logic/src/pipeline/simulator/states.py` to support `SimulationDayContext`.
- Added comprehensive type hints and docstrings to `day.py`, `states.py`, and `context.py`.
- Strengthened `test_simulator.py` with `get_daily_results` edge cases.

### Fixed
- Resolved `ImportError` for `GridBase` by repairing the `wsmart_bin_analysis` git submodule.
- Fixed `TestDay` and `TestSimulation` mocks in `test_simulator.py`.
