# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Implemented LKH-3 inspired heuristic in `lin_kernighan.py` with alpha-measure candidate sets and penalty-aware (Lexicographical) optimization for CVRP.
- Implemented `vectorized_three_opt` in `local_search.py` using PyTorch for high-performance batch processing on GPU.
- Created `logic/test/test_vectorized_policies.py` providing comprehensive coverage for vectorized local search, HGS components, and the Split algorithm.
- Added `logic/test/fixtures/vectorized_policy_fixtures.py` for shared vectorized test data and registered it in `conftest.py`.

### Changed
- Integrated LKH-3 into `look_ahead.py` and updated policy dispatch to support the `lkh` policy identifier.
- Integrated `vectorized_three_opt` into the education phase of the `VectorizedHGS` solver in `hgs_vectorized.py`.
- Refactored `TestVectorizedPolicies` into modular classes (`TestVectorizedLocalSearch`, `TestVectorizedPolicies`, `TestVectorizedPopulation`) for better organization.
- Refactored `lin_kernighan.py` for improved readability, efficiency, and adherence to Python best practices.

### Fixed
- Fixed `NameError` in `look_ahead.py` by adding missing `get_multi_tour` import for VRP capacity handling.
- Fixed policy dispatch logic in `look_ahead.py` to correctly identify the `lkh` policy.
- Fixed `ImportError` in `hgs_vectorized.py` caused by a missing import for `vectorized_relocate`.

### Added
- Created `logic/src/utils/io/preview.py` by splitting `processing.py` to separate preview logic.
- Added lazy loading to `logic/src/policies/adapters.py` to prevent circular dependencies.
- Added missing docstrings to `adapters.py`, `processing.py`, `epoch.py`, and `dehb_base.py`.
- Created `logic/test/test_policies_aux_2.py` for extensive coverage of `move.py` and `swap.py`.
- Created `logic/test/test_visualize.py` to cover `visualize_utils.py` and `plot_utils.py`.
- Created `logic/test/test_eval_coverage.py` to test the evaluation pipeline.
- Created `logic/test/test_solutions.py` to cover `look_ahead_aux/solutions.py`.
- Added test fixtures for `file_system.py`.

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
- Splitted `logic/src/utils/io/processing.py` into `processing.py` and `preview.py` for better I/O modularity.
- Merged `interfaces.py` and `registry.py` into `logic/src/policies/adapters.py` to standardize policy architecture.
- Updated imports across `neural_agent.py`, `policy_vrpp.py`, `regular.py`, `look_ahead.py`, and `last_minute.py` to use new `adapters` module.
- Refactored `io_utils.py` to use a facade pattern for backward compatibility.
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
- Modified `logic/src/utils/io/splitting.py` to add zero-padding to chunk filenames.
- Updated `logic/test/test_io.py` to use correct nested dictionary structures.
- Increased code coverage threshold to 75% in `pyproject.toml`.

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
- Fixed `ValueError` in `test_file_system.py` by correcting mocked return values.
- Fixed infinite loop in `find_initial_solution` within `logic/src/policies/look_ahead_aux/solutions.py` by adding break conditions for all zones and checking for state stagnation.
- Fixed `UnboundLocalError` in `find_initial_solution` by properly initializing the `stop` variable in all zone loops.
- Resolved infinite loops in `move.py` and `swap.py` within test mocks by updating `test_policies_aux.py` to use `random.sample` for bin selection.
- Significantly improved code coverage for `file_system.py` (0% -> 51%) by adding tests for preview and statistics modes.
- Improved coverage for `move.py` and `swap.py` to >90%.
- Consolidated integration tests by merging `logic/test/test_policy_integration.py` into `logic/test/test_integration.py` and expanding coverage for simulation workflows.
