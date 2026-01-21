# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
