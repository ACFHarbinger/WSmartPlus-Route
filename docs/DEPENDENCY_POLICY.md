# Dependency Management Policy

This document outlines the standards for managing third-party dependencies in WSmart-Route.

## 1. Governance Principles

- **Lightweight Core**: The base installation should only include absolute requirements for the simulator and core neural logic.
- **Explicit Extras**: Features requiring specialized hardware (GPU) or commercial/large solvers should be sequestered into optional dependency groups.
- **Version Flexibility**: Prefer version ranges (e.g., `>=X.Y.Z`) over exact pins (`==X.Y.Z`) in the project manifest to avoid dependency hell.
- **Security First**: All dependencies must be audited for vulnerabilities before being added or updated.

## 2. Dependency Groups

Dependencies are categorized into the following groups:

### 2.1 Core Dependencies (`dependencies`)

Essential for the core framework, environment physics, and basic CLI/GU operations.

### 2.2 Optional Dependencies (`project.optional-dependencies`)

- `gpu`: Packages required for CUDA acceleration (e.g., `torch`-specific builds, `triton`).
- `solvers`: External optimization engines (e.g., `gurobipy`, `hexaly`, `ortools`, `pyvrp`, `vrpy`).
- `docs`: Tools for building the documentation (e.g., `sphinx`, `myst-nb`).

### 2.3 Development Dependencies (`dependency-groups.dev`)

Tools for testing, linting, formatting, and building the application (e.g., `pytest`, `ruff`, `mypy`).

## 3. Versioning Standards

- **New Dependencies**: Always use `>=` followed by the current stable version.
- **Pins**: Exact pins (`==`) are allowed only for `dev` tools where output consistency is critical (e.g., `ruff`, `mypy`) or when a bug in a specific version must be avoided.
- **Ranges**: Prefer `~=` or `==X.Y.*` if a dependency is known to break compatibility across minor or major versions.

## 4. Maintenance Workflow

### 4.1 Adding a Dependency

1. Verify the license (BSD, MIT, Apache 2.0 preferred).
2. Run `uv run pip-audit` to check for security issues.
3. Add to the appropriate group in `pyproject.toml`.
4. Run `uv sync` to update the lockfile.

### 4.2 Updating Dependencies

1. Periodically run `uv lock --upgrade` to check for updates.
2. Review changelogs for breaking changes.
3. Run the full test suite (`python main.py test_suite`) after any update.

### 4.3 Security Audits

- A security audit via `pip-audit` should be performed before every major release.
