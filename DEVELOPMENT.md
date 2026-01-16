# WSmart-Route Developer Guide

Welcome to the WSmart-Route development documentation. This guide covers setup, testing, and contribution workflows.

## Environment Setup

The project uses `uv` for dependency management and `python 3.9+`.

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync Dependencies**:
   ```bash
   uv sync
   source .venv/bin/activate
   ```

## Development Workflow

We use `just` as a command runner. Ensure you have `just` installed (or use the commands in `justfile` manually).

### Common Commands

| Command | Description |
|---|---|
| `just check-docs` | Verify all modules have docstrings. |
| `just test` | Run the full test suite. |
| `just test-unit` | Run unit tests only (faster). |
| `just lint` | Run code linting (Ruff). |
| `just format` | Format code (Black/Ruff). |
| `just build-docs` | Build Sphinx documentation. |

### CLI Usage

The project features a modular CLI with a Text User Interface (TUI).

- **Launch TUI**:
  ```bash
  python main.py tui
  ```

- **Run Command Directly**:
  ```bash
  python main.py train --model am --problem vrpp
  python main.py test_sim --policies regular alns --days 31
  ```

## Documentation

Documentation is built using Sphinx.

- **Build Docs**:
  ```bash
  cd logic/docs
  make html
  ```
- **View Docs**: Open `logic/docs/build/html/index.html` in your browser.

## Project Structure

- `logic/src`: Core application logic (Models, Simulator, Pipeline).
- `gui/src`: PySide6 application code.
- `main.py`: Entry point.
- `scripts/`: Helper scripts.

## Contribution Guidelines

1. Ensure all code is documented (docstrings).
2. Run `just lint` and `just test` before committing.
3. Use Type Hints where possible.
