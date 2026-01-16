# Contributing to WSmart-Route

Thank you for your interest in contributing to WSmart-Route! We welcome contributions from the community to help improve this project.

## Development Setup

This project uses `uv` for dependency management and environment handling.

1.  **Install `uv`**: Follow the instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).
2.  **Clone the repository**:
    ```bash
    git clone https://github.com/ACFPeacekeeper/WSmart-Route.git
    cd WSmart-Route
    ```
3.  **Sync environment**:
    ```bash
    uv sync
    ```

## Running Tasks

We use `just` as a command runner. If you don't have it installed, you can use the raw `uv` commands, but `just` simplifies the workflow.

- **Run Tests**:
    ```bash
    just test
    # Or: uv run pytest
    ```
- **Lint Code**:
    ```bash
    just lint
    # Or: uv run ruff check
    ```
- **Format Code**:
    ```bash
    just format
    # Or: uv run ruff format
    ```

## Code Style

We strictly enforce code style using `ruff`. Please ensure your code passes linting and formatting checks before submitting a Pull Request.

- **Linter**: `ruff`
- **Formatter**: `ruff format` (compatible with Black)

## Pull Request Process

1.  Fork the repository and create your branch from `main`.
2.  Ensure all tests pass and code is formatted.
3.  Submit a Pull Request with a clear description of your changes.

## License

By contributing, you agree that your contributions will be licensed under the project's license.
