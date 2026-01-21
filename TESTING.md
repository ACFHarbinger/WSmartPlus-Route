# WSmart-Route Testing Strategy

## Overview
WSmart-Route utilizes `pytest` for unit, integration, and simulation testing.
We enforce a minimum code coverage of 60% (target: 80%).

## Running Tests
Run all tests:
```bash
uv run pytest
```

Run with coverage:
```bash
uv run pytest --cov
```

## Test Organization
- `logic/test/`: Logic layer tests.
    - `fixtures/`: Shared test data and mocks.
- `gui/test/`: GUI layer tests.

## Continuous Integration
Tests are automatically run on every push and pull request to `main` via GitHub Actions.
