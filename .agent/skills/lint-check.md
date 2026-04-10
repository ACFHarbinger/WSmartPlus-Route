---
description: Run ruff linting, ruff formatting, and mypy type checks on the codebase.
---

You are a code quality engineer enforcing WSmart+ Route coding standards.

## Steps

1. **Lint with ruff**:
   ```bash
   uv run ruff check .
   ```
   Fix all reported errors. Auto-fixable issues:
   ```bash
   uv run ruff check . --fix
   ```

2. **Format with ruff**:
   ```bash
   uv run ruff format .
   ```

3. **Type check with mypy** (non-blocking, advisory):
   ```bash
   uv run mypy . || true
   ```
   Flag `error` level issues for the user; treat `note` level as advisory.

4. **Report**: Summarize error counts per tool and list the files with issues.

## Standards to Enforce (from AGENTS.md §6.3)

- Imports: stdlib → third-party → local, each group separated by a blank line.
- Type hints: Use `from typing import List, Dict, Optional, Tuple` (NOT lowercase built-ins without `__future__`).
- No `nn.LayerNorm` — use `logic.src.models.modules.normalization.Normalization`.
- No hardcoded `.cuda()` — use `logic.src.utils.configs.setup_utils.get_device()`.
- Weight matrices: prefix `W_` (e.g. `W_query`, `W_key`).

## Guardrails
- Never add `# noqa` silencers without explaining why to the user.
- Ruff formatting is mandatory; mypy errors are advisory (report but do not block).
