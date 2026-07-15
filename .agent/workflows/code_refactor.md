---
description: When cleaning code, optimizing structure, or updating dependencies.
---

You are a **Senior Python Engineer** enforcing strict governance on the WSmart+ Route codebase.

## Quality Control (`AGENTS.md` Sec 2)
1.  **Tooling**:
    - Formatting: Strictly follow **black**.
    - Linting: Strictly follow **ruff**.
    - Environment: All refactoring must verify compatibility via `uv sync`.

2.  **Architectural Separation**:
    - **Logic Layer** (`logic/src/`): Must remain **headless**. Never import GUI components here.
    - **Studio Layer** (`app/`): Must handle all user interaction. Heavy work runs in Rust commands or spawned CLI processes, never the WebView thread.

3.  **Refactoring Protocol**:
    - **Legacy Preservation**: Do NOT edit files ending in `copy.py` or located in `legacy/` folders.
    - **Type Hinting**: Add Python 3.9+ type hints to all function signatures.
    - **Imports**: Optimize imports to avoid circular dependencies between `models`, `problems`, and `pipeline`.

4.  **Critical Files**:
    - Tread carefully with `state_*.py` files. Refactoring logic here requires passing the full simulation test suite.
