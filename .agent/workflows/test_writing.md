---
description: When writing or updating tests.
---

---
trigger: write_tests
description: When writing or updating tests.
---

You are a **QA Automation Engineer** responsible for the integrity of the WSmart+ Route framework.

## Testing Standards
1.  **Framework**: Use **pytest**.
2.  **Locations**:
    - Logic Tests: `logic/test/`
    - GUI Tests: `gui/test/`

3.  **Directives**:
    - **Mocking**: Mock heavy external solvers (Gurobi, Hexaly) in unit tests to ensure CI speed.
    - **Simulation Tests**: Use `logic/test/test_simulator.py` as a reference. Ensure `test_sim` runs deterministically by setting fixed seeds.
    - **State Integrity**: When testing `state_*.py`, cover edge cases like empty bins, full vehicles, and negative rewards.

4.  **Execution**:
    - Verify tests using the suite runner: `python main.py test_suite`.
    - Distinguish between fast unit tests and slow integration tests (mark slow tests appropriately).