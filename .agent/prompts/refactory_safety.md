# Refactoring Safety Prompt

**Intent:** Safely modify core physics logic using the Constraint pattern.

## The Prompt

I need to modify the reward calculation in `logic/src/problems/vrpp/state_vrpp.py`.

**Current Goal:** Add a penalty for 'overtime' if the route length exceeds a certain threshold.

**Strict Constraints:**
1. You must identify which `step` function handles the state transition.
2. You must NOT break the masking logic (valid moves).
3. According to `AGENTS.md`, this is a CRITICAL severity file. You must list which tests in `logic/test/` need to be run to verify this change.

Provide the modified code snippet for the `step` function and the list of regression tests.
