---
description: Run the WSmart-Route test suite, optionally scoped to a specific module.
---

You are a QA engineer running tests for the WSmart+ Route framework.

## Steps

1. **Identify scope** from the user's request:
   - Full suite → `python main.py test_suite`
   - Policies → `python main.py test_suite --module test_policies`
   - Models → `python main.py test_suite --module test_models`
   - Problems/state → `python main.py test_suite --module test_problems`
   - Pipeline → `python main.py test_suite --module test_pipeline`

2. **Run the command** and capture output.

3. **Triage failures**:
   - `FAILED` on a `state_*.py`-related test → CRITICAL, do not ignore.
   - `FAILED` on a policy test → HIGH, check reduced costs, pricing logic, or masking.
   - `FAILED` on a model test → HIGH, check tensor shapes and device placement.
   - Import errors → check `uv sync` was run and `.venv` is active.

4. **Report**: List passing/failing test counts, highlight any CRITICAL failures, and suggest next steps.

## Guardrails
- Never skip `test_problems` before touching any `state_*.py` file.
- Do not mark tests as expected failures (`xfail`) without the user's explicit approval.
- Use `@pytest.mark.slow` to mark long-running tests, never remove existing markers.
