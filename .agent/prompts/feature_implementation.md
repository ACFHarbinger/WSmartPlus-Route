# Feature Implementation Prompt

**Intent:** Add a new heuristic policy using a Few-Shot/Template pattern.

## The Prompt

Act as a Senior Operations Research Engineer. I need to implement a new heuristic policy called `PolicyGreedyService` in `logic/src/policies/greedy_service.py`.

**Task:** Write the Python code for this policy. It should:
1. Prioritize visiting nodes with the highest reward / distance ratio.
2. Adhere to the Policy interface pattern used in logic/src/policies/regular.py or last_minute.py.
3. Include type hints and docstrings.

**Output Format:** `logic/src/policies/greedy_service.py`

```python
# Imports (Reference logic.src.tasks and utils)

def policy_greedy_service(state, ...):
    """
    [Docstring explaining the logic]
    """
    # Implementation
```

**Safety Check:** Ensure you import `boolmask` from `logic/src/utils/boolmask.py` to mask invalid nodes (visited or exceeding capacity) before selection.
