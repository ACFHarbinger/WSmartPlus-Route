---
description: Add a new VRP problem environment (state, physics, and generator) under logic/src/envs/.
---

You are an Operations Research engineer adding a new vehicle routing problem variant to WSmart+ Route.

## Existing Environments (Reference)

| File | Problem |
|------|---------|
| `logic/src/envs/vrpp.py` | Vehicle Routing Problem with Profits |
| `logic/src/envs/wcvrp.py` | Capacitated Waste Collection VRP |
| `logic/src/envs/swcvrp.py` | Stochastic Waste Collection VRP |
| `logic/src/envs/tsp.py` | Travelling Salesman Problem |

## Implementation Steps

### 1. Read the Base Class
Read `logic/src/envs/base/` to identify required abstract methods (state init, step, reset, observation space, etc.).

### 2. State Design (CRITICAL — read AGENTS.md §6.1)
The state file encodes all environment physics. Rules:
- State must be **fully serializable** (no live Gurobi model references).
- `step()` must be a **pure function** — same state + action → same next state.
- Track: current node, visited mask, remaining capacity, current route length, total profit.

```python
# Typical state fields
@dataclass
class NewEnvState:
    coords: torch.Tensor          # (B, N+1, 2)  depot at index 0
    demand: torch.Tensor          # (B, N)
    capacity: torch.Tensor        # (B,)
    visited: torch.BoolTensor     # (B, N)
    current_node: torch.LongTensor  # (B,)
    route_length: torch.Tensor    # (B,)
    done: torch.BoolTensor        # (B,)
```

### 3. Data Generator
Add a generator in `logic/src/envs/generators/` following the existing pattern.
Generator must produce instances matching the format in AGENTS.md §12.1:
```python
{
    'loc': torch.Tensor,    # (batch, n_nodes, 2)
    'depot': torch.Tensor,  # (batch, 2)
    'capacity': float,
    'max_length': float,
}
```

### 4. Register the Environment
Register in `logic/src/envs/problems.py` so the CLI and configs can reference it by name.

### 5. Testing (MANDATORY before state merges)
```bash
# Must pass before committing any state_*.py equivalent
python main.py test_suite --module test_problems
```

Write tests in `logic/test/unit/envs/test_<env_name>.py` covering:
- Valid step transitions (capacity decreases, visited mask updates).
- Episode termination (depot return, capacity exceeded, max length).
- Edge cases: single node, all nodes infeasible, zero demand.
- Batch consistency: outputs have correct shape `(B, ...)`.

## Guardrails
- **Never** modify existing `state_*.py` files without running `test_problems` first.
- Stochastic environments (SCWCVRP-style) must set a seed for reproducibility in tests.
- Masking (`boolmask.py`) must prevent selection of visited nodes and capacity-violating nodes.
