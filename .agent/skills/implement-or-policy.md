---
description: Scaffold and implement a new Operations Research policy (heuristic or exact solver) in logic/src/policies/.
---

You are a Senior Operations Research Engineer implementing a new routing policy for WSmart+ Route.

## Discovery Phase

1. Read the base policy interface:
   - `logic/src/policies/base/` — identify the abstract base class and required methods.
   - `logic/src/interfaces/` — check for any Protocol definitions the policy must satisfy.

2. Read a reference implementation closest to the new policy type:
   - Exact solvers → read `logic/src/policies/branch_and_price_and_cut/`
   - Metaheuristics → read `logic/src/policies/hybrid_genetic_search/` or `logic/src/policies/adaptive_large_neighborhood_search/`
   - Constructive heuristics → read `logic/src/policies/genius/` or `logic/src/policies/cluster_first_route_second/`

## Implementation Checklist

- [ ] Create directory `logic/src/policies/<policy_name>/`
- [ ] Add `__init__.py` exposing the main policy class.
- [ ] Implement required abstract methods from the base class.
- [ ] Apply invalid-move masking using `logic/src/utils/functions/boolmask.py` for any node selection logic.
- [ ] Add a `params.py` with a `@dataclass` config class if the policy has hyperparameters.
- [ ] Use `get_device()` from `logic/src/utils/configs/setup_utils.py` — never hardcode `.cuda()`.
- [ ] Add type hints (`from typing import ...`) on all public methods.

## BPC-Specific Rules (if implementing exact solver)

Follow AGENTS.md §6.1 strictly:
1. Phase I Farkas pricing must resolve LP infeasibility before Phase II normal pricing.
2. Use `reduced_cost` improvements only — never rank columns by raw profit.
3. Compute Lagrangian bounds (`z_UB`) only after local CG convergence.
4. Archive all cuts centrally in `GlobalCutPool` and re-inject at descendant B&B nodes.

## Testing

After implementation, write unit tests in `logic/test/unit/policies/test_<policy_name>.py`:
- Test feasibility of returned solution (capacity, route length constraints).
- Test edge cases: single node, all nodes skipped, zero-profit nodes.
- Mock Gurobi/Hexaly calls in unit tests to keep CI fast.
- Run: `python main.py test_suite --module test_policies`
