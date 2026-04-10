---
description: Diagnose and fix issues in the Branch-and-Price-and-Cut (BPC) solver.
---

You are an expert in exact combinatorial optimization debugging the BPC solver at `logic/src/policies/branch_and_price_and_cut/`.

## Key Files

| File | Role |
|------|------|
| `bpc_engine.py` | Main B&B orchestrator |
| `master_problem.py` | LP/IP master (Gurobi) |
| `rcspp_dp.py` | Pricing subproblem (RCSPP via DP) |
| `branching.py` | Branching strategy |
| `cutting_planes.py` | Cut generation |
| `separation.py` | Cut separation routines |
| `params.py` | `BPCParams` configuration dataclass |

## Diagnostic Checklist

### LP Infeasibility at Root Node
- Symptom: `GurobiError: Model is infeasible` early in solve.
- Check: Phase I Farkas pricing must run before Phase II. Verify `master_problem.py` handles the infeasible-LP case by minimizing infeasibility (Farkas objective) first.
- Fix: Ensure the master switches to Phase II only after feasibility is confirmed.

### Negative Reduced Cost Columns Not Found (Stalling)
- Symptom: CG loop terminates with LP bound not improving.
- Check: `rcspp_dp.py` — verify `reduced_cost = dual_price - sum(node_duals)` uses current dual values passed from master.
- Check: `rc_tolerance` in `BPCParams` — too tight a tolerance may miss valid columns.

### Bound Not Converging (Optimality Gap Stuck)
- Symptom: `z_UB - z_LB` not decreasing across B&B nodes.
- Check: Lagrangian bound `z_UB` is computed only after local CG convergence, NOT mid-CG.
- Check: `GlobalCutPool` — cuts must be re-added at every descendant node. Verify `cutting_planes.py` re-injects archived cuts on node entry.

### Incorrect Branching
- Symptom: Fractional solutions at leaf nodes.
- Check: `branching.py` — confirm branching on arc flow variables, not just node visits.
- Check: Strong branching candidate list (`strong_branching_size` in `BPCParams`).

### Cut Redundancy / Slow Separation
- Symptom: Many cuts added but bound barely improves.
- Check: `cut_orthogonality_threshold` in `BPCParams` — increase to filter near-duplicate cuts.
- Check: `enable_heuristic_rcc_separation` — heuristic RCC separation may add weak cuts; try disabling.

## Debugging Commands

```bash
# Run BPC-specific unit tests
python main.py test_suite --module test_policies

# Run with verbose Gurobi output (set in BPCParams)
# params.exact_mode = True disables dual smoothing for cleaner duals

# Check a single instance
python -c "
from logic.src.policies.branch_and_price_and_cut.bpc_engine import BPCEngine
from logic.src.policies.branch_and_price_and_cut.params import BPCParams
params = BPCParams(time_limit=30, exact_mode=True)
engine = BPCEngine(params=params)
# load instance and call engine.solve(instance)
"
```

## Guardrails (AGENTS.md §6.1)
- NEVER compute `z_UB` before CG convergence at a node.
- NEVER rank columns by raw profit — only `reduced_cost` improvements are valid.
- Phase I (Farkas) MUST precede Phase II (normal pricing) at every node.
- All cuts MUST be archived in `GlobalCutPool` and re-injected at descendant nodes.
