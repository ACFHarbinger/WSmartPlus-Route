# Brief — Kimi (lane D: exact solvers, hyper-heuristic, neural agent, policy base)

Read `.agent/tasks/minimal-export-review-common.md` first. Bus: `.agent/bus/2026-09-25.md`.

## Lane D scope

- `logic/src/policies/route_construction/base/` — `base_routing_policy.py`, `base_multi_period_policy.py`, `factory.py`, `registry.py`.
- `logic/src/policies/route_construction/exact_and_decomposition_solvers/branch_and_price_and_cut/` (BPC) and `smart_waste_collection_two_commodity_flow/` (SWC-TCF).
- `logic/src/policies/route_construction/hyper_heuristics/ant_colony_optimization_hyper_heuristic/` (ACO-HH) and its shared dependency `meta_heuristics/ant_colony_optimization_k_sparse/pheromones.py`.
- `logic/src/policies/route_construction/learning_algorithms/neural_agent/` (NA: `policy_na.py`, `agent.py`, `simulation.py`, `batch.py`, `params.py`).
- `logic/src/policies/route_construction/other_algorithms/travelling_salesman_problem/` (`tsp.py`, `two_opt.py` — shared helpers).
- `logic/src/policies/helpers/solvers_and_matheuristics/` (30 files, 11.7 kLOC: column generation, pricing/RCSPP, cutting planes, smoothing, branching, MILP builders).

Flowcharts of the intended control flow (built from source on 2026-09-24) are in
`assets/diagrams/code/{bpc,swc_tcf,aco_hh,pg_clns}_flowchart.dot` on the
`feat/paper-results-and-website` branch (untracked copies may exist under
`assets/diagrams/code/` in the main checkout). Use them to spot divergence
between intent and code.

## Questions this lane must answer

1. **Base policy contract.** `base_routing_policy.py::execute` builds the sub-problem (mandatory bins, `vrpp` flag), maps local ids to global ids, computes km on the full matrix, applies the improver. Check the id mapping (depot 0, `+1` offsets), the `vrpp` short-circuit ("mandatory empty and not vrpp → `[0, 0]`"), and the `_log_solver_params`/`run = None` remnants left by the tracking removal (dead code → removal rows).
2. **BPC.** `column_generation.py`, `pricing/solver.py`, `cutting_planes.py`, `smoothing.py`, `branching/*`: label-correcting DP correctness (dominance, ng-cycles, resource bounds), Lagrangian bound pruning (`BPCPruningException`), timeouts (`rcspp_timeout`, `time_limit`) actually enforced, `exact_mode: false` semantics, and what happens on Gurobi infeasibility (empty tour vs exception). Which branching strategies/cut families in `policy_bpc.yaml` are wired vs silently ignored?
3. **SWC-TCF.** `gurobi.py` constraints (1)–(7) vs the MILP in the flowchart; OR-Tools fallback path (`ortools_wrapper.py`) — reachable and correct when `framework: ortools`? `delta`, `psi`, `Omega` units; `dist_matrix` 6,000 km arc filter.
4. **ACO-HH.** `hyper_aco.py`/`hyper_operators.py`: operator vertices vs the `operators` list in the yaml (five listed, eleven implemented — are the six unlisted ones dead?), pheromone/visibility updates, penalty halving, `construct()` fallback. Which `helpers/operators/*` files does it really import?
5. **Neural Agent.** `agent.py`/`simulation.py`/`batch.py`: model loading through `utils/model/loader.py`, mandatory mask injection, decoding strategy from yaml (`beam_width: 5` with `strategy: greedy` — used or ignored?), `route_improvement: []` handling, the `_viz_record` `hasattr` guards, and the `lookahead` selector wiring via `vector/selection`.
6. **helpers/solvers_and_matheuristics.** Per file: imported by BPC/SWC-TCF (or by any retained policy) or only by removed matheuristics? Removal rows with LOC; note anything BPC imports lazily inside functions.

Post claims and findings on the bus under `### Kimi — 2026-09-25 (...)`.
