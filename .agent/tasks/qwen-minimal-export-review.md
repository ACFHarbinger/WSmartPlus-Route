# Brief — Qwen (lane E: metaheuristics and the operator library)

Read `.agent/tasks/minimal-export-review-common.md` first. Bus: `.agent/bus/2026-09-25.md`.

## Lane E scope

- `logic/src/policies/route_construction/meta_heuristics/` — `adaptive_large_neighborhood_search` (ALNS), `hybrid_genetic_search` (HGS), `pheromone_guided_cooperative_large_neighborhood_search` (PG-CLNS, 35 files), `particle_swarm_optimization_memetic_algorithm` (PSOMA), `simulated_annealing_neighborhood_search` (SANS, 40 files incl. the legacy `og` engine).
- `logic/src/policies/helpers/local_search/` (`local_search_base.py`, `local_search_hgs.py`) and `logic/src/policies/helpers/operators/` (102 files, 25 kLOC — the single largest block left in the package).

Flowcharts built from source on 2026-09-24: `assets/diagrams/code/{alns,hgs,pg_clns,psoma,sans}_flowchart.dot` (on `feat/paper-results-and-website`; untracked copies may exist in the main checkout under `assets/diagrams/code/`).

## Questions this lane must answer

1. **Operator reachability (the big one).** Build the import graph from the five policy packages + `fast_tsp.py` + `local_search_hgs.py` into `helpers/operators/**` (static imports and the string-keyed registries/factories — grep for `register(`, `OPERATORS = {`, `importlib`). Produce the list of operator files that are *not* reachable and a removal row per family (destroy / repair / exchange / move / route / unstringing / ...), with LOC. State clearly which optional yaml flags (`use_cross_exchange`, `use_lambda_interchange`, `use_ejection_chains`, `use_3opt`, `extended_operators`, `profit_aware_operators`) pull in extra families, so the owner can decide whether to keep them.
2. **ALNS.** `alns.py`: segment weight update (`w ← (1−r)w + r·π/θ`, division by zero when `θ=0`), `q` bounds (`min(n,4)` vs `max(lower, min(100, ⌊ξn⌋))` when `n<4`), noise slot handling, `visited` hashing of routes, dynamic `T₀` when `best_profit ≤ 0`. `policy_alns.py` engine dispatch (`package`/`ortools` engines — reachable? removal rows if not).
3. **HGS.** `hgs.py`/`evolution.py`/`split.py`/`individual.py`: LinearSplit correctness for the profit objective (unlimited vs limited fleet), biased fitness (`nb_elite/|pop|`), penalty adaptation in VRPP mode, restart logic when `time_limit` is set, `pyvrp_wrapper.py` (engine `pyvrp` — reachable from the yaml? removal row if not), `dispatcher.py` trivial-instance path.
4. **PG-CLNS.** `pg_clns.py`, `aco.py`, `lns.py`, `local_search.py`, `operators/*`: pheromone τ₀ init, q₀ rule, global update with elitist weight, LNS operator weights/decay, SA acceptance, replacement rate rounding for `population_size=10`. Which of its 35 files are dead?
5. **PSOMA.** `solver.py`/`particle.py`: ROV decode, `_sa_search` surrogate rejection sign, EMA reward normalisation with `ε`, stagnation `L`, `metropolis_steps = n(n−1)` cost at `n=350`.
6. **SANS.** `dispatcher.py`, `heuristics/sans.py`, `anneal.py`, `sans_operators.py`, `refinement/*`, `search/random_search.py`, `common/solution_initialization.py`: the `new` engine's 18 operators, re-heat, `compute_profit` units (`V`, density, `PENALTY_MANDATORY_NODES_MISSED`), and the legacy `og` engine (`og_a`/`og_b` variants in the yaml): is `og` still reachable from `test_sim.yaml` (`sans: ${p.sans.sans.new}` only)? If not, propose removing the `og` engine (`find_solutions`, `refine_solution`, `rebalance_solution`, ~40 files) as a single removal row.

Post claims and findings on the bus under `### Qwen — 2026-09-25 (...)`.
