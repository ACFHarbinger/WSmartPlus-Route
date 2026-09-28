# Kimi code-cleanup lane (#80 / #81 / #82) — handoff

**Base:** `main@6d500ed02`. Worktree: `~/.cache/wsr-review/kimi-code` (shared checkout
untouched). Two patches, applied **in this order**:

1. `.agent/cache/patches/kimi/issue-81-bpc-dead-code.patch` (1,925 lines) — pure
   deletions, reviewable as dead code only.
2. `.agent/cache/patches/kimi/issue-80-bpc-aco-swc-fixes.patch` (1,555 lines)
   — fixes + the remaining cleanups + the M-kimi-02/04 refactors.

Both `git apply --check` on a pristine `6d500ed02` clone (81 alone, then 80 on top);
the fully patched tree passes 163/163 affected unit tests and the BPC brute-force
harness (25/25 plain @ seed 11, 25/25 mandatory @ seed 7, 40/40 plain @ seed 42 run
in the worktree). 33 files, +448/−2,054.

## Packaging note (deviation from "one patch per issue", disclosed)

The B dead-option rows' sanctioned fix IS deletion (B-kimi-36/39/40/41: "delete"),
so deletions and fixes interleave in the same files. The split is by file type:
issue-81 is the pure-deletion file set; issue-80 carries every file that also
contains a behavioral/clarity fix, plus the M-kimi-02/04 refactors. Row→patch map:

- **issue-81 closes:** D-kimi-01 (dead `_select_nodes_knapsack` copy in the BPC
  engine — the live twin in the MS engine stays), D-kimi-02 + B-kimi-39 (DSSR
  wrapper, call site, `enable_dssr`/`dssr_max_iters`), D-kimi-03 + B-kimi-36
  (`GlobalCutPool.apply_to_master`, `_inject_multistar_cut`, `CutInfo`; the two
  engine docstrings and the constraints.py comment that claimed automatic
  re-injection now describe the shared-master reality), B-kimi-37/38 (the four
  inert engines — MinCut, TriangleClique, NodeProfitBound, PathElimination —
  classes, factory branches, "all"-composite wiring, supplemental list, and their
  four mock-only tests), B-kimi-40 + D-kimi (`reduced_cost_arc_fixing` function
  and its inert `_forbidden_arcs` engine block; the LIVE unconditional
  `_apply_reduced_cost_edge_fixing` is untouched), B-kimi-41 (Wentges machinery:
  `enable_dual_smoothing`, `_apply_dual_smoothing`, prev-dual state, recovery
  branches in both CG loops, UB-prune exclusion, Protocol attrs, yaml comment;
  the unreachable machinery is gone, not wired — wiring would change BPC
  numerics and needs its own owner decision), B-kimi-42, B-kimi-43, D-kimi-04
  (`has_artificial_variables_active` ×2, `Label.is_feasible`,
  `validate_tour`, `compute_tour_cost`, `_get_coords_from_model`, `BIG_M`),
  D-kimi-05 (legacy `SeparationEngine.separate` + the comb cluster
  `_separate_comb_heuristic`/`_grow_handle`/`_find_teeth_for_handle`/`_grow_tooth`/
  `_compute_comb_violation`/`USE_COMB_CUTS` — all test-only or legacy-only — and
  `test_separate_comb_heuristic`; the live `separate_integer/fractional` helpers
  stay), D-kimi-06 (`max_cut_iterations`, `use_spatial_partitioning`,
  `knapsack_proc_selection`, `enable_hybrid_search` — params, from_config,
  config dataclass, yaml).
- **issue-80 closes:** B-kimi-34 (tree reads the real `max_bb_nodes` /
  `search_strategy` / `branching_strategy`; both engines stop passing explicit
  args → per-run DeprecationWarning gone; regression test
  `test_bpc_tree_params.py` fails on old code with `best_first` ≠ `depth_first`),
  B-kimi-50 (the three ACO removal wrappers catch `(ValueError, IndexError,
  KeyError)` and log; programming errors propagate; regression test in
  `test_aco_hh_fixes.py`), B-kimi-51 (`math.ceil` sync count via
  `_elitism_sync_count`; regression test fails on old code), B-kimi-52 (4 stale
  docstrings), B-kimi-55 (MIP_GAP 1% applied to pyomo gurobi/scip/highs and
  best-effort OR-Tools string params), B-kimi-56 (native gurobi empty day now
  `[0, 0]` in both the no-incumbent and all-zero-solution paths), B-kimi-57
  (fleet fallback `n_bins` in all three wrappers), B-kimi-58 (arc-tracing
  extraction robust to fractional `k_var`; policy adapter splits the
  depot-delimited flat list into routes instead of stripping all zeros;
  `n_vehicles` from the context is now honoured — see behavior changes),
  B-kimi-59 (typed path flattens engine-nested `runtime_overrides` like the
  legacy path; regression test fails on old code with `KeyError: time_limit`).
- **issue-82 in issue-80's patch:** M-kimi-02 (shared
  `_route_extraction.extract_depot_delimited_route` consumed by all three SWC
  backends + cross-backend parity test: identical profit, same collected set on
  a fixed instance) and M-kimi-04 (the ten shadowed `MasterProblemSupport`
  stubs trimmed).

## Regression tests (all fail on the old code, pass on the new)

`logic/test/unit/policies/solvers/test_bpc_tree_params.py` (2),
`logic/test/unit/policies/solvers/test_swc_tcf_backends.py` (5: empty-day shape,
fleet fallback, depot-split, typed overrides, backend parity),
`logic/test/unit/policies/hyper_heuristics/test_aco_hh_fixes.py` (+2: ceil sync
count, wrapper exception semantics). Updated: `test_cutting_planes.py` (factory
names), `test_separation_engine.py` (legacy/comb test removed).

## Behavior changes (for the export README / #83)

1. **SWC-TCF fleet cap removed**: the adapter used to pass `number_vehicles=1`
   (the simulator's `n_vehicles=0` never reached it), silently capping every day
   at one vehicle. It now passes 0 = unbounded (n_bins fallback), matching BPC
   and the paper's "no fleet bound". Days whose load exceeds one payload can now
   produce multi-route plans; the route-splitting fix keeps their boundaries.
   Archived results were produced under the 1-vehicle cap.
2. **ACO-HH elitism** syncs `ceil(n_ants × ratio)` ants (diagram semantics);
   identical for the shipped 0.5/1.0 ratios, differs for fractional products
   (e.g. 0.35 × 10: 3 → 4).
3. **Native-gurobi SWC-TCF empty days** now return `[0, 0]` instead of `[0]`;
   the dispatcher's OR-Tools-GUROBI fallback check (`result[0] == [0, 0]`) now
   behaves as originally intended.
4. BPC, MS-BPC-SP: no behavior change (deletions were provably unreachable;
   alias fix makes the shipped DFS/limits apply as documented — brute-force
   exactness holds).

## Not done (with reasons)

- **M-kimi-01 (MS-BPC-SP helper copies)**: skipped per the brief's own gate.
  Plan: extend the shared helpers with the small fixes the 0.95+ copies gained
  (D1 pricing/branching fixes), then import the seven twins
  (`_perform_strong_branching`, `_select_nodes_knapsack` (the live MS twin),
  `_compute_lr_bound_at_node`, `_apply_branching_to_master`,
  `_extract_forced_sets_from_constraints`, `_reset_master_constraints`); keep
  the DIVERGED `_column_generation_loop`/Farkas/pricing twins untouched. Parity
  bar: MS engine results bit-identical on a fixed 5-node instance suite before
  merging each function, plus `bpc_bruteforce_check`-style exactness on the BPC
  side after each merge.
- **M-kimi-03 (`_build_tcf_data`)**: plan: shared dataclass
  `{valid_arcs, S_dict, id_map, pure_binsids, pares_viaveis}` built once in the
  dispatcher and handed to thin per-backend builders; the three model builders
  keep their solver-specific constraint code (the divergence risk lives there,
  not in the prep). Deferred: the prep is ~15% of each wrapper and a rushed
  merge risks the parity now enforced by the M-kimi-02 test.
- **B-kimi-35 (strong branching)**: out of this round's item list; the yaml
  comment now documents the defect (Claude's C1 round).
- **AGENTS.md §strict-rule-4** ("cuts re-injected automatically at descendent
  B&B nodes") now describes deleted machinery — Mistral's docs pass should
  reword it (cuts persist via the shared master; archival is central, replay is
  not implemented).

## Verification summary

- `compileall` clean; ruff clean on every touched package.
- `import_sweep.py`: 0 new failures (the two `logic/gen` failures are
  pre-existing on `6d500ed02` and in Mistral's lane).
- Unit tests: 163/163 across helpers/solvers/hyper_heuristics (patched pristine
  clone).
- `.agent/cache/tools/bpc_bruteforce_check.py`: 40/40 (plain, seed 42, n=6),
  25/25 (plain, seed 11, n=6), 25/25 (mandatory, seed 7, n=7).
- test_sim: 10-day riomaior-20 (plastic, generated dataset) of `swc_tcf aco_hh bpc`
  (cf70/cf90 × lm+cls), `sim.cpu_cores=2`, flock, on the patched pristine clone:
  all six runs complete. Cross-constructor consistency on the shared realization:
  cf70 collected 546.7 kg by all three (SWC-TCF and ACO-HH agree to the km and
  the 106.0 profit; BPC 218.5 km / 100.7 within its gap stops); cf90 collected
  346.8 kg; kg_lost 3.8 tracked; no crashes, no empty-day anomalies. (At n=20 one
  payload covers the network, so the unbounded-fleet path is exercised by the
  unit tests rather than here.)

## Revision (2026-09-28, Codex integration review finding §10.2 #6)

`.agent/cache/patches/kimi/issue-80-revision-swc-tests.patch` (197 lines, applies
on `a323312b3`; verified apply + 6/6 tests on a pristine clone).

- **Deterministic no-incumbent test**: the 1 ms-budget race is replaced by a
  mock at the solver boundary (`gp.Model` returns a fake whose `SolCount = 0`,
  `Status = TIME_LIMIT`; an `_AnyExpr` stub absorbs the model-building
  arithmetic), plus a real-solve zero-solution witness (`R = 0` → the optimal
  plan is empty → `[0, 0]` via the empty-walk normalisation; deterministic
  outcome, not timing).
- **Stronger fleet test**: a capacity-requiring witness — four bins at 100 %
  with `Q = 250 %` cannot fit one route; the unbounded fleet must return two
  routes with all four bins, and the complementary `number_vehicles=1` call
  must drop a bin (bound enforced, not ignored). Together with the adapter's
  `number_vehicles == 0` capture in the depot-split test, the prior
  implementation (policy default 1 + strip-all-zeros) is distinguished.
- Also fixes the cosmetic `Collected:` log count (now counts non-depot nodes).
- Fails-before evidence: with the package reverted to `6d500ed02`, three tests
  fail (`assert [0] == [0, 0]` ×2 on the empty-day shape, and the depot-split
  equality); the wrapper-level witness passes against the old wrapper by
  design — the regression it guards lives at the policy adapter, where the
  split test fails on the old code.
