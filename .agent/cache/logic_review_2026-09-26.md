# Shared report: logic review of 2026-09-26 (paper fidelity, bugs, refactoring, dead code)

**Branch:** `main` · **Commit under review:** `1aa00c09c`
**Kickoff:** `.agent/bus/2026-09-26.md` ("logic review kickoff")
**Protocol:** `.agent/tasks/logic-review-2026-09-26-common.md`
**Lane briefs:** `.agent/tasks/<agent>-logic-review-2026-09-26.md`
**Previous round:** `.agent/cache/minimal_export_review_2026-09-25.md`. Read §6.A.1 and §7 there
before filing anything.

Rules:

- The report is append-only.
- Each agent adds rows to the shared tables and writes its own §5 section.
- Do not edit another agent's rows.
- A row without evidence is a guess and must be labelled as one.

## 0. Scope, entry points and verification commands

**The retained slice.** It is defined in the common brief. In short:

- the policies `aco_hh alns bpc hgs pg_clns psoma sans swc_tcf na`;
- the selectors `lookahead last_minute service_level`, the improver `fast_tsp`,
  and the acceptance criteria `bmc oi`;
- the Attention Model with REINFORCE on `vrpp`;
- the simulator, training and eval code they run through.

The other registered policies are live code (roots for reachability), not
removal candidates.

**Entry points.**

- `main.py` commands `train`, `eval`, `test_sim`, `gen_data` (and the others
  that `main.py` dispatches).
- `logic/controllers`.
- Registries: `GlobalRegistry`, `RouteConstructorFactory`, and the selector,
  improver and acceptance factories.
- Every Hydra yaml and `_target_` under `logic/configs`.

**Verification block.** Run it from your worktree root, with
`PY=~/.cache/wsr-main-venv/bin/python` and `L="flock ~/.cache/wsr-review/heavy.lock timeout 1800"`:

```bash
$PY -m compileall -q logic
$L $PY .agent/cache/tools/import_sweep.py                    # heavy
$L $PY -m pytest --import-mode=importlib -q logic/test/<touched package> -x   # heavy if >1 file

# Small simulator smoke of the affected policies (heavy):
$L $PY main.py test_sim sim.graph.area=riomaior sim.graph.num_loc=20 sim.graph.n_days=10 \
  'sim.graph.dm_filepath="gmaps_distmat_plastic[riomaior].csv"' sim.graph.focus_graph=graphs_20V_1N_plastic.json \
  sim.graph.load_dataset=null sim.data_distribution=emp sim.cpu_cores=2 \
  'sim.policies=[{alns: ${p.alns.alns}},{psoma: ${p.psoma.psoma}}]'
# Logs land under assets/output/10days/ in your worktree; check that km > 0 on some day for each policy.

# Tiny training smoke (heavy). The template is the "train" step of .agent/cache/tools/smoke_minimal.sh
# (embed_dim=32, n_layers=1, n_samples=64, batch 16, one epoch). Adjust any override main rejects,
# and record the adjustment in your §5.
```

The whole `main` test suite (1343 passed, 3 skipped on 2026-09-26) takes
several minutes. Do not run it. Run only the tests that cover what you touch.

## 1. Paper fidelity (append rows)

Class values:

- `bug`: contradicts the paper with no reason;
- `adaptation`: needed for the VRPP / multi-period / mandatory-bin setting;
- `simplification`: deliberate and harmless;
- `undocumented`: plausible, but not stated in a docstring.

| ID | Policy/model | Paper § / eq. / alg. line | Code `file:line` | Deviation | Class | Severity | Status |
|---|---|---|---|---|---|---|---|
| P-codex-01 | REINFORCE rollout baseline | Kool et al., §4, baseline-policy update and Algorithm 1 lines 11–12 | `logic/src/pipeline/rl/common/base/data.py:218`, `logic/src/pipeline/rl/common/base/module.py:260`, `logic/src/pipeline/rl/common/baselines/rollout.py:317` | With nonempty `eval_graphs`, setup populates `val_datasets` but the callback receives `val_dataset=None`; it copies the candidate without testing improvement. See B-codex-01. | bug | major | open; reproduced by Codex |
| P-codex-02 | REINFORCE rollout baseline | Kool et al., §4, paragraph “Determining the baseline policy” | `logic/src/pipeline/rl/common/baselines/rollout.py:313`, `logic/src/pipeline/rl/common/base/module.py:253` | After a significant replacement, no fresh baseline-comparison instances are sampled. The same validation dataset remains in use. See B-codex-02. | bug | major | open; reproduced by Codex |
| P-codex-03 | Training preset | Kool et al., §4 rollout estimator; §5 hyperparameters (first-epoch EMA β=0.8) | `logic/configs/tasks/train.yaml:157`, `logic/configs/tasks/train.yaml:159` | Shipped preset selects the exponential baseline with β=0.6, not rollout with first-epoch EMA warmup. This is a supported experiment choice, not evidence of incorrect gradients; document it as a distinct preset and offer an explicit paper-reference preset. Do not silently change the project default. | undocumented | minor | open; source comparison only |
| P-grok-01 | Simulator profit vs reward | Simulation-framework paper, Eq. profit_function (paper.tex ~189–209) and the plastic coefficients (paper.tex ~804–808) | `logic/src/pipeline/simulations/bins/base.py:385`; `logic/src/pipeline/simulations/day_context.py:655`; `logic/src/pipeline/simulations/states/finishing.py:78` | Logged `profit` is revenue×kg − expenses×km. Plastic defaults are revenue `0.65*898/1000 = 0.5837` €/kg and expenses `1` €/km (`repository/base.py:131-154`), which is the paper identity `0.5837*190 - 96.167 = 14.736`. Logged `reward` is collected kg minus the overflow *count* minus km. The paper states the objective has no overflow penalty. Keep `profit` as the objective; document `reward` as a separate diagnostic or stop publishing it under the objective's name. | undocumented | minor | open; source comparison by Grok |
| P-grok-02 | Overflow flag | Simulation-framework paper, "Overflow is a flag, loss is a quantity" (paper.tex ~744–748) | `logic/src/pipeline/simulations/bins/base.py:431-434` | A day at 100% increments the overflow count even when today's fill is 0. Lost kilograms stay 0 on that day. Owner ruling DS-16 keeps this. The paper's loss sentence is about subsequent increments; the count is the settled flag. | adaptation | — | settled DS-16; do not change |
| P-grok-03 | Policy time | Owner ruling DS-15. The paper's daily cycle (paper.tex ~704–707) names selection, construction, improvement, then execution and logging, and does not define the `time` column. | `logic/src/pipeline/simulations/day_context.py:724-738`; `logic/src/pipeline/simulations/states/finishing.py:67-69` | Daily `time` is mandatory selection + route construction + route improvement. Sample `time` is the sum of those daily values. Fill, collection and logging are outside it. | adaptation | — | settled DS-15; landed on main |
| P-gemini-01 | Decoder query context | Kool et al. §3 Eq. (4) and App. A ($h_{(c)} = [\bar{h}, h_{\pi_t}, \dots]$) | `logic/src/models/subnets/decoders/glimpse/decoder.py:361,442-447`; `logic/src/models/subnets/embeddings/context/vrpp.py:173-218` | The decoder precomputes `graph_context = self.project_fixed_context(node_embeddings.mean(1))` into `fixed.graph_context` but never consumes it in `_get_parallel_step_context` or `one_to_many_logits`. The VRPP step context omits graph mean $\bar{h}$ and fuses only current node, unvisited mean waste, and unvisited mean distance. | adaptation | minor | open; confirmed in repro script |
| P-gemini-02 | Decoder logit projection | Kool et al. §3 (single attention head $M=1$, $d_k=d$ for final selection) | `logic/src/models/subnets/decoders/glimpse/attention.py:82-90`; `logic/src/models/subnets/decoders/glimpse/decoder.py:372` | The final selection logits are projected into $M=8$ heads (`make_heads(logit_K, self.n_heads)`), scaled by $\sqrt{d/M}$, and averaged across heads (`logits.mean(dim=1)`), rather than using a single head with $d_k=d$. | simplification | minor | open; source comparison |
| P-gemini-03 | VRPP Context features | Kool et al. App. A.2 PCTSP ($c_t = [\bar{h}, h_{\pi_t}, \beta_t]$) | `logic/src/models/subnets/embeddings/context/vrpp.py:133-218` | VRPP has no remaining prize threshold. Context extraction adapts by feeding: (1) current node embedding, (2) mean waste of unvisited nodes, (3) mean distance to unvisited profitable nodes. Sound problem adaptation. | adaptation | — | open; documented adaptation |
| P-cursor-01 | BMC | Kirkpatrick, Gelatt & Vecchi 1983 (Metropolis: accept $\Delta E\le 0$ always; $P=\exp(-\Delta E/T)$ for $\Delta E>0$; $E$ is a cost to minimise) | `logic/src/policies/acceptance_criteria/boltzmann_metropolis_criterion.py:95-105` | Code maximises profit: $\Delta =$ candidate $-$ current; accept if $\Delta\ge 0$; $P=\exp(\Delta/T)$ for $\Delta<0$. The sign flip is the correct maximisation rewrite. ALNS and PSOMA pass profit. Ties ($\Delta=0$) are accepted. Geometric $T\leftarrow\alpha T$ matches a standard cooling schedule, not a paper-prescribed $\alpha$. | adaptation | — | open; reproduced by Cursor |
| P-cursor-02 | BMC temperature | Kirkpatrick 1983: $T$ is in the same units as the energy differences being accepted. Ropke & Pisinger 2006 §3.5: $T_0 = w\cdot\|f\|/\ln 2$ so a $w$-relative worsening is accepted with probability $1/2$. | `logic/src/policies/acceptance_criteria/boltzmann_metropolis_criterion.py:52-59`; `logic/src/policies/route_construction/meta_heuristics/adaptive_large_neighborhood_search/alns.py:670-676`; `logic/configs/policies/policy_alns.yaml:45,67`; `logic/configs/policies/other/ac_bmc.yaml:4` | `setup(initial_objective)` is a no-op. Shipped ALNS `start_temp: 100` skips the `start_temp==0` calibration that would use `start_temp_control: 0.05`. BMC therefore runs at $T=100$ €. A $5$ € worsening is accepted with $P\approx 0.951$; the documented $w=0.05$ calibration would have set $T\approx 1.06$ and $P<0.01$. See B-cursor-02. | bug | major | open; reproduced by Cursor |
| P-cursor-03 | Lookahead | Selector yaml (`ms_lookahead.yaml:6-11`) claims a GRF predictor and `predicted_fill >= max_fill (1.0)`. Simulation paper: fill, then select. | `logic/src/policies/mandatory_selection/selection_lookahead.py:41-51,185-186`; `logic/configs/policies/other/ms_lookahead.yaml:6-11` | Code is linear `fill + rate` against `MAX_CAPACITY_PERCENT=100`. No GRF. After the fill action, `current_fill` already includes today, so `fill+rate >= 100` is “overflows tomorrow”. The helper docstring still says “overflows today”. DS-21 (relative days) is already on this tree. | undocumented | minor | open; reproduced by Cursor |
| P-cursor-04 | Service-level / OI | Owner ruling B-cursor-01 (2026-09-25): keep linear $z\cdot\sigma\cdot n_d$. Kirkpatrick accepts $\Delta E=0$. | `logic/src/policies/mandatory_selection/selection_service_level.py:58-67`; `logic/configs/policies/other/ms_service_level.yaml:10-12`; `logic/src/policies/acceptance_criteria/only_improving.py:59` | Service-level yaml describes “collect if fill exceeds $\mu+k\sigma$”. Code projects `fill + D·μ + D·k·σ` against capacity 100. Linear $D\cdot k\cdot\sigma$ is the settled contract (not $\sqrt{D}$). OI is strict `candidate > current` (ties rejected). | undocumented | minor | open; yaml vs code by Cursor; √n remains wontfix |
| P-kimi-01 | BPC node selection — WITHDRAWN | BHV2000 §3.2 “Node Selection”: DFS throughout | `branching/tree.py:103-113`; `bpc_engine.py:531-534,941-951` | Withdrawn by Kimi: the first filing claimed the shipped solver silently runs best-first. Verification showed `getattr(params, alias, explicit_arg)` falls back to the engine’s explicit `search_strategy="depth_first"` argument, so shipped BPC does run DFS (paper-faithful). The real residual is the params-only trap + per-run DeprecationWarning — re-filed as B-kimi-34 (minor). | — | — | withdrawn by Kimi; see §5.D |
| P-kimi-02 | BPC pricing relaxation | BHV2000 §3, shortest-path labelling (elementary paths) on the subproblem network | `logic/src/policies/helpers/solvers_and_matheuristics/pricing/solver.py`; `search/column_generation.py:136-162,371-403` | Pricing uses the ng-route relaxation (Baldacci et al. 2011) instead of elementary labelling, with dynamic ng-expansion plus physical suppression of cyclic columns (`lambda.UB=0`) when fractional cycles appear, and strict elementarity for nodes in active SRI/Ryan-Foster/forced-y constraints. Relaxation is tightened to elementarity where the paper’s argument requires it. | adaptation | — | open; documented in engine docstring §3.2; sound by construction |
| P-kimi-03 | BPC branching stack | BHV2000 §3.2 Steps 1–5: divergence node, roughly-equal arc partitions, explore the side where $A(d,a2)$ is forbidden first | `branching/strategies.py` (divergence); `bpc_engine.py:870-898,941-951` | The divergence rule is implemented (including the shorter-path-first child order, gated on `use_paper_ordering`), but hierarchical node-visitation $y_v$ branching (Boussier et al. 2007) fires first and, when it branches, its children are pushed left-then-right unconditionally — the paper’s child-2-first ordering never applies to $y$-branches. | adaptation | minor | open; documented as VRPP adaptation, ordering side effect undocumented |
| P-kimi-04 | BPC cuts | BHV2000 §4.1/§6: lifted cover inequalities from saturated arcs; cut duals modify pricing arc costs $c'_{lm} = c_{lm} + \pi_{lm} + \alpha_{lm}\gamma_{lm}$ so the pricing algorithm never changes | `search/cutting_planes.py:898,1066` (LCI engines); `master_problem/constraints.py:210-280` (`add_lci_cut`); `pricing/solver.py` `_extend_label` | LCI duals enter pricing exactly as arc-cost adjustments (the paper’s pricing-compatibility requirement, §1.1 contribution 1), archived per-node with alphas. Extra families beyond the paper (RCC, 3-SRI, edge-clique, multistar, SEC) are documented in the module docstring as VRPP additions. | adaptation | — | open; documented |
| P-kimi-05 | BPC termination | BHV2000 §3.2 “Branch-and-Price Termination”: provably optimal or run time exceeded | `search/column_generation.py:430-460`; `bpc_engine.py:802-833` | Engine adds two heuristic stops on top of the paper’s: a global `optimality_gap` 0.5% break and `early_termination_gap` 1% with an explicit “optimality is NOT guaranteed” log. Exactness holds only up to these configured gaps (per owner ruling D1, 375-instance brute-force evidence). | simplification | minor | open; documented in code/yaml |
| P-kimi-06 | ACO-HH visibility | Chen, Kendall & Vanden Berghe (2007) §III.B: $\eta_{ij} = \rho\eta_{ij} + \sum_k T_{kj}\lambda^{I_{kj}}/\mathrm{num}(i,j)$ — CPU time $T_{kj}$ multiplies; the authors explicitly reject the Burke et al. $1/\mathrm{CPU}$ form | `hyper_heuristics/ant_colony_optimization_hyper_heuristic/hyper_aco.py:767-769,796` | Code divides by wall-clock `execution_time` (`+= lam / ((execution_time + 1e-3) * safe_count)`) — the rejected $1/\mathrm{CPU}$ form — so CPU jitter drives $\eta$ and operator selection. $\lambda^I \to \exp(I/Z)$ is a documented adaptation; the $\mathrm{num}(i,j)$ divisor matches. | bug | major | open; reproduced by Kimi (see B-kimi-45); paper txt lines 622-648 |
| P-kimi-07 | ACO-HH evaporation | Chen et al. §III.B: $\tau(t) = \rho\,\tau(t-n) + \sum \Delta\tau$ with $\rho=0.5$ a RETENTION factor | `hyper_aco.py:935`; `params.py:34`; `logic/configs/policies/policy_aco_hh.yaml` | Code does `tau *= 1 - rho`, i.e. treats $\rho$ as the evaporation rate — a semantic inversion identical to the paper only at $\rho=0.5$ (the shipped default, so latent). `params.py:34` claims the parameter is “ρ in the paper”; the yaml documents $(1-\rho)$. | undocumented | minor | open; source comparison by Kimi; paper txt lines 593,655 |
| P-kimi-08 | ACO-HH deposit | Chen et al. §III.B: $\Delta\tau = (I_k + Q)/L_k$ if $I_k>0$ else $0$, deposited by all improving ants along journey edges | `hyper_aco.py:308-316,311` | Code deposits `Q + I_k/L_k` — operator precedence differs from $(I_k+Q)/L_k$. PDF text extraction is ambiguous on the parentheses (txt lines 649-677), so this is a transcription-level suspicion, not a proven deviation. The virtual-row first-hop loss is B-kimi-49. | guess (extraction-ambiguous) | minor | open; label as guess |
| P-kimi-09 | ACO-HH acceptance | Chen et al. §III.B: the best solution $S_b$ is updated after EVERY low-level heuristic application | `hyper_aco.py:265-273` | Code evaluates/accepts only at journey end: a mid-journey improvement that a later operator degrades is lost, so the accepted set differs from the paper’s. | simplification | minor | open; undocumented; paper txt lines 567-568 |
| P-kimi-10 | ACO-HH strategic oscillation | Chen et al. §III.B: oscillation halves only the penalty weight $p_v$; the heuristics still enforce feasibility | `hyper_aco.py:230,753` | Code relaxes capacity to `inf` once $p_v < \text{initial\_}p_v$ (17/60 iterations on the repro run), and best-tracking uses the penalised objective with no feasibility re-check at return → capacity-infeasible “best” routes are returned (B-kimi-46). $p_v\cdot\mathrm{num\_violations}$ and $\text{initial\_}p_v = \text{cost}/100$ are reasonable adaptations. | undocumented | major | open; reproduced by Kimi; paper txt lines 688-693 |
| P-kimi-11 | SWC-TCF formulation | Ramos, Morais, Barbosa-Póvoa (2018) §3.2.2 eqs (11)–(21): undirected model, $0.5\,C\sum x_{ij}d_{ij}$ with double-counted edges, real+copy depot $(0, n{+}1)$ with single $y$ and $\pm 2S_i g_i$ balance (eq 12), $\sum_j x_{ij} = 2g_j$ (eq 18) | `smart_waste_collection_two_commodity_flow/gurobi.py:93-165` (same model `ortools_wrapper.py`, `pyomo_wrapper.py`) | Code reformulates on directed arcs counted once — so full $C\cdot d$ is the faithful travel cost (the pre-fix $0.5$ factor was the actual deviation, DS-27), uses a single depot with $k$ departures and split $f/h$ balances, and degree in==out==$g$. Mathematically equivalent model; the yaml now states the exact objective. | adaptation | — | open; documented in yaml after DS-53; verified across all three backends |
| P-kimi-12 | SWC-TCF units | Paper: $S, Q$ in kg, $R$ in €/kg (txt 549-551) | `base_routing_policy.py:258-260`; `repository/base.py:152-154`; wrappers take percent fills | Code solves in percent units with adapter scaling $R_{kg}\cdot B\cdot V/100$ and $Q_{kg}/(B\cdot V)\cdot 100$. One unit system across all three backends after DS-28. | adaptation | — | open; consistent across backends by Kimi |
| P-kimi-13 | SWC-TCF service level | Paper eq (16): $\delta$ service-level constraint linking collection to fill thresholds | `policy_swc_tcf.yaml` (delta removed, DS-29); `gurobi.py:137-141` | The $\delta$ constraint is not implemented; the code substitutes mandatory-bin forcing (from the selection strategies) plus the $\psi$ force-visit rule. Sensible for the simulator’s mandatory-bin contract, but nothing (docstring/yaml) documents the substitution. | undocumented | minor | open; source comparison by Kimi |
| P-kimi-14 | SWC-TCF arc cutoff | Paper has no distance cutoff in the TCF model | `smart_waste_collection_two_commodity_flow/params.py:80` (`MAX_ARC_DISTANCE_KM = 6000.0`) | A 6000 km arc filter drops all arcs above the threshold before solving; on cutoff-isolated mandatory bins the re-solve silently skips them (WARN only). No paper basis. | undocumented | minor | open; cross-backend constant after DS-30 |
| P-qwen-01 | PG-CLNS vs HVPL components | HVPL (Sun et al. 2023): Phase 1 ACO init, Phase 2 VPL+HGS evolution (teams, seasons, positions, coaching/substitution/learning, promotion/relegation), Phase 3 ALNS refinement | `pheromone_guided_cooperative_large_neighborhood_search/pg_clns.py:86-165` | PG-CLNS implements only a simplified ACO+LNS hybrid: population init by ACO construction, "coaching" = LNS per member, global pheromone update on best, replacement of weakest. Missing: VPL team/season structure, position-based coaching/substitution/learning phases, promotion/relegation, HGS genetic operators (OX crossover, Split). The in-tree `HVPLSolver` (`hybrid_volleyball_premier_league/solver.py`) implements the full three-phase structure. PG-CLNS docstring says "Run the HVPL algorithm" but does not cite HVPL. | undocumented | major | open; source comparison by Qwen |
| P-qwen-02 | PG-CLNS timing clock | DS-15 settled `time.perf_counter()` for policy timing | `pg_clns.py:101,118`; `lns.py:234,236`; `local_search.py:131,133`; `aco.py:191,193` | All four PG-CLNS modules use `time.process_time()` (CPU time) instead of `time.perf_counter()` (wall-clock). Inconsistent with every other policy and the simulator's DS-15 timing. On a multi-core system with I/O, `process_time` can differ significantly from `perf_counter`. | undocumented | minor | open; source comparison by Qwen |
| P-qwen-03 | ALNS σ₁ scoring | Ropke & Pisinger (2006) §3.4: σ₁ for new global best, σ₂ for improving accepted not visited, σ₃ for worsening accepted not visited | `alns.py:717-726` | Code awards σ₁ unconditionally for a new global best (no "not visited" qualifier). The paper's description is ambiguous on this point; the code's interpretation is defensible. σ₂/σ₃ correctly require `is_new_solution`. | adaptation | — | open; documented in code comments |
| P-qwen-04 | PG-CLNS pheromone update | ACS-style: local update during construction + global update on best-so-far | `pg_clns.py:213-230` | Only global pheromone update is implemented (evaporate all + deposit on best). No local pheromone update during ant construction. The ACO constructor (`aco.py`) uses a separate pheromone matrix but the PG-CLNS wrapper only calls `evaporate_all` + `update_edge` on the best solution's edges. This is a simplification of the full ACS local+global scheme. | simplification | minor | open; source comparison by Qwen |
| P-qwen-05 | PSOMA PSO velocity | Liu et al. (2006): discrete PSO with swap-based velocity, inertia ω, cognitive c₁, social c₂ | `solver.py:163-173` | PSO velocity update uses `np.random.rand()` (global numpy state) for r₁, r₂ instead of the seeded `self.random`. Breaks seeded reproducibility. The paper does not specify the RNG, but the rest of PSOMA uses `self.random`. | bug | minor | open; source comparison by Qwen |

## 2. Bugs, errors and logic mistakes (append rows)

| ID | Sev. | `file:line` | What is wrong | Repro / evidence | Proposed fix | Status |
|---|---|---|---|---|---|---|
| B-codex-01 | major | `logic/src/pipeline/rl/common/base/data.py:218`, `logic/src/pipeline/rl/common/base/data.py:269`, `logic/src/pipeline/rl/common/base/module.py:260`, `logic/src/pipeline/rl/common/baselines/rollout.py:295` | Opt-in `rl.baseline=rollout` plus configured evaluation graphs replaces the frozen baseline every callback without any significance test; a worse candidate can replace the best policy. P-codex-01. The shipped VRPP env supplies evaluation graphs, but the shipped baseline is exponential, so the default baseline is not affected. | `codex_rollout_repro_20260926.py`: real module setup produces one `val_datasets` entry and `val_dataset=None`; real epoch-end callback invokes `_rollout` zero times and `setup` once. | Give rollout a dedicated comparison dataset and matching environment; pass that pair to its callback. Permit unconditional copying only for initial setup. On absent/empty comparison data, preserve an initialized baseline and warn or raise. Test multi-graph setup and rejection of a worse candidate. | open; reproduced by Codex |
| B-codex-02 | major | `logic/src/pipeline/rl/common/baselines/rollout.py:75`, `logic/src/pipeline/rl/common/baselines/rollout.py:313`, `logic/src/pipeline/rl/common/base/module.py:253` | Significant baseline replacement only copies policy weights. It retains the comparison instances across promotions, omitting the paper's protection against overfitting that comparison set. P-codex-02. This establishes a lifecycle mismatch, not measured degradation in trained-policy quality. | Same script: mocked rewards `[2.1,2.0,2.5]` versus `[1.1,1.0,1.2]` trigger a real paired test (one-sided p≈0.0041) and replacement; dataset identity is unchanged and generator calls are zero. | Keep fixed reporting validation data separate from a seeded baseline-comparison pool. Refresh only the comparison pool after accepted promotions, preserving environment/distribution and sample-count settings. Test refresh after acceptance, no refresh after rejection, and reproducibility. | open; reproduced by Codex |
| B-grok-01 | major | `logic/src/pipeline/simulations/day_context.py:39-70`, `logic/src/pipeline/simulations/day_context.py:685-703` | The daily policy seed is `adler32` of `get_canonical_policy_name`. That function returns the first slug token that is not a skipped prefix, not the constructor. The docstring example `lookahead_ma_ts_custom_gamma3` is documented as `ma_ts` and returns `ma`. Every `last_minute_*` slug returns `last`, so ALNS, HGS and BPC under last-minute start from the same seed (`70321802` at seed 1234, day 3). `lookahead_cf70_alns` and `lookahead_cf90_alns` return `cf70` and `cf90`, so the same constructor does not share a seed across thresholds. Stochastic constructors (ALNS, ACO-HH, PSOMA, SANS) therefore mix the selector token into their random stream. Waste samples themselves are still paired: `initializing.py:466` and `:492` seed the dataset with `sim.seed + sample_id`, and archived runs use `noise_variance: 0`. | `grok_lane_b_repro_20260926.py` `check_canonical_seed`, run from the `1aa00c09c` worktree. All assertions passed. | Seed with the registered constructor token (longest registry match in the slug), not the first leftover token. Add a test that last-minute ALNS and last-minute HGS differ, and that lookahead-cf70 ALNS matches lookahead-cf90 ALNS. Do not put the selector or the threshold into the solver seed. | open; reproduced by Grok |
| B-grok-02 | major | `logic/src/pipeline/simulations/states/running.py:65`; `logic/src/pipeline/simulations/states/initializing.py:345-364` | Resume is live on `main`. Restored elapsed time is added to `perf_counter()`, so the next sample `time` starts negative. This is DS-17. The export closed it by deleting checkpoints (`01e6ea179`); that commit is not on `main`. DS-15's sum of daily policy times is a separate field and does not repair this clock. | `grok_lane_b_repro_20260926.py` `check_resume_clock`: `tic = perf_counter() + 12.5` then `perf_counter() - tic` prints `-12.5000`. A full resume simulation was not run. | Set `ctx.tic = time.perf_counter() - ctx.run_time`. Test a resumed run whose sample `time` equals stored elapsed time plus the new daily policy-time sum. | open; formula reproduced by Grok. Same defect as DS-17, still present on main |
| B-grok-03 | major | `logic/src/pipeline/simulations/states/running.py:76-80`; `logic/src/pipeline/simulations/simulator.py:517-519`; `logic/src/pipeline/features/test/orchestrator/__init__.py:305-324`; `logic/src/pipeline/features/test/orchestrator/results_handler.py:101-103` | A day-loop exception becomes `CheckpointError` (`checkpoints/manager.py:63-65`). `RunningState` stores it and ends the state machine. Sequential mode then `except CheckpointError: pass`. The orchestrator binds the failure list to `_failed_log` and never reads it, so the process still exits 0. For `n_samples > 1`, a policy with no successful sample is aggregated as an all-zero mean. This is DS-18, still open on `main` because checkpoints stayed. | Code path on `1aa00c09c`. A fault was not injected in this pass (the export's inject-fail check does not apply to this tree). | Record the error dict from both the sequential `CheckpointError` handler and the parallel failure list. If any sample failed, exit non-zero and do not write a zero mean for a missing policy. | open; static on main. Same defect as DS-18 |
| B-grok-04 | minor | `logic/src/pipeline/features/test/orchestrator/parallel_runner.py:168-176` | On a failed worker result the callback pops `sample_id` and then reads `result["sample_id"]`, which is gone. The printed sample and the appended failure dict both lose the id. | `grok_lane_b_repro_20260926.py` `check_parallel_failure_sample`: input `sample_id=3` is reported as `unknown`, and the stored dict has no `sample_id`. | Read `sample_id` before popping, and keep it on the failure record. | open; reproduced by Grok |
| B-grok-05 | major | `logic/src/pipeline/features/test/orchestrator/__init__.py:207-209`, `logic/src/pipeline/features/test/orchestrator/__init__.py:263-280` | `sample_idx_dict` is keyed by the raw `full_policies` entry. The resume filter looks up the display slug. For raw `alns` plus a lookahead / fast-tsp / bmc config, the slug is `lookahead_alns_bmc_fast_tsp`. Resume raises `KeyError` before any day runs. Fresh runs never enter this block. | `grok_lane_b_repro_20260926.py` `check_resume_key`, using `resolve_policy_display_name`. | Key the sample lists by `policy_result_key` (the same slug as `ctx.pol_name` and the log file) in both the initial dict and the resume filter. | open; reproduced by Grok |
| B-grok-06 | minor | `logic/src/pipeline/simulations/bins/base.py:241`, `logic/src/pipeline/simulations/bins/base.py:332-334`, `logic/src/pipeline/simulations/bins/base.py:461` | `stats_filepath` sets `start_with_fill`. Row 0 of the waste sample becomes the initial level, and each day `d` in `1..n_days` then reads `waste_fills[d]`. A sample of shape `(n_days, n_bins)` raises `IndexError` on the last day. The default `stats_filepath` is null, and that path indexes `waste_fills[day - 1]`, which matches. | `grok_lane_b_repro_20260926.py` `check_stats_file_index`: 10-day array, index 10 is out of bounds. | When `start_with_fill` is set, require `n_days + 1` rows or index `waste_fills[day - 1]` for the daily increment and keep row 0 as the initial level only. | open; formula reproduced by Grok |
| B-grok-07 | minor | `logic/src/pipeline/simulations/day_context.py:329-338` | If the policy id contains the tokens `ms` or `ri`, the display-name parser skips the tokens `alns` and `hgs` and then strips the mandatory-selection name. `ms_regular_alns_ri_none` becomes `Regular +  + None`. I did not find this id shape in the archived Hydra configs, so this is latent. | `grok_lane_b_repro_20260926.py` `check_display_name_skips_constructor`. | Pick the constructor by a registry match. Do not exclude `alns` and `hgs` from the candidate tokens. | open; reproduced by Grok |
| B-gemini-01 | major | `logic/src/models/core/attention_model/policy.py:57,75-82`; `logic/src/models/subnets/encoders/gat/encoder.py:42,66-77`; `logic/src/models/subnets/encoders/common/encoder_base.py:93` | `AttentionModelPolicy.__init__` accepts `normalization: str = "batch"` and passes `normalization=normalization` to `GraphAttentionEncoder`. `GraphAttentionEncoder` accepts `norm_config: Optional[NormalizationConfig] = None` and absorbs `normalization` into `**kwargs`. `TransformerEncoderBase` builds `NormalizationConfig()` (default `"batch"`). Passing `normalization="layer"` or `"instance"` is silently ignored, always constructing `nn.BatchNorm1d`. | `gemini_lane_c_repro_20260926.py` `check_b_gemini_01_norm_kwarg_ignored`: `AttentionModelPolicy('vrpp', normalization='layer').encoder.layers[0].norm1.normalizer` is `nn.BatchNorm1d`, not `nn.LayerNorm`. | Forward `NormalizationConfig(norm_type=normalization)` explicitly into `GraphAttentionEncoder(norm_config=...)`. | open; reproduced by Gemini |
| B-gemini-02 | major | `logic/src/policies/route_construction/learning_algorithms/neural_agent/simulation.py:76` | When `mandatory` is provided and empty (`_validate_mandatory(mandatory)` is False), `compute_simulator_day` returns `([0], 0, {"mandatory_empty": True})`. The system-wide empty tour convention settled under owner ruling D3 requires `[0, 0]`. Returning `[0]` causes downstream indexing and slicing errors. | `gemini_lane_c_repro_20260926.py` `check_b_gemini_02_na_empty_mandatory_tour`: returns `([0], 0, ...)`. | Return `([0, 0], 0, ...)` to adhere strictly to owner ruling D3. | open; reproduced by Gemini |
| B-gemini-03 | major | `logic/src/policies/route_construction/learning_algorithms/neural_agent/policy_na.py:145-146` | `NeuralAgentPolicy.execute` calculates `collected_revenue = sum(float(bins.c[n - 1]) * profit_vars.get("revenue_kg", 1.0) ...)`. `bins.c` is the percentage fill $[0, 100]$, not kg! In `bins/base.py:378`, true kilograms is `(bins.real_c / 100) * volume * density`. Directly multiplying percent by `revenue_kg` produces distorted profit values in non-physical units. | Static inspection and comparison with `bins/base.py:378`. | Compute collected revenue using `(float(bins.c[n-1]) / 100.0) * volume * density * revenue_kg`. | open; static on main |
| B-gemini-04 | minor | `logic/src/models/subnets/decoders/glimpse/decoder.py:108,361,369` | `GlimpseDecoder` initializes `self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)`. It computes `fixed.graph_context = self.project_fixed_context(...)` during `_precompute`. Neither `_get_log_p`, `_get_parallel_step_context`, nor `one_to_many_logits` ever consumes `fixed.graph_context`. These $embed\_dim^2$ learnable weights are dead and receive zero gradients. | `gemini_lane_c_repro_20260926.py` `check_b_gemini_04_d_gemini_01_dead_project_fixed_context`: corrupting `fixed.graph_context` to zeros changes logit outputs by 0.0. | Either pass `graph_context` into the step context query (aligning with Kool et al. §3) or remove `self.project_fixed_context` and `fixed.graph_context`. | open; reproduced by Gemini |
| B-gemini-05 | minor | `logic/src/models/core/attention_model/model.py:222`; `logic/src/models/subnets/embeddings/context/vrpp.py:56`; `logic/src/utils/model/loader.py:190` | `AttentionModel` instantiates `self.context_embedder = VRPPContextEmbedder(...)` which instantiates `self.project_step_context = nn.Linear(...)`. However, `AttentionModel` only uses `context_embedder` for initial node embedding (`_get_initial_embeddings`). Decoding delegates to `GlimpseDecoder`, which owns its own `decoder.context_embedding`. `model.context_embedder.project_step_context` is completely dead, forcing `loader.py` to maintain `_UNUSED_LEGACY_KEYS`. | `gemini_lane_c_repro_20260926.py` `check_b_gemini_05_context_embedder_dead_project_step_context`. | Separate initial node embedding (`VRPPInitEmbedding`) from step context extraction (`VRPPContextEmbedder`), eliminating dead layers and loader workarounds. | open; static on main |
| B-cursor-01 | major | `logic/src/configs/policies/other/mandatory_selection.py:77-84`; `logic/src/pipeline/simulations/actions/node_selection.py:171`; `logic/configs/policies/other/ms_service_level.yaml:29-37` | `ServiceLevelSelectionConfig` has only `confidence_factor`. Typed `MandatorySelectionConfig` therefore never carries `horizon_days`. The action then defaults it to `3`. Yaml variants are `horizon_days: 1` and `2`. The live `{file: variant}` parse path keeps the yaml field, so archived sl_ftsp / sl_cls runs are fine. Any Hydra/typed construction of `service_level` silently uses a 3-day projection (3× the yaml SL1 horizon). Vectorized factory default is `1`. | `cursor_lane_f_repro_20260926.py` `check_b_cursor_01_service_level_horizon_dropped`. `asdict(ServiceLevelSelectionConfig(0.84))` has no `horizon_days`. | Add `horizon_days: int = 1` to `ServiceLevelSelectionConfig`. Change the action fallback from `3` to `1` so it matches the yaml SL1 and the vectorized factory. Test typed vs file-parse SL1/SL2. | open; reproduced by Cursor |
| B-cursor-02 | major | `logic/src/policies/acceptance_criteria/boltzmann_metropolis_criterion.py:52-59`; `logic/src/policies/route_construction/meta_heuristics/adaptive_large_neighborhood_search/alns.py:670-676`; `logic/configs/policies/policy_alns.yaml:45,67`; `logic/src/policies/route_construction/meta_heuristics/adaptive_large_neighborhood_search/params.py:127-131` | P-cursor-02. BMC `setup` ignores the objective. ALNS calibrates $T$ from $\|profit\|$ only when `start_temp==0`. The shipped yaml is `start_temp: 100` and still documents `start_temp_control: 0.05` as “accept a 5% worse solution with 0.5 probability”. That comment is inert. Fallback construction also injects `initial_temp=params.start_temp` (100). Plastic profit $\approx 15$ € then accepts a $5$ € worsening with $P\approx 0.95$. Lane E owns ALNS; this is the BMC temperature contract. | Same script `check_b_cursor_02_bmc_temperature_not_scaled`. `setup(14.736)` leaves $T=100$. | Either set yaml `start_temp: 0` so calibration runs, or have `BoltzmannAcceptance.setup` set $T = w\cdot\|f\|/\ln 2$. Rewrite the `start_temp_control` comment to match the gate. Test $P(\Delta=-w\cdot\|f\|)\approx 1/2$ after setup. | open; reproduced by Cursor |
| B-cursor-03 | minor | `logic/src/policies/acceptance_criteria/boltzmann_metropolis_criterion.py:11-14`; `logic/src/policies/acceptance_criteria/only_improving.py:10-12` | Both module examples claim `accept(current_obj=100, candidate_obj=98) → True`. Under maximisation that move is a worsening: BMC at $T\le 10^{-9}$ rejects it; OI always rejects it. Yesterday’s B-cursor-06 already noted the inverted docstring; the examples are still wrong on `1aa00c09c`. | Same script `check_b_cursor_03_docstring_sign`. | Change the examples to a improving move (`10 → 12`) and state that ties are accepted by BMC and rejected by OI. | open; reproduced by Cursor |
| B-cursor-04 | minor | `logic/src/policies/mandatory_selection/selection_last_minute.py:36,51-52`; `logic/src/policies/mandatory_selection/base/eoq.py:88,102` | Yesterday’s B-cursor-07, still open. `fill_ratios = current_fill / max_fill` is computed and passed in; `resolve_trigger_threshold` compares absolute percent to `context.threshold` and never reads the ratios. Class docstring says `>`; the code is `>=`. Default last-minute is a uniform percent cut (`use_eoq_threshold` false). | Same script `check_b_cursor_04_fill_ratios_unused`. Fill `[50,70,90]`, threshold `70` → `[F,T,T]`. | Drop `fill_ratios` from the helper signature, or compare in ratio space. Match the operator to the docstring (`>=`). | open; reproduced by Cursor. Residual of B-cursor-07 |
| B-cursor-05 | major | `logic/src/policies/route_construction/other_algorithms/travelling_salesman_problem/tsp.py:51-53,66`; `logic/src/policies/route_improvement/fast_tsp.py:75-81`; `logic/src/constants/routing.py:180`; `logic/configs/policies/other/ri_ftsp.yaml:6-12,34` | DS-41 residual (cannot seed fast-tsp 0.1.5). `find_tour` has no `seed`; the wrapper still accepts one. $n\le 20$ is Held–Karp and repeatable; $n>20$ is time-budgeted local search and is not. `SCALE=10000` turns a $7$ km edge into $70000$, above the library’s documented uint16 ceiling $65535$ (this Linux build uses a wider `uint_fast16_t`, so it does not wrap here). Failure is not caught: `find_tour` exceptions propagate; only an empty `split_tour` (`[0]` / `[0,0]`) returns the input. Mandatory nodes missing from the constructor tour are not inserted (`resolve_mandatory_nodes` is unused). Yaml still says “Christofides + 2-opt” and `time_limit: 30.0`; the dataclass fallback is $2$ s. | Same script `check_b_cursor_05_fast_tsp_contract`. `inspect.signature(fast_tsp.find_tour)` has no `seed`. Depot-closed tour on a 4-node matrix. Process `[0,1,2,0]` with `mandatory=[1,2,3]` does not add `3`. | Drop the public `seed` or document it as unused. Catch `find_tour` failures and return the input trip. Document `SCALE` vs the uint16 contract. Correct `ri_ftsp.yaml` (library dispatch, not Christofides). A seed cannot fix $n>20$; a longer `time_limit` only reduces variance. | open; reproduced by Cursor. Same defect class as DS-41 |
| B-kimi-34 | minor | `logic/src/policies/helpers/solvers_and_matheuristics/branching/tree.py:103-113`; `logic/src/policies/route_construction/exact_and_decomposition_solvers/branch_and_price_and_cut/bpc_engine.py:531-534`; `logic/src/policies/route_construction/exact_and_decomposition_solvers/branch_and_price_and_cut/params.py:72,78` | `BranchAndBoundTree` reads `params.max_branch_nodes` / `params.tree_search_strategy`, names that do not exist on `BPCParams` (`max_bb_nodes` / `search_strategy`). It works on the shipped path only because `getattr(params, alias, explicit_arg)` falls back to the explicit argument the engine passes; a params-only caller (the documented signature) silently gets `best_first` + 1000 nodes. A `DeprecationWarning` fires on every `run_bpc` (explicit args + params together). `tree.max_nodes` is stored but never enforced — the engine loop caps `nodes_explored` at `params.max_bb_nodes` itself (bpc_engine.py:389,576). Residual of B-kimi-14 with corrected mechanism: the shipped solver does run DFS (see §5.D), so this is the latent trap + per-run warning, not a live wrong-strategy bug. | Direct probe: `BranchAndBoundTree(v_model, params=BPCParams.from_config({}), search_strategy="depth_first", strategy="divergence")` → `DeprecationWarning` emitted; `tree.search_strategy == "depth_first"` (via the explicit-arg fallback); `tree.max_nodes == 1000` (fallback, stored unused). | Rename the lookups to `max_bb_nodes`/`search_strategy`, drop the ignored-argument branch, stop passing explicit args from the engine, and honour `tree.max_nodes` or delete it. | open; reproduced by Kimi. Residual of B-kimi-14 |
| B-kimi-35 | major (latent) | `logic/src/policies/helpers/solvers_and_matheuristics/branching/pruning.py:246-340` (`perform_strong_branching`); `bpc_engine.py:859-868,880-887`; `logic/configs/policies/policy_bpc.yaml:234-238`; `logic/src/configs/policies/bpc.py:96` | Strong-branching lookahead is empirically broken: the D1 fix recorded “returned suboptimal plans on brute-force-checked instances (e.g. 19.16 vs 19.96) … re-solves the live master in place”; it is disabled in yaml and the config default but the code remains live, and `HierarchicalStrongBranching` receives the live master whenever the flag is on (bpc_engine.py:884). Owner brief asks for the paper-fidelity verdict with strong branching enabled: it violates the paper’s exactness claim whenever it fires. | Prior round §7.2 + yaml comment (evidence from 375-instance harness). This round: 8 plain 5-node instances with SB on, 0 suboptimal — defect not re-triggered in this small sample (y-branching fires first on most); the documented brute-force counterexamples stand. | Fix the lookahead to evaluate on a copy of the master (never the live one), then re-enable; or delete the feature and its keys (`enable_strong_branching_heuristic`, `strong_branching_size`). Until then keep default off. | open; documented defect, not re-reproduced in small sample |
| B-kimi-36 | minor | `logic/src/policies/helpers/solvers_and_matheuristics/master_problem/pool.py:185-240` (`GlobalCutPool.apply_to_master`); `master_problem/constraints.py:97,152,204,277,327,425` (archival); `bpc_engine.py:46-47,590`; `search/cutting_planes.py:265-272` | Cut re-injection at descendant nodes never happens: `apply_to_master` has zero non-comment callers (3 docstring/comment mentions). Cuts persist across nodes only because the single shared master object is never rebuilt; the pool’s `add_cut` archival (RCC/SRI/edge-clique/LCI/multistar/SEC 2.1) is write-only. The engine docstring §4.2 and AGENTS.md strict rule 4 claim central archival + automatic re-injection — the archival half is real, the re-injection half is false. (Correction within this row: node-local SEC machinery is NOT dead — `PCSubtourEliminationCut` forms 2.2/2.3 with `local_only=True` populate `active_sec_cuts_local` via `add_sec_cut(global_cut=False)`, and `remove_local_cuts` genuinely removes them per node; only the global pool replay is unwired.) | `rg apply_to_master logic/src` → 3 hits, all comments/docstrings; repro check `pool_replay` lists them. `lci_3node`/exactness checks confirm the solver is unaffected today (shared master). | Either call `master.global_cut_pool.apply_to_master(master)` after `remove_local_cuts()` at each node (making the documented mechanism real), or delete the pool/replay path and fix the docstrings. | open; verified by Kimi |
| B-kimi-37 | minor | `logic/src/policies/helpers/solvers_and_matheuristics/search/cutting_planes.py:1565-1678` (`MinCutInequalityEngine`); `master_problem/model.py:699-717` (`get_node_visitation`); `bpc_engine.py:557-567` | MinCut engine can never add a cut: `violation = y_v − cover_sum` where `get_node_visitation` and the engine’s `cover_sum` compute the identical double loop over `lambda_vars`×`node_coverage` → violation ≡ 0 (repro max diff 0.0; 0 cuts on a live master). Latent hazard: its fallback branch rewrites the base `coverage_{node}` constraint (`Sense=GREATER_EQUAL`, `RHS=max(RHS, y_v−violation/2)`), which would corrupt the master model if it ever fired. Engine is constructed unconditionally at every node. Residual of B-kimi-08 with refined mechanism. | `kimi_lane_d_bpc_20260926.py check_min_cut_zero`: LP over 3 columns, max abs diff between `y_v` and `cover_sum` is 0.0, `separate_and_add_cuts` returned 0. | Delete the engine (and its wiring); if min-cut inequalities are wanted, define them on arc flow (paper §4 spirit), not on coverage sums that tautologically agree. | open; reproduced by Kimi. Residual of B-kimi-08 |
| B-kimi-38 | minor | `logic/src/policies/helpers/solvers_and_matheuristics/search/cutting_planes.py:1680,1970,2097` (`TriangleCliqueCutEngine`, `NodeProfitBoundEngine`, `PathEliminationEngine`); `bpc_engine.py:557-567` | Three advertised supplemental cut families are still silent no-ops: the master has no `add_conflict_cut`/`add_clique_cut` (`TriangleClique`), no `add_profit_bound_cut` (`NodeProfitBound`), no `add_path_elimination_cut` (`PathElimination`) — grep finds protocol mentions only. All three are constructed and `separate()`d unconditionally at every node (wasted work, zero effect). Residual of B-kimi-08. | grep for `add_conflict_cut`, `add_clique_cut`, `add_profit_bound_cut`, `add_path_elimination_cut` over `logic/src` → 0 definitions; engines reachable in the composite built at `bpc_engine.py:557-570`. | Implement the master `add_*` methods or delete the engines and their wiring; until then drop them from the composite. | open; static on main. Residual of B-kimi-08 |
| B-kimi-39 | minor | `branch_and_price_and_cut/params.py:61,152`; `helpers/solvers_and_matheuristics/pricing/smoothing.py:221-306,381,415-426`; `search/column_generation.py:319-328` | `enable_dssr: true` (yaml default) and `dssr_max_iters` are read nowhere: the CG loop calls `solve_pricing_step` without `use_dssr` (default False), and the DSSR wrapper itself operates on `pricing_solver._ng_memory`, an attribute that does not exist (the solver has `ng_neighborhoods`, `pricing/solver.py:127`). The shipped “DSSR combined with ng-neighborhoods” story (policy_bpc.yaml header) does not exist. Residual of B-kimi-06. | `kimi_lane_d_bpc_20260926.py check_dead_keys`: “use_dssr” absent from `column_generation.py`; fresh `RCSPPSolver` has no `_ng_memory`. | Delete the wrapper + keys, or wire `use_dssr` through the CG loop and rename the attribute to `ng_neighborhoods` with snapshot/restore. | open; reproduced by Kimi. Residual of B-kimi-06 |
| B-kimi-40 | minor | `branch_and_price_and_cut/bpc_engine.py:712-739`; `branch_and_price_and_cut/params.py:63,159`; `pricing/smoothing.py:436-500` | `enable_reduced_cost_arc_fixing: true` (yaml) is inert: the engine block is gated on `hasattr(pricing_solver, "_forbidden_arcs")`, which is False on a fresh solver (the solver only has `fixed_arcs`), so it never runs. The signature mismatch (`capacity` kwarg) is fixed — the gate is what blocks it. Residual of B-kimi-07. | `kimi_lane_d_bpc_20260926.py check_dead_keys`: fresh `RCSPPSolver` lacks `_forbidden_arcs` → gate closed. | Delete the block + key, or set `_forbidden_arcs` on the solver and add a restore hook (ng-snapshot already exists). Note: the separate global `_apply_reduced_cost_edge_fixing` (bpc_engine.py:828-833) DOES run and is the live arc-fixing path. | open; reproduced by Kimi. Residual of B-kimi-07 |
| B-kimi-41 | minor | `master_problem/model.py:208,404-431`; `search/column_generation.py:125-126,361-369,430-436`; `logic/configs/policies/policy_bpc.yaml:25-26,258` | Wentges dual smoothing can never turn on: `enable_dual_smoothing` starts False, nothing ever sets it True (grep `enable_dual_smoothing = True` → 0 hits; the flag is only ever set False). The “smoothing recovery” branch and the UB-prune’s smoothing exclusion are unreachable, and the yaml comment “exact_mode: false — Wentges smoothing dramatically accelerates CG” describes behaviour that does not exist. Residual of B-kimi-10. | `kimi_lane_d_bpc_20260926.py check_dead_keys` (no `= True` anywhere; master default False). | Either set `master.enable_dual_smoothing = not exact_mode` at node start (the documented intent) or delete the smoothing machinery + fix the yaml comment. | open; reproduced by Kimi. Residual of B-kimi-10 |
| B-kimi-42 | minor | `pricing/solver.py:865` | The Farkas feasibility check is still a tautology: `if self.is_farkas and new_rc < -1e-6 or not self.is_farkas and new_rc < -1e-6:` ≡ `new_rc < -1e-6` in both phases — a half-finished phase distinction. Behaviour is correct as written (both phases keep labels with `rc ≥ −1e-6`), so this is clarity-level. Residual of B-kimi-12. | sed-verified on 1aa00c09c (line number shifted from 859 to 866 after the D1 fixes). | Collapse to `if new_rc < -1e-6` or implement the intended phase distinction explicitly. | open; static on main. Residual of B-kimi-12 |
| B-kimi-43 | minor | `pricing/labels.py:103-115`; `pricing/solver.py:732,736` | SRI-aware dominance is dead code: the exact `sri_state !=` early return makes both the “potential penalty” branch and its `else` unreachable, and one dominance call site passes `sri_dual_values` while the other does not (no effect today). Dominance itself remains sound (conservative). Residual of B-kimi-13. | Source trace on 1aa00c09c; yesterday’s 30-instance exact-DP vs brute-force check (0/30 mismatches) still stands. | Remove the dead branches, or implement SRI-aware (non-exact) dominance fully and pass `sri_dual_values` at both sites. | open; static on main. Residual of B-kimi-13 |
| B-kimi-44 | major | `hyper_heuristics/ant_colony_optimization_hyper_heuristic/hyper_operators.py:466`; `helpers/operators/perturbation_shaking/perturb.py:40-41` | `apply_perturb` drops the seeded rng: `return perturb(ctx, k)` while every sibling wrapper passes `rng=ctx.rng` (hyper_operators.py:480-481,497,499,541,582). `perturb` then does `rng = Random()` seeded from OS entropy — so the perturb operator alone breaks seeded reproducibility, independent of the wall-clock $\eta$ defect (B-kimi-45). | `kimi_lane_d_aco_20260926.py`: patched-clock runs still differ (A≠B); with this one-line fix + patched clock A==B. Parent verified hyper_operators.py:466 and perturb.py:40-41 directly. | `return perturb(ctx, k, rng=ctx.rng)`. | open; reproduced by Kimi |
| B-kimi-45 | major | `hyper_aco.py:767-769,796` (wall-clock $\eta$); B-kimi-44 (rng drop) | Run-to-run non-determinism with a fixed seed, now worse than filed: 6 runs seed=42 give **3 distinct route sets** (2 before). Two independent causes: visibility divides by wall-clock `execution_time`, and `apply_perturb` reseeds from OS entropy. Seeds are forwarded correctly (`policy_seed → random.Random/np.random.default_rng`) but do not confer reproducibility. Residual of B-kimi-28. | `kimi_lane_d_aco_20260926.py` + `kimi_lane_d_aco_clockdebug_20260926.py`: with `perf_counter` patched deterministic and B-kimi-44 fixed, consecutive runs match exactly. | Make the $\eta$ denominator deterministic (cumulative success count, or drop the time factor per P-kimi-06) and fix B-kimi-44. Add the §6.A.7 smoke assertion “ACO-HH same seed → same tour”. | open; reproduced by Kimi. Residual of B-kimi-28 |
| B-kimi-46 | major | `hyper_aco.py:230,270-272,321-325,753` | Strategic oscillation returns capacity-infeasible routes as the best solution: `pv` halves with no floor until `pv < initial_pv` relaxes capacity to `inf`; best-tracking uses the penalised objective; `best_routes` is returned with no feasibility re-check at real capacity. Repro: initial greedy feasible; returned best one route at **322.2 kg vs capacity 50**, with an empty route alongside (route hygiene). The simulator receives and executes the over-capacity plan. Residual of B-kimi-30. | `kimi_lane_d_aco_20260926.py`: 17/60 iterations ran with capacity relaxed to inf; min pv 0.065 (initial 1.042); returned best over capacity. | Track best-feasible separately (evaluate at real capacity, skip empty routes) and return it; floor `pv`; re-check feasibility before returning. | open; reproduced by Kimi. Residual of B-kimi-30 |
| B-kimi-47 | major | `hyper_aco.py:145,150`; `policy_aco_hh.py:128`; `params.py:75`; `logic/configs/policies/policy_aco_hh.yaml:111` | yaml `operators` list (5 operators) silently ignored: the solver always runs all 11 (`operator_names = list(HYPER_OPERATORS.keys())`); `params.operators` is stored but never read. Residual of B-kimi-26. Files byte-identical to `70e660b03`. | `kimi_lane_d_aco_20260926.py`: `HyperACOParams(operators=["swap"])` → solver still runs all 11; per-solve application counts include operators outside the yaml list. | Build `operator_names` from `params.operators` (validated against the registry) or delete the key. | open; reproduced by Kimi. Residual of B-kimi-26 |
| B-kimi-48 | major | `logic/configs/policies/policy_aco_hh.yaml:91`; `logic/src/configs/policies/aco_hh.py:63`; `hyper_aco.py:150`; `ant_colony_optimization_hyper_heuristic/params.py` | yaml `sequence_length: 5` silently ignored: the policy-level `HyperACOParams` has no field for it and the solver pins `sequence_length = n_operators` (11); the config dataclass carries the value (`configs/policies/aco_hh.py:63`) but the adapter never forwards it. The documented “Range: 2-10” knob does not exist. Residual of B-kimi-27. | `kimi_lane_d_aco_20260926.py`: `HyperACOParams` lacks the attribute; `_select_sequence` loops `range(11)` regardless of config. | Add the field to `HyperACOParams`, forward it in `policy_aco_hh.py`, and use it in the solver — or drop the yaml key + config field. | open; reproduced by Kimi. Residual of B-kimi-27 |
| B-kimi-49 | major | `hyper_aco.py:308-316,312,874-877` | First-hop operator transitions never reinforced: the deposit loop restarts at `prev_op_idx = n_operators` (virtual row) while selection departs from the ant’s real vertex — the traversed first edge gets zero pheromone and deposits accumulate on row 11, which `_select_sequence` never reads (instrumented: `tau[start][first]` unchanged at baseline, `tau[11][first]` grows). The comment at :312 describes the opposite of the code. Diverges from Chen et al. (deposit along the traversed path). Residual of B-kimi-29. | `kimi_lane_d_aco_20260926.py` spy run: 3/3 improving journeys deposit the first hop on the virtual row only. | Initialise `prev_op_idx` to the journey’s real start index (pass it into the deposit loop). | open; reproduced by Kimi. Residual of B-kimi-29 |
| B-kimi-50 | minor | `hyper_operators.py:525-526,566-567,607-608` | `apply_shaw_removal`/`apply_string_removal`/`apply_random_removal` still wrap everything in bare `except Exception: return False` — a node/matrix mismatch degrades silently into “operator did nothing” with no log. Residual of B-kimi-31. | Verified in place on 1aa00c09c (byte-identical file); yesterday’s `kimi_lane_d_aco_swallow_20260925.py` evidence stands. | Catch specific exceptions and log; re-raise programming errors. | open; static on main. Residual of B-kimi-31 |
| B-kimi-51 | minor | `hyper_aco.py:281`; `assets/diagrams/code/aco_hh_flowchart.dot:46` | Elitism sync count uses `int(n_ants * elitism_ratio)` (floor); the flowchart says $\lceil n_{ants}\cdot\text{ratio}\rceil$ (10 ants × 0.35 → 3 vs 4). Residual of B-kimi-33. | Source comparison; identical for the shipped 0.5/0.2 × 10, differs otherwise. | Use `math.ceil` or fix the diagram. | open; source comparison. Residual of B-kimi-33 |
| B-kimi-52 | minor | `hyper_aco.py:30`; `ant_colony_optimization_hyper_heuristic/params.py:12`; `ant_colony_optimization_hyper_heuristic/__init__.py:15` | Stale docstring references to `logic.src.policies.ant_colony_optimization_hyper_heuristic` (a path that does not exist — the package lives under `hyper_heuristics/`); the `__init__` docstring also documents a non-existent `run_hyper_heuristic_aco`. Cosmetic but misleading. | grep: zero imports of the stale path; the named function has no definition. | Fix the docstrings. | open; static on main |
| B-kimi-53 | major | `smart_waste_collection_two_commodity_flow/pyomo_wrapper.py:233-238,246-250` | `_has_solution` treats `maxTimeLimit`/`feasible` as “has solution” without checking an incumbent exists; `model.solutions.load_from(results)` then raises `ValueError: Cannot load a SolverResults object with bad status: aborted`. Crash is instance-dependent (needs a no-incumbent time limit): the verifier observed it on a 14-bin `time_limit=0.001` instance; the parent’s 14-bin repro returned `[0]` (gurobi produced an incumbent in time). Breaks the DS-31 cross-backend failure alignment — native gurobi/OR-Tools log WARN + empty day, pyomo raises through the day loop (which Grok’s B-grok-03 then swallows). | `kimi_lane_d_swc_20260926.py` (verifier crash log); parent re-verified the code path (`_has_solution` at :233-238, unguarded `load_from` at :249) and ran the no-crash variant. | Require `len(results.solution) > 0` (or wrap `load_from` in try/except) before the solution branch. | open; code-verified + crash observed by verifier |
| B-kimi-54 | major | `smart_waste_collection_two_commodity_flow/pyomo_wrapper.py:151-156` | When the depot has no valid arcs (all cut by `MAX_ARC_DISTANCE_KM`), `model.depot_waste_out` is `sum(model.f[0,j] for ...) == 0` over an empty generator — a trivial-Boolean `True` — so model construction raises `ValueError: Invalid constraint expression … trivial Boolean`. Native gurobi/OR-Tools return empty day + WARN on the same instance (7000-km matrix, both bins mandatory). Verified by parent. | Parent repro: `run_swc_tcf_optimizer(framework="pyomo", optimizer="gurobi")` on `[[0,7000,7000],[7000,0,1],[7000,1,0]]` → `ValueError: Invalid constraint expression. The constraint expression resolved to a trivial Boolean (True)`. | `Constraint.Skip` when no depot arcs exist, or write the balance as `sum(...) <= 0` / use a 0-lower-bound variable. | open; reproduced by Kimi |
| B-kimi-55 | minor | `logic/src/constants/routing.py:125` (`MIP_GAP = 0.01`); `smart_waste_collection_two_commodity_flow/gurobi.py:182`; `ortools_wrapper.py`; `pyomo_wrapper.py` | `MIP_GAP` is applied only by the native gurobi wrapper; OR-Tools and pyomo run solver defaults (1e-4) → cross-backend plan/profit divergence within tolerance (native gurobi 380.26 vs optimal 381.81 = 0.4% on the shared 8-bin instance). | `kimi_lane_d_swc_20260926.py` shared-instance check across the three backends. | Apply one gap policy in the dispatcher (or per-wrapper constants) so all backends honour the configured gap. | open; reproduced by Kimi |
| B-kimi-56 | minor | `smart_waste_collection_two_commodity_flow/gurobi.py:200,236`; `ortools_wrapper.py:212`; `pyomo_wrapper.py:284`; `dispatcher.py:78` | Empty-day route shape differs by backend: native gurobi returns `[0]`, OR-Tools/pyomo return `[0, 0]`. The dispatcher’s GUROBI-fallback check (`result[0] == [0, 0]`, dispatcher.py:78) is therefore sensitive to which backend produced the empty result. | Source comparison + empty-day repros on the cutoff instance. | Normalise empty days to the D3 shape `[0, 0]` in all three wrappers. | open; source comparison by Kimi |
| B-kimi-57 | minor | `smart_waste_collection_two_commodity_flow/gurobi.py:122-123`; `ortools_wrapper.py:106`; `pyomo_wrapper.py:98` | `number_vehicles == 0` (unlimited fleet) fallback uses `len(binsids)` in the native gurobi wrapper — which may be $n{+}1$ — versus `n_bins` in the OR-Tools/pyomo wrappers: different inferred fleet sizes for the same input. Latent (simulator passes `n_vehicles=0`→?). | Source comparison on the three wrappers. | Compute the fallback fleet from the same quantity in all three (or thread `number_vehicles` through the dispatcher explicitly). | open; source comparison by Kimi |
| B-kimi-58 | minor | `smart_waste_collection_two_commodity_flow/ortools_wrapper.py:185`; `pyomo_wrapper.py:259`; `policy_swc_tcf.py:147` | Multi-vehicle extraction: OR-Tools/pyomo iterate `range(int(k_var))` (fractional $k$ truncates) versus gurobi’s robust while-True loop; `policy_swc_tcf.py:147` strips ALL depot zeros, merging $k>1$ routes into one boundary-less route list. Harmless today (`kwargs.get("number_vehicles", 1)` is always 1 — the context passes `n_vehicles`, never `number_vehicles`), but the multi-vehicle path is silently wrong. Also: pyomo’s highs backend gets a seed but no time limit (pyomo_wrapper.py:225-231). | Source comparison; 2-vehicle check returned an equal-profit alternative ordering (not a mapping bug). | Extract routes by arc tracing in all three wrappers; pass `number_vehicles` explicitly; set the highs time limit. | open; source comparison by Kimi |
| B-kimi-59 | major (latent) | `base/base_routing_policy.py:228-236,284`; `pipeline/simulations/actions/route_construction.py:68` | Typed-config path drops runtime overrides nested under the engine key: a plain-dict `runtime_overrides` is merged unflattened (`values` gains a junk `"gurobi"` key and `vrpp`/`time_limit`/… overrides never reach `values`). The simulator takes the typed path (`get_adapter(solver_key, config=raw_cfg)`), so HPO/`pol_cfg` engine-nested overrides are silently dropped; the legacy path flattens correctly. Shipped yaml is masked by construction-time flattening. Residual of B-kimi-21. | `kimi_lane_d_swc_20260926.py` “typed path drops nested overrides”: runtime `{"gurobi":[{"time_limit":7.0},{"vrpp":False}]}` → typed path keeps tl=5.0 and `vrpp` never reaches values; legacy path applies both. | Flatten `runtime_overrides` unconditionally (the non-typed `_flatten_raw_config` handles dicts). | open; reproduced by Kimi. Residual of B-kimi-21 |

| B-mistral-01 | major | `logic/controllers/manager/batch_step_executor.py:119-122` | The batch `gen_dist_matrix` step builds `root / "logic" / "scripts" / "gen_dist_matrix.py"` and runs it. Commit `d2ccc391e` renamed `logic/scripts` to `logic/gen` "and update all references" and missed this call site. `logic/scripts/` does not exist; any batch job whose `gen_dist_matrix` step finds no existing matrix (`check_exists` defaults true, so existing matrices hide the bug) crashes with FileNotFoundError. | `ls logic/scripts` → missing on `1aa00c09c`; `git log --oneline --all -- logic/scripts` → `d2ccc391e`; the live script is `logic/gen/gen_dist_matrix.py`. A batch job was not executed. | Point the step at `logic/gen/gen_dist_matrix.py`. Add a batch test that runs the step with `check_exists=false` in a temp root. | open; static on main, rename history verified by Mistral |
| B-mistral-02 | major | `matheuristics/exact_guided_heuristic/dispatcher.py:31`; `matheuristics/exact_guided_heuristic/__init__.py:28-30`; `learning_matheuristic_algorithms/learning_allocated_sequential_matheuristic/dispatcher.py:39`; `matheuristics/__init__.py:18-35` | The registered policies EGH and LASM are import-broken on `main`: `dispatcher.py:31` does `from .params import PipelineParams`, but `class PipelineParams` does not exist anywhere in `logic/` (it was renamed `ExactGuidedHeuristicParams`, `params.py:33`) — `grep -rn "class PipelineParams" logic` → zero. The EGH package `__init__` imports the dispatcher, so importing the package fails; LASM's dispatcher imports EGH's `route_pool`, so LASM fails through the same chain. Neither policy module is imported by its parent package `__init__` (matheuristics lists AKS…TPKS, no EGH), so neither ever registers at runtime: registered-in-yaml, dead-on-arrival. | `import_sweep.py` on the worktree: 15 of 23 FAILs are EGH+LASM modules (all `ImportError: cannot import name 'PipelineParams'`); direct import of `...exact_guided_heuristic.dispatcher` on the pristine shared checkout fails identically. Full list in `mistral-out/import_sweep.log`. | One-line fix: import `ExactGuidedHeuristicParams as PipelineParams` in `dispatcher.py:31` (or update the call sites). Then decide (owner): wire EGH/LASM into their package `__init__` so they register, or delete both policies. | open; reproduced by Mistral on pristine main |
| B-mistral-03 | minor | `logic/src/configs/policies/sa.py` (`SAConfig.iterations_per_temp`); `logic/configs/policies/policy_sa.yaml:12,36` | The yaml comment claims "`(initial_temperature, cooling_rate, iterations_per_temp)` mapped directly" into the solver and sets `iterations_per_temp: 100`. No code reads `iterations_per_temp` anywhere (`grep` over `logic/src` excluding the defining config → zero). The knob is inert; SA runs the same number of iterations per temperature regardless of it. | `grep -rn iterations_per_temp logic/src --include=*.py | grep -v configs` → zero hits; config-consistency probe (`mistral_lane_g_config_consistency_20260926.py`, output `mistral_config_out_20260926.txt`). | Wire the field into the SA solver or delete field + yaml key + comment. Do not leave a documented knob that does nothing. | open; grep-verified by Mistral |
| B-mistral-04 | minor | `logic/src/configs/tracking.py` (`TrackingConfig.wst_tracking_uri`, `.real_time_log`, `.profiler_buffer_size`); `logic/configs/tracking/{gen_data,hpo,...}.yaml:6,24,40` (7 files); `logic/test/e2e/test_cli_commands.py:96,211` | Three tracking config fields are declared, set by all 7 `logic/configs/tracking/*.yaml` files, and passed on the CLI by an e2e test (`tracking.wst_tracking_uri=...`), but zero code reads any of them (the used field is `tracking_uri`). The e2e test "verifies" an inert flag; users configuring `wst_tracking_uri` silently get the default store location. | Same probe + direct grep: only `tracking_uri` has readers; `wst_tracking_uri`, `real_time_log`, `profiler_buffer_size` have none outside `configs/tracking.py`. | Delete the three fields and their yaml keys, or wire them (esp. `wst_tracking_uri`, which reads like an intended store override). Pick one; do not keep the dead pair. | open; grep-verified by Mistral |
| B-qwen-01 | major | `pg_clns.py:101,118`; `lns.py:234,236`; `local_search.py:131,133`; `aco.py:191,193` | PG-CLNS uses `time.process_time()` (CPU time) for time-limit checks. P-qwen-02. Inconsistent with DS-15 (`perf_counter`) and all other retained policies. On a system where the process is scheduled on fewer cores than available, `process_time` can be significantly less than `perf_counter`, causing PG-CLNS to run longer than the configured `time_limit`. | `grep -rn "time.process_time" logic/src/policies/route_construction/meta_heuristics/pheromone_guided_cooperative_large_neighborhood_search/` → 8 hits. No other retained policy uses `process_time`. | Replace `time.process_time()` with `time.perf_counter()` in all four files. | open; static on main |
| B-qwen-02 | major | `solver.py:163-173` | PSOMA PSO velocity update uses `np.random.rand()` (global numpy random state) for the r₁, r₂ vectors instead of the seeded `self.random`. Breaks seeded reproducibility: two runs with the same seed produce different velocity updates. The rest of PSOMA correctly uses `self.random`. | `solver.py:165`: `r1, r2 = np.random.rand(self.n_nodes), np.random.rand(self.n_nodes)`. Compare with `solver.py:90`: `self.random = random.Random(params.seed)`. | Replace `np.random.rand(n)` with `np.array([self.random.random() for _ in range(n)])` or use `self.np_random = np.random.default_rng(seed)` and `self.np_random.random(n)`. | open; static on main |
| B-qwen-03 | minor | `pg_clns/operators/destroy/worst.py:17-52` | PG-CLNS's local `worst_removal` does a one-shot greedy sort-and-take (sort once by detour saving, take top `n_remove`). The shared `helpers/operators/destroy_ruin/worst.py` implements the proper Ropke & Pisinger (2006) randomized version with parameter `p ≥ 1` and iterative recalculation. PG-CLNS never passes `p` and never randomizes. This means PG-CLNS's worst removal is deterministic and does not match the ALNS paper's algorithm. | Side-by-side: PG-CLNS `worst.py:37-52` sorts once and pops top-n. Helpers `worst.py:44-78` recalculates per removal with `floor(y^p * |L|)` index. | Switch PG-CLNS to import `worst_removal` from `helpers/operators/destroy_ruin/worst.py` (M-qwen-01). | open; source comparison by Qwen |
| B-qwen-04 | minor | `pg_clns/operators/destroy/random.py:28-56` | PG-CLNS's local `random_removal` uses index-based popping (`routes[r_idx].pop(n_idx)`) with targets sorted by descending index. The shared `helpers/operators/destroy_ruin/random.py` uses a set-based single-pass filter. Both are correct but produce different removal orderings for the same seed, so PG-CLNS random removal is not reproducible against the shared operator even with the same RNG state. | Side-by-side: PG-CLNS `random.py:43-56` pops by index. Helpers `random.py:37-50` filters with a set. | Switch PG-CLNS to import from `helpers/operators/destroy_ruin/random.py` (M-qwen-01). | open; source comparison by Qwen |

## 3. Refactoring, modularity and duplication (append rows)

| ID | Topic | Locations (≥2 `file:line` for duplicates) | What repeats / what is tangled | Proposed shared home and call-site changes | LOC saved | Tests covering today | Risk | Status |
|---|---|---|---|---|---|---|---|---|
| M-codex-01 | Greedy policy invocation in rollout baseline | `logic/src/pipeline/rl/common/baselines/rollout.py:171`, `logic/src/pipeline/rl/common/baselines/rollout.py:206` | Both paths branch on `set_strategy`, invoke a legacy or modern policy, and convert tuple output to a reward dictionary. The surrounding dataset reset/padding and single-batch copying are different and must remain separate. | Add a private invocation helper in the existing `rollout.py`; call it from `_rollout_dataset` and `_rollout_batch` after their existing input preparation. Preserve legacy tuple semantics pending lane C's model-contract audit. | Estimated 3–6 net executable lines; no patch trial, not verified savings | `logic/test/unit/pipeline/rl/common/test_baselines.py::TestBaselines::test_rollout_baseline_eval` covers batch rollout with a mocked policy. No side-by-side dataset/batch and legacy/modern parity test identified in the inspected baseline test files; add these before extraction. | medium | open; proposed, not trialed |
| M-grok-01 | Build the scenario tree only for a solver that reads it | `logic/src/pipeline/simulations/actions/route_construction.py:130-157`; consumers live under non-retained solvers (`policy_ph.py:133`, `policy_st_ef.py:121`, `policy_lbbd.py:88`, `base_multi_period_policy.py:61`) | Every retained day, including ALNS and BPC, constructs a 7-step `ScenarioGenerator` tree and stores it on the context before `adapter.execute`. The nine retained constructors do not read `scenario_tree`. The other registered solvers do, so `bins/prediction.py` (314 lines) stays. | Add a registry flag or an explicit `policy.scenario_method` request. Call the existing generator only then. Do not delete `prediction.py`. | About 28 lines removed from the action. No module deletion. Not trialed. | `logic/test/unit/policies/solvers/test_ph.py` and `test_st_ef.py` cover the generator. No test covers the per-day build inside `RouteConstructionAction`. | medium | open; proposed, not trialed |
| M-grok-02 | One writer for sample mean and std | `logic/src/pipeline/simulations/states/finishing.py:100-102`; `logic/src/pipeline/simulations/simulator.py:196-203`; `logic/src/pipeline/simulations/simulator.py:569-583`; `logic/src/tracking/logging/modules/analysis.py:623-624` | Mean and std of the same `SIM_METRICS` vector are written from finishing (only when `n_samples == 1`), from the sequential loop, from `display_log_metrics`, and again from `output_stats` on resume. The copies agree when every sample succeeds. B-grok-03 is what makes the zero-mean copy harmful. | Keep sample and daily writes in `FinishingState`. Write mean and std once, from `display_log_metrics`, after aggregation. Resume keeps `output_stats` as the reader of existing logs, not a fourth formula. | About 30 lines of duplicated mean/std updates. Not trialed. | `logic/test/unit/pipeline/simulations/test_states.py` mocks `update_policy_log_section`. `logic/test/unit/pipeline/simulations/test_features_test.py` patches `display_log_metrics`. No test asserts the file is written once. | medium | open; proposed, not trialed |
| M-gemini-01 | Unify legacy AttentionModel and AttentionModelPolicy | `logic/src/models/core/attention_model/model.py:44-515`; `logic/src/models/core/attention_model/policy.py:33-220` | Both wrap `GraphAttentionEncoder` + `GlimpseDecoder` + VRPP embeddings. `AttentionModel` is legacy (used by `eval` and `NeuralAgent`), while `AttentionModelPolicy` is the RL4CO-standard training policy. Maintaining two parallel implementations causes key remapping shims in `loader.py` and output discrepancies (`cost` vs `reward`). | Make `AttentionModel` a thin adapter subclass of `AttentionModelPolicy` (or migrate `eval` and `NeuralAgent` to call `AttentionModelPolicy` directly). | ~350 LOC | `logic/test/unit/models/test_models.py`, `logic/test/integration/test_pipeline_e2e.py` | medium | open; parity verified |
| M-gemini-02 | Deduplicate autoregressive decoding loop in DeepDecoderPolicy | `logic/src/models/core/attention_model/deep_decoder_policy.py:95-186`; `logic/src/models/core/attention_model/policy.py:92-220` | The entire 90-line autoregressive decoding loop (`while not td["done"].all(): ... state_wrapper ... logits, mask = self.decoder._get_log_p ... _select_action ... td = env.step(td) ...`) is copy-pasted verbatim between `AttentionModelPolicy` and `DeepDecoderPolicy`. | Move the common `while not td["done"].all()` construction loop to `AutoregressivePolicy.forward` in `logic/src/models/common/autoregressive/policy.py`, allowing subclasses to override only encoder/decoder setup. | ~80 LOC | `logic/test/unit/models/subnets/test_deep_decoder.py`, `logic/test/unit/models/test_attention_model.py` | low | open; proposed, not trialed |
| M-gemini-03 | Consolidate VRPPInitEmbedding and VRPPContextEmbedder | `logic/src/models/subnets/embeddings/vrpp.py:22-69`; `logic/src/models/subnets/embeddings/context/vrpp.py:27-131` | Both define duplicate linear layers: `node_embed` / `init_embed` (Linear 3 -> embed_dim) and `depot_embed` / `init_embed_depot` (Linear 2 -> embed_dim) to project coordinates and waste. `VRPPInitEmbedding` is used by `AttentionModelPolicy`, while `VRPPContextEmbedder` is used by `AttentionModel`. | Have `VRPPContextEmbedder` compose or inherit from `VRPPInitEmbedding` for initial projections, eliminating key remapping `_KEY_MAP` in `loader.py`. | ~45 LOC | `logic/test/unit/models/subnets/test_embeddings.py` | low | open; proposed, not trialed |
| M-cursor-01 | SANS and NA re-implement `execute` instead of `_run_solver` | `logic/src/policies/route_construction/base/base_routing_policy.py:447-593`; `logic/src/policies/route_construction/meta_heuristics/simulated_annealing_neighborhood_search/policy_sans.py:104-125`; `…/dispatcher.py:81-159`; `logic/src/policies/route_construction/learning_algorithms/neural_agent/policy_na.py:61-150` | Seven of the nine retained adapters (`alns hgs bpc pg_clns psoma aco_hh swc_tcf`) only implement `_run_solver` and inherit validation, area-param load, subsetting, tour mapping and cost. SANS stubs `_run_solver` and re-does validate / `_load_area_params` / DataFrame prep / mandatory insertion in `execute_new`. NA overrides `execute` entirely (B-gemini-02/03 live there). Factory and registry do not duplicate each other: the factory only imports packages and looks up the registry. | Keep SANS/NA as execute overrides if they must, but call the existing `_validate_mandatory`, `_load_area_params` and `_compute_cost` (SANS already does the first two) and stop rebuilding a depot-prefixed DataFrame that `_create_subset_problem` already expresses. Do not merge SANS `execute_og` (DS-35 / Qwen). | ~40–60 net lines in the SANS dispatcher if it reused subset/map. Not trialed. | `logic/test/unit/policies/` SANS/NA tests if present; no test asserts the nine adapters share one execute body. | medium | open; proposed, not trialed |
| M-cursor-02 | VRPP training reward vs simulator profit | `logic/src/envs/tasks/vrpp.py:71-76`; `logic/src/pipeline/simulations/bins/base.py:378-385`; `logic/src/policies/route_construction/base/base_routing_policy.py:258-260`; `logic/src/constants/tasks.py:55-59` | Training `VRPP.get_costs` returns `neg_profit = length * cost_km - waste * revenue_kg` with legacy defaults `COST_KM=REVENUE_KG=1.0` and `waste` in $[0,1]$. Simulator profit is `(fill/100)*volume*density*revenue_kg - expenses*km` with plastic `0.5837` €/kg. The base policy already scales $R$ to € per 1% fill for the classical solvers. Three formulas, three unit systems. P-grok-01 is the logged `profit` vs `reward` split; this row is the env vs sim split. | Keep one helper (prefer `base_routing_policy`’s `revenue_scaled`) and have `VRPP.get_costs` multiply fraction waste by `volume*density` before `revenue_kg`, or document training as a normalised proxy and stop comparing its magnitude to sim profit. | ~20 lines plus call-site edits. Not trialed. | `logic/test/unit/envs/test_problems.py` asserts `neg_profit == -26` on a toy instance; no test compares env reward to a sim day log. | high | open; proposed, not trialed |
| M-cursor-03 | Scalar vs vectorized retained selectors | `logic/src/policies/mandatory_selection/selection_{last_minute,lookahead,service_level}.py`; `logic/src/policies/vector/selection/{last_minute,lookahead,service_level}.py` | Two copies of each retained selector. After `4c5ef0d0c` they agree on units (percent threshold vs fraction fill), relative lookahead days, and depot column (training prepends it). Remaining drift is packaging: service-level `horizon_days` default 1 (vector) vs action fallback 3 (B-cursor-01); lookahead “today” formula is shared. | Do not merge the copies in this round. After B-cursor-01, add a parity test (the existing `selector_regression.py`) that scalar 1-based IDs equal the vectorized mask on the same fills. A later extract into `mandatory_selection/base/` is optional. | 0 now; a parity test is ~40 lines. Merging would save ~150 LOC at high risk. | `selector_regression.py` covers DS-19/20/21. No scalar↔vector service-level parity test. | medium | open; proposed, not trialed |
| M-kimi-01 | MS-BPC-SP engine re-implements helper modules | `exact_and_decomposition_solvers/multi_stage_branch_and_price_and_cut_with_set_partition/ms_bpc_sp_engine.py:621,724,856,220,178,208,823,1284` ↔ `helpers/solvers_and_matheuristics/branching/pruning.py`, `lagrangian_relaxation/pre_pruning.py`, `search/column_generation.py`, `branch_and_price_and_cut/bpc_engine.py:182` | Seven functions copied from the shared helpers into the MS engine (AST + difflib similarity): `_perform_strong_branching` 1.000 verbatim, `_select_nodes_knapsack` 0.999, `_compute_lr_bound_at_node` 0.990, `_column_generation_loop` 0.960, `_apply_branching_to_master` 0.950, `_extract_forced_sets_from_constraints` verbatim-modulo-import, `_reset_master_constraints` 0.928. The MS pricing/Farkas steps, `_separate_cuts`, `_detect_cycles`, `_is_solution_integer`, `_apply_reduced_cost_edge_fixing` have DIVERGED (sim 0.12–0.55) and must NOT be merged blindly. | Import the seven from their helper homes (extend helpers where the 0.95+ copies gained small fixes, e.g. the D1 pricing/branching fixes, then have both engines call one implementation). Do not touch the diverged twins. | ~950–970 LOC | `logic/test` coverage of the MS engine paths is thin; run the MS policy unit tests + a small MS `test_sim` before and after. | high | open; proposed, not trialed |
| M-kimi-02 | Tour extraction from MIP solutions, 3 clusters | `exact_and_decomposition_solvers/branch_and_cut/bc.py:873` ↔ `integer_l_shaped_benders_decomposition/master_problem.py:398` (sim 0.47; docstring admits copying bc.py); `smart_waste_collection_two_commodity_flow/{gurobi.py:206-227, ortools_wrapper.py:180-205, pyomo_wrapper.py:257-275}` (arcos_ativos tracing, sim 0.74–0.86); `branch_and_bound/{mtz.py:552, dfj.py:123}` (~20 LOC verbatim adjacency walk) | Three independent copies of “walk active arcs/edges from the depot and cut routes at the depot” across the exact solvers. | One `_extract_routes_from_flow` util under `policies/helpers/` (signature: arc-flow dict or edge multiset → list of depot-closed routes); SWC additionally gets a shared `_trace_tcf_routes`. | ~130–160 LOC across clusters | No test asserts tour extraction parity across solvers; add one on a fixed instance. | low–medium | open; proposed, not trialed |
| M-kimi-03 | SWC-TCF model built 3× | `smart_waste_collection_two_commodity_flow/gurobi.py` (~110 lines), `ortools_wrapper.py` (~130), `pyomo_wrapper.py` (~200) | The whole two-commodity-flow formulation (x/f/h/g/k variables, capacity, flow balance, depot balances, ψ forcing, infeasible-retry) is re-declared per solver API. The divergence already shows (B-kimi-54 exists only in pyomo; B-kimi-55/56/57/58 are per-backend drift). | Shared arc/params prep (`_build_tcf_data` returning common structures) consumed by thin per-backend model builders. | ~60–100 LOC | The verifier’s cross-backend parity checks (`kimi_lane_d_swc_20260926.py`) cover the shared instance; extend to constraint-set equality. | medium | open; proposed, not trialed |
| M-kimi-04 | `MasterProblemSupport` stub declarations shadowed by the constraints mixin | `helpers/solvers_and_matheuristics/master_problem/problem_support.py:324-525` ↔ `master_problem/constraints.py` | All 11 same-named methods (`add_edge_clique_cut`, `add_subset_row_cut`, `add_capacity_cut`, `add_lci_cut`, `add_multistar_cut`, `add_sec_cut`, `remove_local_cuts`, `find_and_add_violated_rcc`, `_find_customer_components`, `count_set_crossings`, `has_artificial_variables_active`) exist as Protocol-style `...`-stub bodies with full docstrings on `MasterProblemSupport`; `VRPPMasterProblem(ConstraintsMixin, SupportMixin, MasterProblemSupport)` MRO makes the mixin the sole implementation, so the stubs are unreachable docstring duplicates. | Trim to one-line stubs or delete the shadowed names from `MasterProblemSupport` (keep the real mixin docs). | ~190 docstring lines (docs-only, zero executable risk) | None needed; pure deletion of shadowed declarations. | low | open; proposed, not trialed |

| M-mistral-01 | Yaml-link updaters | `logic/src/utils/target/ms_updater.py:41-177`; `logic/src/utils/target/ri_updater.py:41-177` | The two 177-line modules are byte-identical except for the regex (`(mandatory_selection:...)` vs `(route_improvement:...)`), the function names (`list_available_ms_strategies` vs `list_available_ri_improvers`), and docstring examples. Same `_resolve_stem`, same yaml read/rewrite, same `update_*` entry point. | Parameterise one module in `utils/target/` (field label + docstring nouns as arguments); `controllers/cli/tgt_parser.py:44-49` imports both today and becomes one import with two configured instances. | ~150 | None: no test imports `tgt_parser`, `ms_updater` or `ri_updater` (`grep logic/test` → zero). Adding one round-trip test before the merge is a precondition. | low | open; proposed, not trialed |
| M-mistral-02 | Config dataclass ↔ policy params re-declaration | `logic/src/configs/policies/bpc.py:22` (`BPCConfig`) ↔ `exact_and_decomposition_solvers/branch_and_price_and_cut/params.py:20,174` (`BPCParams.from_config` re-declares every field); `configs/policies/ms_bpc_sp.py:34` ↔ `ms_.../params.py:23` (18 dup windows @8); `configs/policies/egh.py:77` ↔ `matheuristics/exact_guided_heuristic/params.py:33` (22 @12); `configs/policies/lasm.py:107` ↔ `learning_.../params.py:45` (18) | Each policy keeps a `params.py` dataclass whose fields and multi-line docstrings re-state the matching `configs/policies/*.py` dataclass, then converts via `from_config`. Drift is already on record: Kimi's B-kimi-34 (alias trap), D-kimi-06 (`enable_hybrid_search` declared in config, dropped by `from_config`), and B-mistral-02 (the params/policy divergence itself) are all consequences of the two copies. | Make `params.py` classes hold or wrap the config dataclass (iterate `dataclasses.fields` in `from_config` instead of re-declaring), one pattern for all four; the config dataclass stays the single field authority. | ~300 (mostly duplicated docstrings/fields) | BPC unit tests cover construction; EGH/LASM are import-broken today (B-mistral-02) and must be fixed or deleted first; MS-BPC-SP has thin coverage. | medium | open; proposed, not trialed |
| M-qwen-01 | PG-CLNS operators/ duplicates helpers/operators/ | PG-CLNS `operators/destroy/{random,worst,shaw,string,cluster}.py`, `operators/repair/{greedy,regret,greedy_blink}.py`, `operators/move/{relocate,swap}.py`, `operators/exchange/or_opt.py`, `operators/route/{swap_star,two_opt_intra,three_opt_intra,two_opt_star}.py` (~20 files, ~2000 LOC) ↔ `helpers/operators/destroy_ruin/`, `helpers/operators/recreate_repair/`, `helpers/operators/intra_route_local_search/`, `helpers/operators/inter_route_local_search/` | PG-CLNS has its own complete operator suite that duplicates the shared `helpers/operators/` infrastructure. Key behavioral differences: PG-CLNS `worst_removal` lacks randomization parameter `p` (B-qwen-03); PG-CLNS `random_removal` uses index-based popping vs set-based filter (B-qwen-04); PG-CLNS `greedy_insertion` takes `R, cost_unit` params while helpers takes `noise`; PG-CLNS repair operators lack the noise parameter for ALNS-style clean/noisy slot expansion. | Switch PG-CLNS to import destroy/repair operators from `helpers/operators/`. Adapt the call sites to pass the shared operator signatures (add `noise` parameter forwarding, drop `R/cost_unit` in favor of the shared profit-aware variants). Keep PG-CLNS-specific route operators (swap_star, two_opt_intra, etc.) if they have different interfaces, but move common ones to `helpers/operators/`. | ~1500–2000 LOC (the PG-CLNS operators/ directory) | No test covers PG-CLNS operator parity with helpers/operators. Add a test that PG-CLNS destroy/repair produce the same output as the shared operators given the same seed and input. | high | open; proposed, not trialed |
| M-qwen-02 | SANS operators/ duplicates helpers/operators/ | SANS `operators/{swap,move,intra_swap,intra_move,inter_swap,inter_move}.py` (~6 files, ~600 LOC) ↔ `helpers/operators/intra_route_local_search/{swap,relocate}.py`, `helpers/operators/inter_route_local_search/` | SANS has its own intra/inter swap and move operators. The SANS versions are tightly coupled to the SANS solution representation (routes as lists with specific profit computation). The helpers/operators versions are more general. | Evaluate whether SANS operators can be replaced by `helpers/operators/` versions with adapter wrappers. If the SANS solution representation is too different, keep SANS operators but document the divergence. | ~400 LOC (if replaceable) | SANS tests cover the SANS-specific operators. | medium | open; proposed, not trialed |
| M-qwen-03 | ALNS engine copy in HMLNS | `meta_heuristics/adaptive_large_neighborhood_search/alns.py` (814 lines) ↔ `hybrid_memetic_large_neighborhood_search/alns.py` (498 lines per Mistral's cluster map) | Mistral's duplication cluster map identified this pair. The HMLNS copy was not deeply inspected in this pass, but the 498-line overlap at window 12 suggests significant code sharing. | If HMLNS's ALNS is a simplified copy, switch it to import from the canonical `adaptive_large_neighborhood_search/alns.py`. | ~400 LOC | Not inspected; HMLNS is not a retained policy. | medium | open; not trialed, per Mistral's cluster map |

## 4. Dead code (append rows)

Kind values: `dead` (no path from any entry point, registry or yaml) and
`test-only` (referenced only by tests).

| ID | Kind | Symbol / file | Importer + registry + yaml evidence | Local deletion + verification result | LOC | Status |
|---|---|---|---|---|---|---|
| D-grok-01 | dead | `logic/src/utils/configs/yaml_to_env.py` (`to_bash_value`, `load_yaml_env`, `deep_merge`) | `rg` over `*.py` finds the three names only in this file and in `logic/src/utils/configs/__init__.py`, which re-exports them. No Hydra `_target_` and no registry entry. Runtime config loading uses `config_loader.py`. | Deleted in the `1aa00c09c` worktree together with the re-export. `compileall` of `logic/src/utils/configs` succeeded and `from logic.src.utils.configs import load_config` succeeded. The worktree file was restored afterwards. Full `import_sweep` was not run. | 186 | open; import-checked, sweep not run |
| D-gemini-01 | dead | `GlimpseDecoder.project_fixed_context` (`logic/src/models/subnets/decoders/glimpse/decoder.py:108,361,369`) | Grepping `project_fixed_context` shows it is only assigned in line 108 and called in line 361 to set `fixed.graph_context`. No method in `GlimpseDecoder`, `AttentionDecoderCache`, or `one_to_many_logits` reads `fixed.graph_context`. | Verified in `gemini_lane_c_repro_20260926.py` that zeroing out `fixed.graph_context` produces 0.0 diff on decoder logit outputs. | ~15 LOC (+ eliminates $embed\_dim^2$ unused weights) | open; verified by Gemini |
| D-cursor-01 | dead | `logic/src/constants/simulation.py:129-136` `MAX_LENGTHS` | `rg MAX_LENGTHS logic --include='*.py'` hits only this definition and a *local* copy in `envs/generators/op.py:32`. No `from logic.src.constants… import MAX_LENGTHS`. Not in `constants/__init__.py`. Yesterday’s R-cursor-07 also named `CRITICAL_FILL_THRESHOLD`, `OPERATION_MAP`, `FS_COMMANDS`, `TQDM_COLOURS`; on `main` those four are live (HRL manager, CLI). Only `MAX_LENGTHS` remains unused. | Not deleted this pass. Import-checked only; `compileall` / import sweep not run for a deletion. | ~8 | open; import-checked, not deleted |
| D-kimi-01 | dead | `bpc_engine.py::_select_nodes_knapsack` (`logic/src/policies/route_construction/exact_and_decomposition_solvers/branch_and_price_and_cut/bpc_engine.py:182-312`) | No caller in the BPC engine (the “No node pre-selection” comment is accurate); the LIVE twin is the copy in `ms_bpc_sp_engine.py:1284` (lane-D-out but live policy). No test references. | Deleted in the `1aa00c09c` worktree together with the D-kimi-02..06 batch: `compileall` of `logic/src/policies` OK; targeted imports (BPC engine, pool, smoothing, ACO, SWC dispatcher) OK; BPC 3-node LCI case still returns obj 25.0. Restored afterwards. | 131 | open; deletion verified locally |
| D-kimi-02 | dead | `pricing/smoothing.py::dssr_pricing_wrapper` (`…/solvers_and_matheuristics/pricing/smoothing.py:334-428`) + its `if use_dssr:` call site (:306-317) | Reachable only when `use_dssr=True` is passed; the CG loop never passes it (column_generation.py:319-328,349-358) → wrapper unreachable. Also operates on the nonexistent `pricing_solver._ng_memory` (B-kimi-39). | Deleted (wrapper + call site, replaced by the direct `pricing_solver.solve(**solver_kwargs)`) in the same batch: compileall OK, imports OK, BPC 3-node obj 25.0. Restored. | ~100 | open; deletion verified locally; goes with B-kimi-39 |
| D-kimi-03 | dead | `master_problem/pool.py` replay path: `GlobalCutPool.apply_to_master` (:188-240), `_inject_multistar_cut` (:150-186), `CutInfo` (:23-37) | `apply_to_master` has zero non-comment callers (B-kimi-36); `_inject_multistar_cut`’s sole caller is `apply_to_master`; `CutInfo` has zero references anywhere. All `add_cut` archival except the SRI vectors (read at cutting_planes.py:440) is write-only today. | Deleted in the same batch: compileall OK, imports OK, BPC 3-node obj 25.0, `GlobalCutPool()` no longer exposes `apply_to_master`. Restored. | ~105 | open; deletion verified locally; final go/delete is the owner’s B-kimi-36 decision |
| D-kimi-04 | dead | Misc zero-caller symbols: `has_artificial_variables_active` (`master_problem/problem_support.py:312-321` stub and `:783-797`), `Label.is_feasible` (`pricing/labels.py:123-132`), `validate_tour` (`vrpp_model.py:223-254`), `compute_tour_cost` (`vrpp_model.py:279-288`), `_get_coords_from_model` (`branching/tree.py:144-161`), `BIG_M` (`master_problem/model.py:163`, assigned, never read) | grep over `logic/src` + `logic/test`: definitions only (doctests not executed — pyproject addopts lacks `--doctest-modules`). `tree.max_nodes` (tree.py:111,115) is written but never read (engines cap nodes themselves, B-kimi-34). | Deleted in the same batch: compileall OK, imports OK, BPC 3-node obj 25.0. Restored. | ~95 | open; deletion verified locally |
| D-kimi-05 | test-only | `SeparationEngine.separate` legacy (`separation/engine.py:198-243`) | Referenced only by `logic/test/unit/policies/test_separation_engine.py:265`; production callers use `separate_fractional` / `separate_integer`. | Not deleted (owner decides test-only rows). | ~46 | open; test-only |
| D-kimi-06 | dead | Dead config knobs: `BPCParams.max_cut_iterations` (`branch_and_price_and_cut/params.py:81`, yaml policy_bpc.yaml:98), `BPCParams.use_spatial_partitioning` (params.py:92), `BPCParams.knapsack_proc_selection` (params.py:70), `enable_hybrid_search` (`logic/src/configs/policies/bpc.py:101`, yaml:249 — not a `BPCParams` field, dropped by `from_config`) | No code reads any of them (grep). (`enable_dssr`, `dssr_max_iters`, `enable_reduced_cost_arc_fixing` are the same kind but stay with their B rows B-kimi-39/40 until the owner picks fix-or-delete.) | Fields + `enable_hybrid_search` config line removed in the same batch: compileall OK, `BPCParams.from_config(yaml dict)` still constructs. Restored. | ~7 | open; deletion verified locally |

| D-mistral-01 | dead | RL contextual-bandit helper cluster: `helpers/reinforcement_learning/agents/contextual/{__init__,base,gpcmab,linucb,thompson}.py`, `agents/contextual_bandits.py`, `evolution_cmab.py`, `features/context.py` (11 files) | No importer outside the cluster: the live RL-ALNS solver uses `agents/bandits*` and `features/state` (`learning_heuristic_algorithms/reinforcement_learning_adaptive_large_neighborhood_search/solver.py:42-66`), not these. `logic/test` has no imports of them (the live `pipeline/rl/meta/contextual_bandits.py` is a different module). Config twin `RLConfig.evolution_cmab`/`EvolutionaryCMABConfig` also unread (D-mistral-11). | Deleted as one batch in the `1aa00c09c` worktree: `compileall` OK; `import_sweep.py` green (the sweep's 23 FAILs are 4 test-only deletions + 20 pre-existing EGH/LASM/gen breakages, none in or via this cluster); `pytest logic/test/unit/{utils,policies}` 569 passed, 1 pre-existing Gurobi-license test-pollution failure identical on pristine `main`. Restored afterwards. | 1408 | open; deletion verified locally |
| D-mistral-02 | dead | `logic/src/policies/context/__init__.py` (re-export shim over `interfaces.context`) | No consumer imports `logic.src.policies.context` (grep for `policies.context` / `policies import context` → zero outside the file); consumers import `interfaces.context` directly. | Deleted in the same batch; same verification as D-mistral-01. Restored. | 55 | open; deletion verified locally |
| D-mistral-03 | dead | `helpers/operators/crossover_recombination/pattern_and_itinerary.py` | Dead twin of the live `pattern_and_itinerary_crossover.py` (imported by `hybrid_genetic_search_with_adaptive_diversity_control/policy_hgs_adc.py:24`). No importer of the bare module; 26 dup windows @12 between the twins. | Deleted in the same batch; same verification. Restored. | 98 | open; deletion verified locally |
| D-mistral-04 | dead | Orphaned exact-solver params: `exact_and_decomposition_solvers/{constraint_programming_with_boolean_satisfiability, logic_based_benders_decomposition, progressive_hedging, scenario_tree_extensive_form}/params.py` | The four policy classes are registered and live, but nothing imports their `params.py` (grep per module path → zero). NOT `exact_guided_heuristic/params.py`, which is live-but-broken (B-mistral-02). | Deleted in the same batch; same verification. Restored. | 350 | open; deletion verified locally |
| D-mistral-05 | dead | `meta_heuristics/simulated_annealing_neighborhood_search/heuristics/sans_opt.py` | Zero importers, tests included; the shipped SANS path uses `heuristics/{anneal,sans,sans_neighborhoods,sans_operators,sans_perturbations,sans_state}.py`. Lane E cross-check welcome. | Deleted in the same batch; same verification. Restored. | 86 | open; deletion verified locally |
| D-mistral-06 | dead | `logic/src/utils/functions/monkey_patch.py`; `logic/examples/` (`dr_alns_train.py` + `__init__.py`) | Zero importers for all three (grep incl. `logic/test`); `examples/dr_alns_train.py` is the only content of the package. The rest of the dr_alns cluster is test-only (D-mistral-09). | Deleted in the same batch; same verification. Restored. | 318 | open; deletion verified locally |
| D-mistral-07 | dead | `logic/src/utils/model/export_onnx.py` | Zero importers (grep incl. tests). It is also the only importer of the optional `gpu` extras `onnx` and `onnxsim` — deleting it makes both removable from `logic/pyproject.toml`. | Deleted in the same batch; same verification. Restored. | 422 | open; deletion verified locally |
| D-mistral-08 | dead | `logic/store/**` (10 files) and `logic/migrations/**` (2 files) | Zero references anywhere in the tree, justfile, `tools/`, CI configs or docs (`grep` for `logic.store`, `logic/store`, `logic.migrations`, `logic/migrations` → zero). The tracking DB it resembles is `tracking/database` (live via `tools/database/justfile`). `pyarrow` (tracking extras) is imported only by dead `logic/gen/export_for_studio.py`. | Deleted in the same batch; same verification. Restored. | 1143 | open; deletion verified locally; `gen_pruned_configs.py` is export-branch tooling |
| D-mistral-09 | test-only | dr_alns cluster: `envs/dr_alns.py`, `models/core/dr_alns/{dr_alns_solver,ppo_agent,ppo_trainer}.py`, `pipeline/rl/core/dr_alns.py` | Referenced only by `logic/test/unit/models/test_dr_alns.py`. No `_ALGO_REGISTRY` key, no `_POLICY_REGISTRY_SPEC` entry; `train.yaml:212` has an `rl.dr_alns` section and `configs/rl/core/dr_alns.py` (`DRALNSConfig`) stays reachable via `configs/rl/__init__` — config for a feature with no live consumer. This corrects my bus post, which listed the cluster as dead. | Import sweep after deletion: only `test_dr_alns.py` fails (`ModuleNotFoundError`), confirming test-only. Restored. | 2015 | open; test-only, owner decides; goes with D-mistral-11 (yaml section) and dep `gymnasium` |
| D-mistral-10 | test-only | `pipeline/features/eval/drift_detection.py`, `pipeline/rl/core/stepwise_ppo.py`, `utils/tasks/task_utils.py`, `utils/functions/boolmask.py`, `models/subnets/modules/{distance_graph_convolution,normalized_activation_function}.py`, `pipeline/rl/core/time_tracking.py`, `tracking/integrations/gradient_tracker.py`, `utils/tasks/training_utils.py`, `helpers/reinforcement_learning/ks_aco_qlearning.py`, `validation/debug_utils.py` | Each is imported only by `logic/test/**` (per-module grep). Note `boolmask.py`: AGENTS.md §6.1 names it as the masking utility for decoders, but no decoder imports it — they mask via `masked_fill`; the doc reference is stale either way. This corrects my bus post items 3 and 8 (drift_detection/stepwise_ppo/task_utils were listed dead; they are test-only). | Import sweep after deletion: exactly the four test files failed; restored. | ~2500 | open; test-only, owner decides |
| D-mistral-11 | dead | Config fields/keys no code reads: `SAConfig.iterations_per_temp` (+`policy_sa.yaml:36`, B-mistral-03); `TrackingConfig.{wst_tracking_uri,real_time_log,profiler_buffer_size}` (+7 yamls, B-mistral-04); `TrainConfig.{route_improvement_epochs, lr_route_improvement, efficiency_weight, overflow_weight, eval_only}` (+`enable_scaler`, `checkpoint_epochs`, fixture-only); `BCConfig.use_exact_separation`; `configs/rl/policies/hgs.py::{min_diversity,diversity_change_rate}`; `OptimConfig.lr_min_decay`; `AdaptiveKernelSearchConfig.time_limit_stage_1`; `GRPOConfig.group_size`; `HPOConfig.hop_range`; `MetaRLConfig` 19 `hrl_*/cb_*/morl_*` fields; `RLConfig.{gp_cmab,evolution_cmab}`; `MandatorySelectionConfig.bernoulli`; `RewardShapingConfig.rewards_size`; `ContextFeatureExtractorConfig.selection_threshold`; `MandatoryManagerSelectionConfig.manager_critical_threshold`; `DemonAlgorithmConfig.max_demon_credit` | Probe over all 2153 config dataclass fields (`mistral_lane_g_config_consistency_20260926.py`): 58 with zero readers. The first four groups re-verified by direct grep; the `MetaRLConfig`/probe tail rows are probe-only (string search, no deletion trial) and need one manual confirmation each before deletion. Full list: `mistral_config_out_20260926.txt`. | No deletion run (config-only). `enable_hybrid_search` stays with D-kimi-06. | ~60 fields | open; probe-verified subset, rest labelled |
| D-mistral-12 | dead (owner call) | `logic/gen/{export_for_studio,export_website_data,export_loss_landscape,gen_dataset_analysis,gen_presentation,gen_simulation_analysis}.py` | No recipe, code or CI invokes them; `app/src/gen/**` headers state the Studio ported this pipeline to TypeScript ("archived logic/gen pipeline"). Live remainders: `gen_paper_latex.py` (`tools/script/justfile:32`), `gen_dist_matrix.py` (B-mistral-01), `report_utils.py` (imported by `gen_paper_latex.py:96`). They are also not package-importable (`from report_utils import ...` is a script-style import; the sweep FAILs are pre-existing, not deletions). `pyarrow` and `python-pptx`/`docxtpl` deps hang off this cluster (§5.G). | Not deletion-trialled (owner call — the Studio port claim should be confirmed by the app lane). | 7106 | open; owner call |
| D-qwen-01 | dead | `simulated_annealing_neighborhood_search/heuristics/sans_opt.py` (86 LOC) | Zero importers confirmed by `grep -rn "import.*sans_opt\|from.*sans_opt" logic` → zero hits. Mistral's D-mistral-05 independently identified this. The shipped SANS path uses `heuristics/{anneal,sans,sans_neighborhoods,sans_operators,sans_perturbations,sans_state}.py`. | Not deletion-trialled (Mistral already did; see D-mistral-05). Cross-confirm from lane E. | 86 | open; cross-confirmed by Qwen (matches D-mistral-05) |

## 5. Per-agent sections

Each agent adds a section headed `## 5.<lane>. <Agent> — lane <X> — 2026-09-26`.
It lists:

- the files and papers the agent read;
- the IDs it filed;
- its disagreements;
- its questions for the owner.

Codex adds its review notes under its own section.

## 5.A. Codex — lane A — 2026-09-26

### First contribution: baseline lifecycle and actionable proposals

**Scope and snapshot.** This is an initial contribution, not a completed lane audit.
Code was read and reproduced at detached `1aa00c09c`. The shared checkout was
`c6facedc1` when this contribution began; `git diff 1aa00c09c HEAD --
logic/src/pipeline/rl/common` was empty. No production code was changed.
Rows filed: **3 P, 2 B, 1 M, 0 D**. Verified removal savings: **0 LOC**.
The M-row estimate is provisional and must not be counted as implemented savings.

**Sources read.** Today's bus/index, common and Codex briefs, previous report
§6.A.1 and §7; `bibliography/models/Attention_Model.pdf` §4 and §5 hyperparameters;
`common/baselines/{base,rollout,exponential,warmup,__init__}.py`,
`common/base/{module,data}.py`, baseline-related portions of `steps.py`,
`common/epoch.py`, `core/reinforce.py`, training baseline settings and VRPP
evaluation-graph settings. Existing tests inspected:
`logic/test/unit/pipeline/rl/common/test_baselines.py` and
`logic/test/unit/models/test_baselines_expanded.py`.

**Reproduction.** Script:
`.agent/cache/tools/codex_rollout_repro_20260926.py` (shared cache, git-ignored).
Run from the reviewed worktree, after initializing its `wsmart_bin_analysis`
submodule:

```bash
git worktree add --detach ~/.cache/wsr-review/codex 1aa00c09c
cd ~/.cache/wsr-review/codex
git submodule update --init logic/src/pipeline/simulations/wsmart_bin_analysis
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.cache/wsr-main-venv/bin/python \
  /home/pkhunter/Repositories/Doc/WSmart-Route/.agent/cache/tools/codex_rollout_repro_20260926.py
```

The temporary worktree was removed after verification; recreate it with the
commands above to repeat the checks. The shared evidence script remains on disk.

The script executes real module construction, dataset setup, epoch-end callbacks,
and the paired t-test. Synthetic three-instance data replaces graph generation;
expensive policy rollouts and the policy-copy call are mocked to expose callback
decisions. All assertions passed. It reproduces both new B rows and the DS-06
residual below. Control assertions accept a significant improvement and reject a
worse or non-improving candidate when valid comparison data is supplied. Thus the
sign/significance gate itself is not being reported as broken. Full training,
GPU evaluation and the repository test suite were not run for this initial
report contribution. Initial import failed before worktree submodule setup;
initializing the pinned submodule resolved it. Optional torch-geometric binary
warnings and a SciPy precision warning in the constant-difference negative
control did not prevent assertions from passing.

**Proposed implementation sequence.** Address B-codex-01 and B-codex-02 together
as a baseline-comparison lifecycle change. Keep reporting validation graphs fixed
and give baseline promotion its own seeded pool and matching environment. Add
behavioral tests for accepted/rejected promotions, missing data, graph/environment
pairing, and deterministic pool refresh. Then trial M-codex-01 with parity tests.
P-codex-03 needs documentation or a named preset, not an automatic default change.

**Prior-round reconciliation (do not create duplicate bugs).**

- DS-06 residual: `_init_baseline` now lifts saved kwargs correctly, but
  `module.py:204` constructs `WarmupBaseline(baseline, warmup_epochs)` without
  forwarding `exp_beta`. The new script requests 0.123 and observes 0.8 in
  `warmup_baseline.beta`. Existing `lane_a_regression.py` checks beta on an
  unwrapped baseline and warmup length separately, so it misses this combination.
  Proposed follow-up: explicitly forward the configured EMA beta and test the
  wrapped baseline. Keep this linked to DS-06 rather than assigning a new B ID.
- Prior §7's “DS-07 moot (critic removed)” applies to the export, not automatically
  to main: `common/baselines/__init__.py` still registers `critic`. Its earlier
  correctness finding needs main-specific reconciliation; it is not re-filed here.
- DS-09 (`train_time` instance order) remains prior-round work; no duplicate row.
- Honor D3, DS-15, DS-16, and D4. No new BPC exactness or dead-code claims are made.

**Cross-lane review.** At this contribution no other lane had filed signed rows
in this report. No cross-lane finding has therefore been independently confirmed.
Leave §6 unconsolidated until the other lanes report and their evidence is checked.
Remaining lane-A work includes full eval/config review, Appendix B checks,
loss/decoding duplication, and registry-aware dead-code verification.

**Owner decisions.** No new decision is needed to record these proposals.
Before implementation, settle whether the desired baseline comparison pool
represents the current curriculum stage or an explicitly configured fixed
distribution; do not implicitly pick the first reporting validation graph.

## 5.B. Grok — lane B — 2026-09-26

### First contribution: day loop, seeds, resume, and where results are written

**Scope and snapshot.** Initial contribution, not a completed lane audit.
Read and reproduced at detached `1aa00c09c` in `~/.cache/wsr-review/grok`.
`git diff 1aa00c09c HEAD` is empty for `logic/src/pipeline/simulations`,
`logic/src/pipeline/features/test`, `logic/src/pipeline/callbacks/simulation`,
`logic/src/tracking/logging`, `logic/src/utils/{infrastructure,data,input,configs}`,
`logic/controllers` and `main.py`. The shared checkout was `c6facedc1`.
No production code was changed. Rows filed: **3 P, 7 B, 2 M, 1 D**.
Verified removal: **186 LOC** for D-grok-01, limited to `compileall` plus an import
of `load_config` (the full import sweep was not run). M-row estimates are not savings.

**Sources read.** Today's bus and index, the common brief, the lane B brief,
previous report §6.A.1 and §7 (DS-13 through DS-18 and the D2/D7 rulings).
Simulation-framework paper (`paper.tex` problem dynamics, Eq. profit_function,
protocol, plastic coefficients, overflow paragraph). Code: `day_context.run_day`,
`actions/{fill,node_selection,route_construction,route_improvement,time_constraints,collection,logging}.py`,
`actions/base.py` `_flatten_config`, `states/{running,initializing,finishing}.py`,
`bins/base.py` `collect` / `_process_filling`, `simulator.py`, the test orchestrator
and `parallel_runner.py`, `results_handler.py`, `repository/base.py` area parameters.

**What already matches, so it is not re-filed.**

- Day order is fill, mandatory selection, route construction, route improvement,
  then collection, then logging (`day_context.py:715-723`). `TimeConstraintAction`
  sits between improvement and collection and returns immediately unless
  `problem=ctop`. The paper's reported experiment does not impose a shift limit.
- Collected kilograms are taken from `bins.collect` on the pre-collection levels
  (`bins/base.py:378-382`). Overflows and lost kilograms are computed in
  `load_filling`, which runs before the tour (`fill.py:38-42`).
- An empty mandatory set with `vrpp: false` returns `[0, 0]` (`route_construction.py:109-116`).
  The nine retained policy yamls set `vrpp: true`, so the solver still runs.
  D3 stays: `[0, 0]` remains the empty tour.
- DS-13 (empirical grid ids) and DS-14 (one result key) are already implemented
  on this tree: `initializing.py:475-479` builds the grid from routed ids, and
  `policy_result_key` is the display slug. They are not re-opened.
- DS-15 and DS-16 are settled. See P-grok-02 and P-grok-03.

**Reproduction.** `.agent/cache/tools/grok_lane_b_repro_20260926.py`. From the worktree:

```bash
cd ~/.cache/wsr-review/grok
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/.cache/wsr-main-venv/bin/python \
  /home/pkhunter/Repositories/Doc/WSmart-Route/.agent/cache/tools/grok_lane_b_repro_20260926.py
```

`PYTHONPATH` is required because the script lives outside the worktree, so
Python would otherwise put the tools directory on `sys.path`. All six checks
passed. No `test_sim`, training run, or multi-file pytest was started.

**Proposed implementation order.**

1. B-grok-01. One constructor token in the daily seed. This changes random streams
   for ALNS, ACO-HH, PSOMA and SANS. It does not change the waste sample while
   `noise_variance` is 0.
2. B-grok-05, then B-grok-02 and B-grok-03 together. Resume cannot skip finished
   samples until the dict key is the slug. The clock sign and the swallowed
   `CheckpointError` are the same checkpoint wrapper. Do not delete `checkpoints/`
   on `main`; the previous round kept it here.
3. B-grok-04 is the one-line companion to B-grok-03 on the parallel path.
4. B-grok-06 only matters when `stats_filepath` is set. B-grok-07 only matters
   for policy ids that contain the tokens `ms` or `ri`.
5. Trial M-grok-01 and M-grok-02 after the result keys and the failure exit are
   fixed, so a refactor does not preserve the zero-mean path.
6. D-grok-01 can land on its own. Re-run the import sweep before deleting it
   from `main`.

**Not signed as bugs.**

- `node_selection.py:165` reads `max_capacity` (default 100). The day context
  stores the truck capacity under `vehicle_capacity`. Lookahead, last-minute and
  service-level do not read `vehicle_capacity`. Capacity-aware selectors outside
  this lane do. Left unsigned until lane F says those selectors are in the run.
- CTOP still adds fill percent into a kilogram load at `collection.py:89`
  (old B-grok-08). The retained problem is `vrpp`, where that branch does not run.
- The per-day scenario tree is wasted work for the nine policies (M-grok-01).
  It is not dead code: other registered solvers read it.

**Still open in this lane.** Config flatten for psoma and sans dict yaml,
worker-shared mutable state, distance-matrix loaders, and action boilerplate.
Those are not in this contribution.

## 5.C. Agy (Gemini) — lane C — 2026-09-26

### First contribution: Attention Model fidelity, subnets, loader, and Neural Agent

**Scope and snapshot.** Initial lane audit contribution.
Read and reproduced at detached `1aa00c09c` in worktree `~/.cache/wsr-review/gemini`.
Scope covers:
- `logic/src/models/core/attention_model/**` (`model.py`, `policy.py`, `decoding.py`, `deep_decoder_policy.py`, `symnco_policy.py`);
- imported subnets: `subnets/decoders/glimpse/**`, `subnets/encoders/gat/**`, `subnets/encoders/common/**`, `subnets/embeddings/context/vrpp.py`, `subnets/embeddings/vrpp.py`;
- `logic/src/utils/model/loader.py` (shared with lane A);
- `logic/src/policies/route_construction/learning_algorithms/neural_agent/**` (`policy_na.py`, `agent.py`, `simulation.py`, `params.py`).

Rows filed: **3 P, 5 B, 3 M, 1 D**.
Estimated refactoring savings: **~475 LOC** across M-gemini-01..03. Verified dead code: **~15 LOC** (plus eliminates $embed\_dim^2$ unused weights) for D-gemini-01.

**Sources read.**
- Source papers: `bibliography/models/Attention_Model.pdf` (Kool et al. 2019 §3, §4, Appendix A); `bibliography/models/Pointer_Networks.pdf` (Vinyals et al. 2015).
- Model code: `models/core/attention_model/{policy,model,decoding,deep_decoder_policy,symnco_policy}.py`, `models/common/autoregressive/{policy,constructive}.py`.
- Subnet code: `subnets/decoders/glimpse/{decoder,attention}.py`, `subnets/encoders/gat/{encoder,gat_multi_head_attention_layer}.py`, `subnets/encoders/common/{encoder_base,multi_head_attention_layer}.py`, `subnets/embeddings/context/{base,vrpp}.py`, `subnets/embeddings/vrpp.py`, `subnets/modules/{normalization,connections}.py`.
- Environment & Agent: `envs/routing/vrpp.py`, `envs/tasks/vrpp.py`, `envs/tasks/base.py`, `envs/base/ops.py`, `utils/data/td_state_wrapper.py`, `policies/route_construction/learning_algorithms/neural_agent/{policy_na,agent,simulation,params}.py`.
- Existing tests: `logic/test/unit/models/test_models.py`, `logic/test/unit/models/subnets/test_deep_decoder.py`, `logic/test/unit/envs/test_problems.py`.

**Component-by-component paper fidelity (Kool et al. 2019 §3 & Appendix A):**
1. **Normalization type and placement:**
   - *Paper:* Batch Normalization (BN) placed after the residual skip connection ($\text{BN}(x + \text{MHA}(x))$, $\text{BN}(x + \text{FF}(x))$).
   - *Code:* `MultiHeadAttentionLayerBase` uses post-residual normalization (`SkipConnection` then `Normalization`). Default `norm_config` is `BatchNorm1d` (matching the paper).
   - *Defect:* `AttentionModelPolicy(normalization='layer')` silently ignores the kwarg and builds `BatchNorm1d` because `GraphAttentionEncoder` takes `norm_config` and swallows `normalization` into `**kwargs` (B-gemini-01).
2. **Feed-forward width:**
   - *Paper:* FF sublayer width 512 ($d_{FF} = 512$).
   - *Code:* `feed_forward_hidden` defaults to 512 in `GraphAttentionEncoder`. B-claude-04 already fixed `model.py:300` forwarding `hidden_dim` instead of hardcoding.
3. **$\sqrt{d_k}$ scaling:**
   - *Paper:* Glimpse scaled by $\sqrt{d_k} = \sqrt{d/M}$.
   - *Code:* `one_to_many_logits:63` computes `attn_scores / math.sqrt(key_size)` where `key_size = embed_dim // n_heads`. This is exactly $\sqrt{d/M}$.
   - *Final logits:* The paper computes single-head dot products scaled by $\sqrt{d}$; the code scales multi-head dot products by $\sqrt{d/M}$ and averages heads (`logits.mean(dim=1)`, P-gemini-02).
4. **Tanh clipping placement:**
   - *Paper:* Clipping with $C=10$ sits on the final logits before softmax: $u_{(c)j} = C \cdot \tanh(\cdot)$.
   - *Code:* `one_to_many_logits:95-99` applies `logits = torch.tanh(logits) * tanh_clipping`, then applies `logits.masked_fill(mask, mask_val)`. Tanh clipping sits on the logits, NOT on the glimpse attention scores. Masking is correctly placed after tanh clipping so invalid actions remain $-\infty$.
5. **VRPP context features:**
   - *Paper:* In Appendix A.2 (PCTSP), context is $[\bar{h}, h_{\pi_t}, \beta_t]$ where $\beta_t$ is required remaining prize.
   - *Code:* `VRPPContextEmbedder` fuses (1) current node embedding, (2) mean waste of unvisited nodes, (3) mean distance to unvisited profitable nodes (P-gemini-03). However, the global graph embedding $\bar{h}$ is omitted from the query despite being precomputed in `fixed.graph_context` (P-gemini-01, B-gemini-04, D-gemini-01).
6. **Depot mask rule:**
   - *Paper:* Disallow visiting the depot if already at the depot in the previous step (unless all nodes are visited).
   - *Code:* In `VRPPEnv._get_action_mask`, depot is blocked while pending mandatory bins remain (`mask[:, 0] = ~has_pending_mandatory`). When no mandatory bins remain, depot is legal (`mask[:, 0] = True`). Step transition in `OpsMixin._check_done` sets `done = (node == 0) & (steps > 0)`.
7. **Decoding strategies:**
   - Greedy (argmax) and multinomial sampling with temperature are implemented across `GlimpseDecoder._select_node` and `utils/decoding/{greedy,sampling}.py`. B-gemini-04 (CUDA generator) and B-gemini-05 (infinite sampling loop) were verified as resolved on this branch.

**The two model classes (`AttentionModelPolicy` vs `AttentionModel`):**
- **Parity:** Evaluated in `gemini_lane_c_repro_20260926.py` (`check_m_gemini_01_parity`). For identical weights, `policy.encoder(x)` and `legacy.encoder(x)` outputs match exactly (`max diff = 0.0`).
- **Architectural divergence:** `AttentionModelPolicy` is the RL4CO-standard training policy using `TensorDict`. `AttentionModel` is the legacy constructive policy used by `eval` and `NeuralAgent`. Both wrap `GraphAttentionEncoder` and `GlimpseDecoder`, but maintain separate initial embedders (`VRPPInitEmbedding` vs `VRPPContextEmbedder`) and incompatible forward return structures (`reward` vs `cost`/`reward`).
- **Migration proposal (M-gemini-01):** Make `AttentionModel` a thin adapter subclassing `AttentionModelPolicy`, translating `dict` inputs to `TensorDict` and mapping return keys. This eliminates ~350 LOC, removes the `_KEY_MAP` hack in `loader.py`, and eliminates dead weights in `model.context_embedder.project_step_context` (B-gemini-05).

**Neural Agent audit:**
- **State normalization:** `policy_na.py:119` divides `bins.c` by 100.0, scaling percent $[0, 100]$ to $[0, 1]$. This matches `VRPPGenerator` $[0, 1]$ training inputs.
- **Unit mismatch bug (B-gemini-03):** `policy_na.py:145-146` calculates `collected_revenue` by multiplying `float(bins.c[n - 1])` (which is percent, e.g. 75.0) directly by `revenue_kg`. In `bins/base.py:378`, true kilograms is `(bins.real_c / 100) * volume * density`. This corrupts the policy's returned profit metric.
- **Mandatory enforcement:** `_get_action_mask` unmasks mandatory nodes even if waste is 0 and blocks the depot until all mandatory nodes are visited. The model mask forces all mandatory nodes to be visited before termination is possible.
- **Depot framing bug (B-gemini-02):** When mandatory bins are configured but the set is empty, `simulation.py:76` returns `([0], 0, ...)`. Owner ruling D3 requires `[0, 0]` for all empty tours.

**Reproduction and verification.**
Evidence script: `.agent/cache/tools/gemini_lane_c_repro_20260926.py`. Executed in the `1aa00c09c` worktree:
- B-gemini-01: confirmed `AttentionModelPolicy('vrpp', normalization='layer')` constructs `BatchNorm1d`.
- B-gemini-02: confirmed `NeuralAgent.compute_simulator_day` returns `[0]` instead of `[0, 0]`.
- B-gemini-04 / D-gemini-01: confirmed `fixed.graph_context` is completely disconnected from logit computation (diff = 0.0 when zeroed).
- B-gemini-05: confirmed `model.context_embedder.project_step_context` is dead code.
- M-gemini-01: confirmed encoder output parity (diff = 0.0).
- Suite checks: `compileall -q logic` clean; `logic/test/unit/models/test_models.py` (19 passed in 0.36s); `logic/test/unit/models/subnets/test_deep_decoder.py` (4 passed in 0.11s).

**Proposed implementation order:**
1. Fix B-gemini-02: change `([0], 0, ...)` to `([0, 0], 0, ...)` in `simulation.py:76`. One-line fix adhering to owner ruling D3.
2. Fix B-gemini-01: forward `NormalizationConfig(norm_type=normalization)` explicitly in `policy.py:75`.
3. Fix B-gemini-03: align `policy_na.py:145` revenue calculation with `bins/base.py:378` physical kg conversion.
4. Remove dead code D-gemini-01 (`project_fixed_context` in `glimpse/decoder.py`).
5. Execute M-gemini-02 (deduplicate autoregressive decoding loop in `DeepDecoderPolicy`).
6. Execute M-gemini-01 and M-gemini-03 (unify `AttentionModel` with `AttentionModelPolicy` and consolidate embeddings).

## 5.F. Cursor — lane F — 2026-09-26

### First contribution: selectors, fast_tsp, BMC/OI, policy framework

**Scope and snapshot.** Initial contribution, not a completed lane audit.
Read and reproduced at detached `1aa00c09c` in `~/.cache/wsr-review/cursor`.
The shared checkout was `c6facedc1`. No production code was changed.
Rows filed: **4 P, 5 B, 3 M, 1 D**. Verified removal savings: **0 LOC**
(`MAX_LENGTHS` was not deleted this pass). M-row estimates are not savings.

**Sources read.** Today's bus and index, the common brief, the lane F brief,
previous report §5.F / §6.A.1 / §7 (DS-19/20/21, DS-41, B-cursor-01 √n wontfix,
D3, DS-15, DS-16). Kirkpatrick, Gelatt & Vecchi 1983 (`bibliography/policies/Simulated_Annealing.pdf`
and the duplicate `Simulated Annealing.pdf`) for BMC. `Old_Bachelor_Acceptance.pdf`
was not read: `old_bachelor_acceptance.py` is registered, and none of the nine
retained policies reach it (ALNS/PSOMA use BMC; HGS yaml lists OI; SANS has its
own schedule). `Travelling_Salesman_Problem.pdf` as wrapper background only.
Selector yamls (`ms_{lookahead,last_minute,service_level}.yaml`, `ri_ftsp.yaml`,
`ac_{bmc,oi}.yaml`, `policy_alns.yaml`) against the simulation-paper day order
(fill, then select).

Code: `mandatory_selection/selection_{lookahead,last_minute,service_level}.py` +
`base/eoq.py` + factory; `vector/selection/{last_minute,lookahead,service_level}.py`;
`route_improvement/fast_tsp.py` + `common/helpers.py` + `tsp.py::find_route`;
`acceptance_criteria/{boltzmann_metropolis_criterion,only_improving}.py`;
`route_construction/base/{base_routing_policy,base_multi_period_policy,factory,registry}`;
the nine adapters' `execute` / `_run_solver`; `configs/policies/other/mandatory_selection.py`;
`pipeline/simulations/actions/node_selection.py`; `envs/tasks/vrpp.py`;
`constants/{simulation,routing,tasks}.py`.

**What already matches, so it is not re-filed.**

- DS-19 (training prepends a depot column), DS-20 (last-minute threshold is
  percent 70/90; vectorized `LastMinuteSelector` divides by 100), DS-21
  (lookahead expansion is relative to `current_collection_day`). Reproduced
  in the script. Commits `4c5ef0d0c` / `ec3feeb5c`.
- B-cursor-01 (2026-09-25): keep linear $D\cdot k\cdot\sigma$, not $\sqrt{D}$.
  Both scalar and vectorized copies are already linear. P-cursor-04 records
  the yaml-vs-code percentile story only.
- D3: `BaseRoutingPolicy.execute` returns `[0, 0]` on an empty mandatory set.
  B-gemini-02 (`[0]` from NA) is lane C.
- DS-15 / DS-16: settled; not this lane.
- Factory vs registry: they compose. The factory imports the packages (side
  effect: registry populate) and looks up the class. Not a second registry.
- `BaseMultiPeriodRoutingPolicy` is unused by the nine (ALNS-IPO is not
  retained). Other registered solvers still use it, so it is not dead.
- `IDistanceMetric` and `JointSelectionConstructionContext` are used by
  non-retained but live policies. Not dead on `main`.
- Yesterday’s R-cursor-07 companions `CRITICAL_FILL_THRESHOLD`, `OPERATION_MAP`,
  `FS_COMMANDS`, `TQDM_COLOURS` are live on `main` (HRL manager, CLI). Only
  `MAX_LENGTHS` remains unused (D-cursor-01).
- Spatial distributions and extra pytorch datasets stay: they are reached by
  yaml / generators outside the export prune. Lane G owns the full
  reachability list.

**Adapter pattern for the nine policies.**

| Policy | Inherits `execute` | Implements `_run_solver` | Notes |
|---|---|---|---|
| `alns` | yes | yes | BMC via yaml; calibrates $T$ only if `start_temp==0` (B-cursor-02) |
| `hgs` | yes | yes | yaml lists OI; yesterday B-cursor-06 (never calls `accept`) is lane E |
| `bpc` | yes | yes | lane D |
| `pg_clns` | yes | yes | lane E |
| `psoma` | yes | yes | BMC via yaml |
| `aco_hh` | yes | yes | lane D |
| `swc_tcf` | yes | yes | lane D |
| `sans` | **no** (`execute` → `execute_new`) | stub | re-does validate / area params / DataFrame prep (M-cursor-01). `execute_og` is DS-35 / Qwen |
| `na` | **no** (full override) | unused on the sim path | B-gemini-02/03 live here |

**Reproduction.** `.agent/cache/tools/cursor_lane_f_repro_20260926.py`.
From the worktree:

```bash
cd ~/.cache/wsr-review/cursor
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/.cache/wsr-main-venv/bin/python \
  /home/pkhunter/Repositories/Doc/WSmart-Route/.agent/cache/tools/cursor_lane_f_repro_20260926.py
```

`PYTHONPATH` is required because the script lives outside the worktree.
All nine checks printed `PASS` and ended `ALL LANE F CHECKS PASSED`.
torch_geometric scatter/sparse warnings and a Gurobi academic-license
banner appeared; they did not fail assertions. No `test_sim`, training
run, or multi-file pytest. `selector_regression.py` was not re-run this
pass (DS-19/20/21 are covered in the first check).

**Proposed implementation order.**

1. B-cursor-01. Add `horizon_days: int = 1` to `ServiceLevelSelectionConfig`
   and change `node_selection.py:171` from `_pi(..., 3)` to `_pi(..., 1)`.
   This is the only new typed-path silent-wrong-result in the selectors.
   Archived `{file: variant}` sl_ftsp / sl_cls runs already keep yaml
   `horizon_days` and are unaffected.
2. B-cursor-02 / P-cursor-02. Coordinate with lane E (ALNS owns
   `start_temp==0`). Either ship `start_temp: 0` so Ropke–Pisinger
   calibration runs, or make `BoltzmannAcceptance.setup` set
   $T=w\cdot\|f\|/\ln 2$. Do not do both. Rewrite the inert
   `start_temp_control: 0.05` comment.
3. B-cursor-05. Document that `seed` is unused (DS-41 cannot forward it).
   Catch `find_tour` failures and return the input trip. Correct
   `ri_ftsp.yaml` (Held–Karp $n\le 20$ / randomised LS $n>20$, not
   “Christofides + 2-opt”). `SCALE` vs uint16 is a documentation item on
   this Linux build (wider `uint_fast16_t`); do not change the scale
   without a distance-matrix audit.
4. B-cursor-03 (docstrings) and B-cursor-04 (`fill_ratios` / `>=`) are
   comment and unused-argument cleanups. Safe anytime.
5. After B-cursor-01, add the scalar↔vector service-level parity test
   (M-cursor-03). Do not merge the two selector copies this round.
6. M-cursor-01 (SANS/NA `execute`) after Qwen finishes DS-35. M-cursor-02
   (env reward vs sim profit) is high-risk; document first, do not
   silently change training `get_costs`.
7. D-cursor-01 can land on its own after an import sweep.

**Confirmations / disagreements.**

- **Grok §5.B unsigned note:** confirmed. Lookahead, last-minute and
  service-level never read `vehicle_capacity`. The action stores
  `context.get("max_capacity", 100.0)` on the context (`node_selection.py:165`).
  Capacity-aware selectors (knapsack / Lagrangian / Whittle) do read it;
  they are registered and live, not retained. No B-row: the three retained
  selectors are fill-threshold policies, so they do not need the truck
  capacity. The `max_capacity` vs `vehicle_capacity` name split is a
  packaging note for Grok, not a defect in this lane.
- **P-grok-01:** agree. Logged `profit` is the paper identity. M-cursor-02
  is the *training* `VRPP.get_costs` (legacy `COST_KM=REVENUE_KG=1.0`,
  waste in $[0,1]$) versus that same sim formula, plus the base policy’s
  `revenue_scaled` (€ per 1% fill). Three formulas, not a dispute of
  P-grok-01.
- **B-gemini-02/03:** agree, and they sit on the NA `execute` override
  that M-cursor-01 names. Fix those two before any SANS/NA refactor.
- **HGS `accept()`:** still Qwen. Not re-filed (yesterday B-cursor-06).
- No disagreement with Codex or Gemini paper rows.

**Owner decisions needed.**

1. B-cursor-01 fallback: change the action default from `3` to `1`, or keep
   `3` and document it as the typed-path horizon? Recommend `1` (matches
   yaml SL1 and the vectorized factory).
2. B-cursor-02: calibrate BMC from the objective (`start_temp: 0` or
   `setup` computes $T$), or keep $T=100$ and rewrite the yaml comment so
   it no longer claims a 5% / $P=0.5$ rule?
3. B-cursor-05 must_go: should Fast-TSP insert mandatory nodes that the
   constructor omitted, or is “improve the given trip only” the contract?

**Still open in this lane.** Interfaces inventory (which protocols have
no retained implementer), data-generator yaml reachability, improver
factory `time_limit` 30 vs dataclass 2, and whether `base_multi_period_policy`
helpers should be called out as unused-by-nine (they are not dead). Those
are not in this contribution.

## 5.D. Kimi — lane D — 2026-09-26

### Contribution: BPC, SWC-TCF, ACO-HH + solver helpers — verification of the ported fixes and new findings

**Scope and snapshot.** Read and reproduced at detached `1aa00c09c` in
`~/.cache/wsr-review/kimi`; `git diff 1aa00c09c HEAD` empty over lane-D files.
The shared checkout was `c6facedc1`. No production code changed (one local
deletion trial, restored — see D-kimi-01..06). Rows filed: **14 P (one
withdrawn in-row), 26 B, 4 M, 6 D**.

**Sources read.** Today’s bus/index, common + lane-D briefs, previous report
§2 lane-D header, §5.D, §6.A.4.1, §6.A.5, §7.2 (DS-22…31, 40, 52, 53;
owner rulings D1/D3/D4). Papers (pdftotext extracts under
`~/.cache/wsr-review/papers/`): Barnhart, Hane & Vance 2000 (BPC; §3.1–3.2,
§4.1, §6 — full algorithm read); Barnhart et al. 1998 (B&P overview);
Ramos, Morais, Barbosa-Póvoa 2018 (SWC-TCF; §3.2.2 eqs (11)–(21));
Chen, Kendall & Vanden Berghe (ACO-HH; §III.B). `Branch-and-Cut.pdf` is a
scanned image (0 extractable text) — not used. Code: the three policy
folders, all of `helpers/solvers_and_matheuristics/` (30 files), yamls
`policy_{bpc,swc_tcf,aco_hh}.yaml`, `configs/policies/{bpc,swc_tcf,aco_hh}.py`,
`base/base_routing_policy.py`, `actions/route_construction.py`, and — for
reachability — `branch_and_price/`, `branch_and_cut/`, `branch_and_bound/`,
`multi_stage_branch_and_price_and_cut_with_set_partition/`,
`integer_l_shaped_benders_decomposition/`,
`adaptive_branch_and_price_and_cut_with_heuristic_guidance/`.

**Reproduction.** `.agent/cache/tools/kimi_lane_d_bpc_20260926.py` (all seven
checks), `kimi_lane_d_swc_20260926.py` (21 checks, 13 pass / 8 investigated),
`kimi_lane_d_aco_20260926.py` + two clock-debug scripts. All run from the
worktree under `flock heavy.lock` with `OMP/MKL_NUM_THREADS=1`.

**Ported fixes — verified FIXED on main.** DS-22 (3-node LCI case returns
exactly 25.0 with shipped yaml), DS-23 (LR pre-pruning gated on
`vehicle_limit is not None` and fleet-scaled, bpc_engine.py:611-647),
DS-24 (80-bin 4 s timeout returns 14 routes/obj 68.95 instead of an empty
day), DS-25 (non-OPTIMAL LP → `(None, {})` + raise, model.py:315-319,
column_generation.py:240-241,495-500), DS-26 (heuristic pricing confirmed
by one exact pass; UB prune gated on `converged and pricing_exhausted and
not timed_out`, column_generation.py:346-358,430-436), DS-27/28/29/30/40/52/53
(all three SWC backends return the brute-force optimum on the shared
instances; delta gone; one 6000 km constant; status-branching; float
`time_limit`; yaml objective rewritten to the implemented semantics),
DS-31 (gurobi + ortools re-solve once without forcing + WARN; pyomo
partially — B-kimi-53), plus an exactness spot-check **18/18 vs brute force**
(plain/mandatory/fleet, 5 nodes, seed 0) on top of the prior 375.

**Still open / new (headline).**
- **B-kimi-44** (new, major): `apply_perturb` drops `ctx.rng` — seeded
  reproducibility broken independent of the wall-clock η (parent re-verified
  hyper_operators.py:466 + perturb.py:40-41).
- **B-kimi-45**: seeded ACO-HH runs give 3 distinct route sets / 6 runs —
  worse than yesterday’s 2/6; two independent causes (η wall-clock,
  B-kimi-44).
- **B-kimi-53/54** (new, major): pyomo-only crashes — no-incumbent
  time-limit (`_has_solution` ignores incumbent existence) and
  cutoff-isolated depot (empty-sum `== 0` trivial Boolean, parent-verified).
- **B-kimi-46**: ACO-HH returns a 322 kg route at capacity 50.
- Residuals confirmed byte-identical to `70e660b03`: B-kimi-06/07/08/10/12/13
  (BPC dead keys / no-op cut engines / smoothing), B-kimi-21 (typed path
  drops engine-nested overrides), B-kimi-26/27/28/29/30/31/33 (ACO).
- New BPC items: B-kimi-34 (tree param-alias trap + DeprecationWarning per
  run), B-kimi-35 (strong branching stays disabled-broken), B-kimi-36
  (GlobalCutPool replay unwired), B-kimi-37/38 (MinCut violation identically
  0 + 3 more no-op cut families).
- SWC minors B-kimi-55…58 (gap policy per backend, empty-day shape,
  fleet fallback, multi-vehicle extraction).

**Disagreements / corrections (one line each).**
- **My earlier claim corrected:** B-kimi-36 originally said nothing populates
  `active_sec_cuts_local` — wrong: `PCSubtourEliminationCut` forms 2.2/2.3
  (`local_only=True`) populate it via `add_sec_cut(global_cut=False)` and
  `remove_local_cuts` genuinely cleans them per node. Only the global pool
  replay is unwired. Row fixed in place.
- **P-kimi-01 withdrawn:** I first filed “BPC silently runs best-first” —
  wrong: `getattr(params, alias, explicit_arg)` falls back to the explicit
  arg the engine passes, so shipped BPC does run DFS (paper-faithful). The
  row is marked withdrawn; the real defect (params-only trap + warning) is
  B-kimi-34, severity downgraded major→minor.
- **B-kimi-35 not re-triggered:** 8 plain 5-node instances with strong
  branching ON all matched brute force (y-branching fires first on most);
  the yaml-documented counterexamples (19.16 vs 19.96) stand as the
  evidence. Filed as “documented defect, keep off”.
- Confirmations: B-gemini-02/03 sit in the NA execute override (M-cursor-01
  territory); nothing in lane D disputes Codex/Grok/Gemini/Cursor rows.

**Cross-lane notes.**
- Grok: B-kimi-53’s pyomo `ValueError` propagates through the day loop →
  your `CheckpointError` swallow (B-grok-03) turns it into a zero-mean day;
  worth fixing together.
- Cursor: B-kimi-59 (typed path drops engine-nested overrides incl.
  `vrpp:false`) is the typed-config twin of yesterday’s B-kimi-21; affects
  any policy whose runtime config nests under the engine key.
- Qwen/lane E: ACO-HH determinism (B-kimi-45) and the operator-keys rows
  (B-kimi-47/48) are yours to confirm against the HVPL/ALNS rows; no
  overlap found with my files.
- Mistral/lane G: the whole `solvers_and_matheuristics` package is live only
  through the BPC/MS/BC/BP/BB/ILS/ABPC policy folders (all registered). The
  D-kimi batch (~440 LOC incl. DSSR wrapper + pool replay + knapsack
  selector) is deletion-verified locally; the full import sweep was not
  re-run after restore.

**Owner questions.**
1. BPC: wire `GlobalCutPool.apply_to_master` per node or delete the pool
   replay (B-kimi-36)? Same fix-or-delete for DSSR keys (B-kimi-39),
   arc-fixing key (B-kimi-40), smoothing (B-kimi-41) — delete is
   deletion-verified (D-kimi-02/03).
2. Strong branching (B-kimi-35): fix on a master copy, or delete the
   feature + keys? My probe suggests it rarely even fires before
   y-branching — deletion may be the honest option.
3. ACO-HH: is run-to-run reproducibility a requirement? (Fixes for
   B-kimi-44/45 are small and verified in the repro.) And wire
   `operators`/`sequence_length` or delete the keys (B-kimi-47/48)?
4. SWC-TCF: align the pyomo failure semantics with gurobi/ortools
   (B-kimi-53/54), and normalise empty days to `[0, 0]` (B-kimi-56)?
5. D-kimi-05 (`SeparationEngine.separate` legacy, test-only): keep for the
   test or delete both?

**Not done in this lane.** Brute-force re-verification of MS-BPC-SP and
BP/BC against their papers (out of scope — live policies); full-tree import
sweep after the deletion trial; running the nine-policy `test_sim` (report-only
round; my changes never touched the tree).

## 5.M. Muse — independent verifier (M/D rows) — 2026-09-26

Independent verification of the filed refactoring (`M-`) and dead-code (`D-`)
rows. No production code touched, no worktree (read-only in the shared
checkout), no heavy jobs. Every file cited below is byte-identical between the
review commit `1aa00c09c` and the shared checkout `c6facedc1` (`git diff
--stat 1aa00c09c HEAD -- <all cited files>` is empty), so findings transfer to
the review commit. Method: importer checks (`muse.search` over `logic/`,
plus repo-wide for shell consumers), registry/yaml checks, and side-by-side
reads with one measured `difflib` similarity. Rows I did not re-check are
untouched by this section — no opinion either way.

### Verdicts

| Row | Verdict | Evidence |
|---|---|---|
| D-gemini-01 (`project_fixed_context` dead in `GlimpseDecoder`) | **confirmed** | `glimpse/decoder.py:361` writes `fixed.graph_context`; `_get_log_p` (`:399`,`:414-423`) passes only `fixed.node_embeddings/glimpse_key/glimpse_val/logit_key` to `_get_parallel_step_context`/`one_to_many_logits`. No reader of `fixed.graph_context` in the glimpse decoder. (Sibling decoders `gat`, `mdam` do consume it — the row is correctly scoped to glimpse.) |
| D-cursor-01 (`MAX_LENGTHS` dead) | **confirmed** | Only hits in `logic/`: the definition (`constants/simulation.py:129-136`) and an independent local copy (`envs/generators/op.py:32,98-99`). No importer, no `logic/configs`/`app`/`infra`/`ci`/`tools` reference. |
| D-kimi-03 (pool replay path dead) | **confirmed** | `apply_to_master` (`pool.py:188`) has zero code callers — only docstring/comment mentions (`bpc_engine.py:47`, `ms_bpc_sp_engine.py:47`, `constraints.py:325`). `_inject_multistar_cut` is called only from `apply_to_master` (`:239`); `CutInfo` has no references outside `pool.py`. Agree with the B-kimi-36 framing: global replay is dead, node-local SEC handling is separate. |
| D-kimi-04 (misc zero-caller symbols) | **confirmed** | No code callers of `vrpp_model.validate_tour`/`compute_tour_cost` (envs `validate_tours` is a different method), `Label.is_feasible` (docstring example only; no `--doctest-modules` in `pyproject.toml`/`setup.cfg`, so docstrings do not execute), `_get_coords_from_model`, `has_artificial_variables_active` (both defs), or reads of `self.BIG_M`/`BIG_M` beyond assignment/annotation/docstring. (DeepACO `_compute_tour_cost` is a different method and is live.) |
| D-grok-01 (`yaml_to_env.py` dead) | **disputed — do not delete as filed** | The `*.py` importer check is correct, but the file has live **shell** consumers outside `logic/`: `desktop/linux/cli/{test_sim,train,hyperparam_optim,gen_data,evaluation,meta_train}.sh` and `desktop/linux/remote/slurm.sh` all run `python logic/src/utils/configs/yaml_to_env.py …` to materialise env vars, and `docs/modules/UTILS_MODULE.md:189-192` documents it as shell integration. Deleting the module breaks those scripts. Owner call: either keep the file (close the row), or re-scope it as "delete together with migrating the 8 shell call sites". |
| M-grok-01 (gate the per-day scenario-tree build) | **confirmed with a scoping caveat** | The build is unconditional: `route_construction.py:150-156` always runs `ScenarioGenerator.generate` (the "if configured or requested" comment at `:130` guards nothing). None of the nine retained adapters reads `scenario_tree` (full-`logic/src` grep; retained `hgs` is `hybrid_genetic_search/policy_hgs.py` registered `"hgs"`, not the `hgs_adc` reader). **Caveat:** live non-retained consumers exist and several *require* it (`hgs_adc` raises `ValueError` when the tree is `None`; also `alns_ipo`, `adp`, `abpc_hg`, `calm`, `cp_sat`, `ils_bd`, `lbbd`, `st_ef`, `ph`, `hna`, `nds_brkga`, knapsack selectors via `context.scenario_tree`, and `helpers/operators` multi-period/forward-looking paths). So the gate must be opt-**out** (build by default, skip only for adapters/selectors that declare they never read it), not opt-in for the retained nine — otherwise a retained-slice test_sim stays green while `hgs_adc` starts crashing. |
| M-kimi-01 (MS-BPC-SP copies of helper modules) | **confirmed (spot-checked)** | `_select_nodes_knapsack`: `bpc_engine.py:182-312` vs `ms_bpc_sp_engine.py:1284-~1415` — `difflib` ratio **0.9987** (131 vs 132 lines). Near-verbatim as claimed. Note the interaction with D-kimi-01: the BPC-side copy has no caller, so the cheapest half of this merge is deleting the BPC copy, not unifying it. Other six pairs not re-measured; no opinion. |
| M-kimi-04 (`MasterProblemSupport` shadowed stubs) | **confirmed (mechanism)** | `VRPPMasterProblem(VRPPMasterProblemConstraintsMixin, VRPPMasterProblemSupportMixin, MasterProblemSupport)` (`master_problem/model.py:43`): the constraints mixin precedes `MasterProblemSupport` in the MRO, so same-named stub bodies on the support class are unreachable. Trimming them to one-line stubs is docs-only, ~zero behavioural risk. |
| M-cursor-01 (SANS/NA re-implement `execute`) | **confirmed (structure)** | `policy_sans.py:75-101` `_run_solver` is an explicit stub (`return [[]], 0.0, 0.0`, "Not used"); `execute` (`:104`) dispatches to `dispatcher.execute_new`/`execute_og`. `policy_na.py:292` `_run_solver` is likewise an explicit stub ("Stub to satisfy BaseRoutingPolicy…"). The duplication premise holds; agree with the row's own caution (keep the overrides, reuse `_validate_mandatory`/`_load_area_params`/`_compute_cost`, do not merge `execute_og`). |

### Disagreements

1. **D-grok-01 (above):** `yaml_to_env.py` is not dead while `desktop/linux/cli/*.sh` and `desktop/linux/remote/slurm.sh` shell out to it. The lane-D dead-code definition (entry points, registries, yaml, re-exports) does not list shell entry points, but "no path from any entry point" should include them — a shell script that runs `python <file>` is an entry point. Recommend the owner either close D-grok-01 as wontfix or convert it into a migration row (module + 8 call sites together).
2. **M-grok-01 gating direction (above):** gate by opt-out, not by retained-list opt-in, because of the live non-retained readers. The row's LOC estimate (~28 lines) is unaffected.

### Questions for the owner

1. D-grok-01: keep `yaml_to_env.py` for the shell CLI, or migrate the shell call sites and then delete?
2. M-grok-01: is an opt-out registry flag (`scenario_tree: false` default true) acceptable, or should the build stay unconditional until the multi-period consumers are revisited?
3. D-kimi batch (D-kimi-01…04, 06): all confirmed at grep level by this verifier (no deletion trial run here — Kimi's worktree trial stands as the deletion evidence). Ready to promote subject to Codex review?

**Not done.** Full-tree import sweep after any deletion (heavy, needs the lock — left for the implementation phase); M-kimi-02/03, M-grok-02, M-gemini-01…03, M-codex-01 not re-checked; Qwen lane E and Mistral lane G rows were not on record when this section was written.

## 5.G. Mistral — lane G (whole-tree reachability, duplication, config consistency, dependencies) — 2026-09-26

### Method

Worktree `~/.cache/wsr-review/mistral` at detached `1aa00c09c`, shared
read-only venv, heavy jobs under `flock`. No production code modified; the
worktree was restored after every trial (`git status` clean).

- **Reachability:** adapted the export-era `reachability.py` into
  `mistral_lane_g_reachability_20260926.py` (registry-, `_target_`- and
  yaml-aware AST walk; ancestor-`__init__` modelling added, which the old
  script lacked — without it every package `__init__` shows as unreachable).
  Roots: `main.py` + `logic/controllers/**`, every `*Registry.register(`
  registrant, `_POLICY_REGISTRY_SPEC` importlib targets, the
  acceptance-factory dir scan, `_target_` in every yaml under
  `logic/configs`, the lazy `data/__init__.__getattr__`. Output:
  `mistral_reach_out_20260926.txt` (2144 modules; 1761 reachable; 134
  unreachable non-test before tooling-entry filtering). Cross-check
  `vulture --min-confidence 80`: 43 hits, mostly unused args; two real
  finds handed to lanes D/E: unreachable code after `raise` in
  `policy_cp_sat.py:131`, unreachable `else` in
  `gen/gen_simulation_analysis.py:1337`.
- **Duplication:** `dup_finder.py logic 12` and `logic_audit.py logic/src`
  (outputs `mistral_dup_out_20260926.txt`, `mistral_audit_out_20260926.txt`).
  The cluster map with lane assignments is on the bus (2026-09-26,
  "lane G duplication cluster map"); I filed M rows only for the
  `utils/**`/`configs/**` clusters (M-mistral-01/02).
- **Config consistency:** `mistral_lane_g_config_consistency_20260926.py`
  (output `mistral_config_out_20260926.txt`): 2153 config dataclass fields,
  58 with zero readers. No two `defaults:` blocks under
  `logic/configs/tasks/*.yaml` are identical (the common brief expected a
  duplicated list; hash comparison says otherwise). The
  "yaml restates dataclass defaults word for word" check was NOT completed
  systematically — treat that bullet as open; the config↔params duplication
  (M-mistral-02) is the closest verified relative.
- **Verification block for the deletion trials:** `compileall` green;
  `import_sweep.py` green for every deleted module (the sweep's 23 FAILs
  decompose into 4 test-file failures caused by my *test-only* deletions
  and 20 **pre-existing** EGH/LASM/`logic/gen` breakages, verified by
  direct import on the pristine shared checkout);
  `pytest logic/test/unit/{utils,policies}` 569 passed / 1 failed, with the
  identical failure on pristine `main` (a Gurobi-license test-pollution
  interaction in `test_ils_rvnd_sp_paper.py`, not related to the deletions).

### Findings summary

4 B / 2 M / 12 D rows filed. Roughly 3.9k LOC deletion-verified dead
(D-mistral-01..08), ~4.5k test-only (D-mistral-09/10, owner decides),
~7.1k owner-call (`logic/gen`, D-mistral-12).

**Two corrections to my own early bus post** (evidence above):
`drift_detection`, `stepwise_ppo`, `task_utils` and the dr_alns production
modules have unit tests — they are test-only (D-mistral-09/10), not dead.
My first importer greps dropped test files whose names end in
`<module>.py` (the `grep -v` pattern ate `test_<module>.py` too).

**Pre-existing import breakage on main** (B-mistral-02): registered
policies EGH and LASM cannot be imported at all — `class PipelineParams`
exists nowhere; `exact_guided_heuristic/dispatcher.py:31` still imports it.
Because neither policy module is pulled in by its parent package
`__init__`, nothing crashes at startup — they are silently
never-registered. The `logic/gen` sweep failures are different: the
scripts use script-style `from report_utils import ...` imports and are
only runnable as files, which the sweep flags on any tree.

### Dependencies (`logic/pyproject.toml`), grep evidence = zero imports of the package anywhere under `logic/` or `main.py`

| Package | Group | Evidence | Note |
|---|---|---|---|
| `einops` | runtime | 0 imports | models use plain torch |
| `pydantic` | runtime | 0 imports | configs are dataclasses |
| `ml-dtypes` | runtime | 0 imports | |
| `latex2mathml` | runtime | 0 imports | not even the gen/presentation scripts use it |
| `onnx`, `onnxsim` | gpu extras | only importer is dead `utils/model/export_onnx.py` (D-mistral-07) | go with D-mistral-07 |
| `gymnasium` | gpu extras | only importer is test-only `envs/dr_alns.py` (D-mistral-09) | goes with the owner's test-only ruling |
| `triton`, `torch-tb-profiler`, `nvidia-ml-py` | gpu extras | 0 direct imports | runtime/implicit use plausible — keep, just noting |
| `dehb` | hpo extras | 0 imports (optuna is used, 12 files) | |
| `litlogger`, `sqlalchemy-utils`, `opentelemetry-proto` | tracking extras | 0 imports; tracking DB uses stdlib `sqlite3` (`tracking/database/commands.py:27`) | |
| `pyarrow` | tracking extras | only importer is owner-call `logic/gen/export_for_studio.py` (D-mistral-12) | goes with D-mistral-12 |
| `python-pptx`, `docxtpl` | runtime | only importers: owner-call `logic/gen/gen_presentation.py` + dead `logic/store` (D-mistral-12/-08) | go with D-mistral-12/-08 |
| `openrouteservice` | geo extras | 0 imports (osmnx/googlemaps/etc. are used) | |
| `pyproj` | geo extras | 0 direct imports | **keep**: `geopandas` requires it transitively |

Also: AGENTS.md/badges say PyTorch 2.2.2 while `logic/pyproject.toml` pins
`torch==2.13.0` — a docs-only inconsistency (low).

### Questions for the owner

1. B-mistral-02: fix the one-line EGH import and wire EGH/LASM into their
   package `__init__`, or delete both policies (they are registered-in-yaml
   but never loadable)?
2. D-mistral-09/10 (test-only, ~4.5k LOC incl. the dr_alns cluster, its
   `rl.dr_alns` yaml section and `DRALNSConfig`): keep as experiment
   scaffolding, or delete with their tests?
3. D-mistral-12: confirm the Studio fully replaced the `logic/gen`
   analysis/export scripts before deleting ~7.1k LOC.
4. Dependencies above: approve removing the zero-import packages
   (`einops`, `pydantic`, `ml-dtypes`, `latex2mathml`, `dehb`, `litlogger`,
   `sqlalchemy-utils`, `opentelemetry-proto`, `openrouteservice`; onnx/onnxsim
   and pyarrow/pptx/docxtpl/gymnasium go with their D rows)?
5. B-mistral-03/04 + D-mistral-11: for each unread config field, wire it or
   delete it? The BPC/SA/tracking ones are exposed in shipped yaml, so
   silently deleting changes documented behaviour — my default proposal is
   delete key + field + comment together.

### Not done

The "yaml restates defaults word for word" sweep (see Method); the
`MetaRLConfig` probe tail of D-mistral-11 needs per-field confirmation;
`test_sim` smoke was not run (no runtime-path code was touched — every
deleted module had zero importers, so there is no runtime path through
them to smoke).

-- Mistral

## 5.E. Qwen — lane E (ALNS, HGS, PG-CLNS, PSOMA, SANS + helpers/operators + helpers/local_search) — 2026-09-26

### Contribution: operator duplication map, paper fidelity for the five metaheuristics, PG-CLNS/HVPL gap analysis

**Scope and snapshot.** Read at `c6facedc1` (shared checkout, advanced past `1aa00c09c`; all cited files byte-identical between the two commits per Muse's verification). No production code modified, no worktree created, no heavy jobs run. Rows filed: **5 P / 4 B / 3 M / 1 D**.

**Papers read.** ALNS: Ropke & Pisinger (2006) — scoring, roulette weights, segment-based learning, SA acceptance, noise. HGS: Vidal (2022) — biased fitness, OX crossover, Split, penalty adaptation, survivor selection, SWAP*. PSOMA: Liu et al. (2006) — discrete PSO with swap velocity, memetic SA local search. SANS: simulated annealing neighborhood search paper. PG-CLNS: HVPL (Sun et al. 2023) + VPL base, with ALNS and ACO papers as secondary references for the adaptive destroy/repair and pheromone guidance. Operator papers: greedy/random/worst/route removal, Shaw removal, regret-k insertion, cyclic transfer, swap*, string removal.

**Code read.** Full ALNS solver (`alns.py`, 814 lines), HGS solver (`hgs.py`, 701 lines), PG-CLNS solver + params + ACO + LNS + local_search + all operators (~35 files), PSOMA solver + particle + params, SANS solver + all operators + heuristics (~40 files), `helpers/operators/` directory tree (106 files), `helpers/local_search/` (7 files), HVPL solver (583 lines) for comparison.

### Top findings

1. **PG-CLNS is not a faithful HVPL implementation (P-qwen-01, major).** The in-tree `HVPLSolver` implements the full three-phase structure: ACO initialization, VPL+HGS evolution (teams, seasons, positions, coaching/substitution/learning, promotion/relegation), ALNS refinement. PG-CLNS is a simplified ACO+LNS hybrid: population init by ACO construction, "coaching" = LNS per member, global pheromone update on best, replacement of weakest. Missing: VPL team/season structure, position-based phases, promotion/relegation, HGS genetic operators. The docstring says "Run the HVPL algorithm" but does not cite HVPL.

2. **PG-CLNS has its own operator suite duplicating helpers/operators/ (M-qwen-01, ~2000 LOC).** PG-CLNS `operators/` contains ~20 files with destroy, repair, move, exchange, and route operators that duplicate the shared `helpers/operators/` infrastructure. Key behavioral differences: `worst_removal` lacks the Ropke & Pisinger randomization parameter `p` (B-qwen-03); `random_removal` uses index-based popping vs set-based filter (B-qwen-04); `greedy_insertion` takes `R, cost_unit` params while helpers takes `noise`; repair operators lack the noise parameter for ALNS-style clean/noisy slot expansion.

3. **PG-CLNS uses `time.process_time()` instead of `time.perf_counter()` (B-qwen-01, P-qwen-02).** All four PG-CLNS modules (pg_clns.py, lns.py, local_search.py, aco.py) use CPU time for time-limit checks. This is inconsistent with DS-15 and every other retained policy.

4. **PSOMA breaks seeded reproducibility (B-qwen-02).** PSO velocity update uses `np.random.rand()` (global numpy state) instead of the seeded `self.random`. Two runs with the same seed produce different velocity updates.

### Paper fidelity details

**ALNS (Ropke & Pisinger 2006).** Faithful implementation:
- Scoring: σ₁=33 (new global best), σ₂=9 (improving, not visited), σ₃=13 (worsening accepted, not visited). σ₁ is awarded without "not visited" qualifier (P-qwen-03); the paper is ambiguous here.
- Weight update: segment-based, `w_{i,j+1} = w_{i,j}(1-r) + r * (π_i / θ_i)` with r=0.1, segment_size=100. Correct.
- Roulette-wheel selection: Eq. 20. Correct.
- Worst removal: randomized with `floor(y^p * |L|)`, p=3.0. Correct.
- Shaw removal: randomized with relatedness function. Correct.
- Clean/noisy repair pairs: separate operator slots (paper Conf. 15). Correct.
- SA acceptance: injected via `acceptance_criterion` (BMC by default). Correct.
- Profit-aware operators: optional flag for ablation. Sound adaptation.
- Mandatory bins: not explicitly protected in destroy operators (a destroy can remove mandatory nodes). The repair then re-inserts them via `mandatory_nodes` parameter in greedy/regret insertion. This is the standard ALNS approach.

**HGS (Vidal 2022).** Faithful implementation:
- Biased fitness: cost rank + diversity rank with `diversity_weight = max(0, 1 - nb_elite/pop_size)`. Correct.
- Broken pairs distance: `1 - |E(A) ∩ E(B)| / max(|E(A)|, |E(B)|)`. Correct.
- Survivor selection: clone removal first, then worst fitness. Correct.
- Penalty adaptation: monitors offspring feasibility rate (CVRP) or coverage/margin quadrant (VRPP). Sound adaptation.
- OX crossover: `route_profit_gpx_crossover` (selective route exchange with profit bias). Sound adaptation.
- Split: `LinearSplit` with VRPP skip edges for unprofitable nodes. Correct.
- Acceptance criterion: injected but only used for `step()` tracking, not for solution acceptance. This is correct for a population-based method — HGS does not need SA acceptance.
- Mandatory bins: preserved in Split via `mandatory_nodes` parameter. Correct.

**PG-CLNS (HVPL-inspired).** Major gaps (P-qwen-01):
- No VPL team/season structure. Population is a flat list.
- No position-based coaching/substitution/learning. "Coaching" = LNS per member.
- No promotion/relegation. Replacement = sort by profit, replace weakest with new ACO constructions.
- No HGS genetic operators (no OX crossover, no Split-based decoding).
- Pheromone: global update only (P-qwen-04). No local update during construction.
- Uses `time.process_time()` (B-qwen-01).

**PSOMA (Liu et al. 2006).** Mostly faithful:
- PSO velocity: discrete swap-based with inertia ω, cognitive c₁, social c₂. Correct structure.
- Memetic SA local search: `_sa_search` with Metropolis acceptance on surrogate distance + true profit. Correct.
- Operator adaptation: training phase evaluates each operator, non-training phase selects by reward probability. Sound.
- Bug: uses `np.random.rand()` for PSO velocity (B-qwen-02).

**SANS.** Self-contained implementation:
- Has its own operator suite (`operators/`) for swap, move, intra/inter variants.
- Neighborhood selection: random, greedy, consecutive strategies.
- Annealing: standard SA with temperature decay.
- Operators are tightly coupled to SANS solution representation.

### Operator duplication map

**Key finding: three of five policies do NOT import from `helpers/operators/`.**

| Policy | Imports from `helpers/operators/` | Has own operators/ |
|---|---|---|
| ALNS | ✅ Yes (destroy_ruin, recreate_repair, etc.) | No |
| HGS | ✅ Yes (crossover_recombination) | No |
| PG-CLNS | ❌ No | ✅ Yes (~20 files, ~2000 LOC) |
| PSOMA | ❌ No (uses HGS Split only) | No (uses inline `_swap`, `_insert`, `_inverse`) |
| SANS | ❌ No | ✅ Yes (~8 files, ~600 LOC) |

**PG-CLNS operators vs helpers/operators — behavioral comparison:**

| Operator | PG-CLNS local | helpers/operators | Behavioral difference |
|---|---|---|---|
| `random_removal` | Index-based pop | Set-based filter | Different removal ordering for same seed |
| `worst_removal` | One-shot sort+take | Randomized with `p`, iterative | PG-CLNS is deterministic, no `p` parameter |
| `shaw_removal` | Local implementation | Shared with randomization | Not deeply compared |
| `greedy_insertion` | Takes `R, cost_unit, expand_pool` | Takes `noise, mandatory_nodes, expand_pool` | Different signatures; PG-CLNS has profit check, helpers has noise |
| `regret_insertion` | Local implementation | Shared with noise | Different signatures |

**SANS operators vs helpers/operators:**

| Operator | SANS local | helpers/operators | Notes |
|---|---|---|---|
| `swap_1_route` (intra) | SANS-specific | `intra_route_local_search/swap.py` | SANS version coupled to SANS profit computation |
| `swap_2_routes` (inter) | SANS-specific | `inter_route_local_search/` | Same |
| `move` (intra/inter) | SANS-specific | `intra_route_local_search/relocate.py` | Same |

### Cross-lane notes

- **Cursor (B-cursor-02, P-cursor-02):** ALNS `start_temp==0` calibration is the correct path. The shipped yaml has `start_temp: 100` which skips calibration. ALNS code is correct; the yaml is the problem.
- **Kimi (B-kimi-44, B-kimi-45):** ACO-HH reproducibility is broken by wall-clock η + rng drop. PG-CLNS ACO also uses wall-clock timing (`time.process_time()`) but does not have the same η visibility issue (PG-CLNS ACO uses distance-based heuristic, not time-based).
- **Grok (B-grok-01):** ALNS seed is affected by the canonical name issue. ALNS under `last_minute` shares seed with HGS under `last_minute`.
- **Mistral (D-mistral-05):** Confirmed `sans_opt.py` is dead. Cross-confirmed as D-qwen-01.
- **Gemini (B-gemini-02, B-gemini-03):** NA empty tour and revenue calculation bugs are in the NA execute override, not in the five metaheuristics.

### Owner questions

1. **PG-CLNS HVPL fidelity (P-qwen-01):** Is PG-CLNS intended to be a faithful HVPL implementation, or is the simplified ACO+LNS hybrid the intended design? If the former, it needs VPL population dynamics, HGS genetic operators, and ALNS refinement. If the latter, the docstring should not say "Run the HVPL algorithm" and should cite the actual design.
2. **PG-CLNS operator consolidation (M-qwen-01):** Should PG-CLNS switch to `helpers/operators/` (saving ~2000 LOC but requiring signature adaptation), or keep its own operators (accepting the behavioral divergence)?
3. **PSOMA reproducibility (B-qwen-02):** Is seeded reproducibility required for PSOMA? The fix is small (replace `np.random.rand()` with seeded RNG).
4. **PG-CLNS timing (B-qwen-01):** Replace `time.process_time()` with `time.perf_counter()` to match DS-15?

### Not done

- Deep comparison of PG-CLNS shaw/string/cluster removal vs helpers/operators versions (only random and worst were compared side-by-side).
- SANS operator behavioral comparison (only structural duplication mapped).
- HMLNS ALNS copy inspection (M-qwen-03, per Mistral's cluster map).
- Deletion trials for M-qwen-01/02 (high risk, needs test_sim).
- `test_sim` smoke for PG-CLNS/PSOMA/SANS (no heavy jobs in this pass).

-- Qwen

## 5.O. OpenCode — independent verifier (M/D rows) — 2026-09-26

### First contribution: M/D verification + proposed edit list

**Scope and snapshot.** Independent verifier, same role as the 2026-09-25
Batch P pass (§5.O there). Read and statically verified at detached
`1aa00c09c` in `~/.cache/wsr-review/opencode` (submodule initialised,
no production edits, no heavy jobs, worktree kept for review Q&A).
All cited files below were diffed `1aa00c09c..c6facedc1` for the
verified rows — empty except where noted — so findings transfer to the
review commit. Method: `rg` importer/registry/yaml checks, one direct
import probe (EGH), `difflib` similarity for one M row, MRO read for
one M row. Rows not listed here are untouched — no opinion either way.

**Verdicts (first batch).**

| Row | Verdict | Evidence |
|---|---|---|
| D-grok-01 (`yaml_to_env.py` dead) | **disputed — agree with Muse, do not delete as filed** | The `*.py` importer check is correct, but the file has live shell consumers: `desktop/linux/cli/{train,test_sim,gen_data,evaluation,hyperparam_optim,meta_train}.sh` + `desktop/linux/remote/slurm.sh` (7 files) all run `python logic/src/utils/configs/yaml_to_env.py …`. Same conclusion as §5.M via an independent grep. Owner call stands: keep, or migrate module + call sites together. |
| B-mistral-02 (EGH/LASM import-broken) | **confirmed** | `dispatcher.py:31` does `from .params import PipelineParams`; `params.py:33` defines only `class ExactGuidedHeuristicParams` (`grep "class PipelineParams" logic` → zero). Direct import probe on the worktree fails with `ImportError: cannot import name 'PipelineParams'`. `matheuristics/__init__.py` has no EGH/LASM mention, so both stay silently unregistered. |
| D-gemini-01 (glimpse `project_fixed_context` dead) | **confirmed** | `glimpse/decoder.py:361` writes `fixed.graph_context` into `AttentionDecoderCache`, but `_get_log_p` (`:399`) passes only `fixed.node_embeddings/glimpse_key/glimpse_val/logit_key` to `_get_parallel_step_context`/`one_to_many_logits`. No reader of `fixed.graph_context` anywhere under `subnets/decoders/glimpse/`. (Sibling `gat` decoder consumes its own graph context at `gat/decoder.py:353` — row correctly scoped to glimpse.) |
| D-cursor-01 (`MAX_LENGTHS` dead) | **confirmed** | Only hits: definition `constants/simulation.py:129` and an independent local copy `envs/generators/op.py:32,98-99`. No importer of the constants symbol. |
| D-kimi-03 (pool replay path dead) | **confirmed** | `apply_to_master` (`pool.py:188`) has zero code callers — only docstring/comment mentions (`bpc_engine.py:47`, `ms_bpc_sp_engine.py:47`, `constraints.py:325`). `_inject_multistar_cut` is called only from `apply_to_master`; `CutInfo` has no references outside `pool.py`. |
| D-mistral-05 / D-qwen-01 (`sans_opt.py` dead) | **confirmed (third witness)** | `grep sans_opt logic` → zero importers, tests included. Shipped SANS path uses `heuristics/{anneal,sans,…}`. Agree with both lanes. |
| D-kimi-01 (BPC `_select_nodes_knapsack` dead) | **confirmed** | BPC-side def at `bpc_engine.py:182` has no callers; the live twin `ms_bpc_sp_engine.py:1284` is called at `:1510`. Cheapest half of M-kimi-01 is deleting the BPC copy. |
| D-mistral-04 (orphaned exact-solver `params.py` ×4) | **confirmed** | CP-SAT, LBBD, progressive hedging, scenario-tree-EF `params.py` have zero importers each; policy classes are registered and live, only the params modules are orphaned. (EGH `params.py` is live-but-broken per B-mistral-02, correctly excluded.) |
| D-mistral-08 (`logic/store` + `logic/migrations` dead) | **confirmed at grep level** | Zero hits for `logic.store`, `logic/store`, `logic.migrations`, `logic/migrations` across `logic tools ci docs app infra justfile`. No deletion trial run here — Mistral's trial stands as deletion evidence. |
| D-mistral-10 (`boolmask.py` test-only) | **confirmed** | Only importers are `logic/test/**` (`test_boolmask.py`, `test_boolmask_properties.py`). No decoder imports it (they mask via `masked_fill`). AGENTS.md §6.1 reference is stale either way. Test-only, owner decides. |
| M-mistral-01 (yaml-link updaters dup) | **confirmed (measured)** | `ms_updater.py` vs `ri_updater.py`: both 177 lines, `difflib` ratio 0.75 — byte-similar except field regex, function names, docstring nouns, as filed. No test imports either module. Merge precondition (round-trip test) agreed. |
| M-kimi-04 (shadowed `MasterProblemSupport` stubs) | **confirmed (mechanism)** | `VRPPMasterProblem(ConstraintsMixin, SupportMixin, MasterProblemSupport)` (`model.py:43`): constraints mixin precedes the support class in MRO, so same-named stub bodies (e.g. `add_lci_cut` at `problem_support.py:380` vs `constraints.py:210`) are unreachable. Docs-only trim, ~zero behavioural risk. |
| M-grok-01 (gate per-day scenario-tree build) | **confirmed with Muse's opt-out caveat** | Build at `route_construction.py:150-156` is unconditional; none of the nine retained adapters reads `scenario_tree`, but live non-retained consumers require it (`hgs_adc` reads `problem.scenario_tree` at `policy_hgs_adc.py:128`; `base_multi_period_policy.py:61` documents the contract). Gate must be opt-out, not retained-list opt-in. |
| M-qwen-01 (PG-CLNS `operators/` duplicates `helpers/operators/`) | **confirmed (structure)** | Zero hits for `helpers/operators` under the PG-CLNS policy dir, and zero hits for shared-operator imports under the SANS dir either — both suites are self-contained, as filed. Behavioural merge risk (B-qwen-03/04 signature divergence) agreed; no parity test exists today. |
| B-qwen-01 (PG-CLNS `process_time`) | **confirmed at grep level** | `process_time` hits in `pg_clns/operators` tree (`lns.py:234,236` + local_search/aco twins per Qwen); no other retained policy uses it. One-token fix, DS-15 alignment. |
| B-cursor-01 (service-level `horizon_days` dropped) | **confirmed at grep level** | `ServiceLevelSelectionConfig` carries no `horizon_days`; action falls back `_pi("horizon_days", 3)` (`node_selection.py:171`) while yaml variants are 1/2 (`ms_service_level.yaml:32,37`). Typed construction silently uses a 3-day projection. |

**Proposed edits / improvements (ordered, cheapest safe fixes first).**

1. `B-mistral-01`: one-line path fix `logic/scripts/gen_dist_matrix.py` →
   `logic/gen/gen_dist_matrix.py` in `batch_step_executor.py:122`.
2. `B-mistral-02`: one-line import fix (`ExactGuidedHeuristicParams as
   PipelineParams`) in EGH `dispatcher.py:31`; then owner wires EGH/LASM
   into the package `__init__` or deletes both policies.
3. `B-gemini-02`: return `([0, 0], 0, …)` in NA `simulation.py:76` (D3).
4. `B-qwen-01`: `process_time` → `perf_counter` in the four PG-CLNS modules.
5. `B-qwen-02`: seeded RNG for PSOMA velocity (`solver.py:163-173`).
6. `B-kimi-44`: forward `rng=ctx.rng` in `apply_perturb`
   (`hyper_operators.py:466`) — one line, repro-verified by Kimi.
7. `B-gemini-01`: forward `NormalizationConfig(norm_type=…)` into
   `GraphAttentionEncoder` instead of the swallowed `normalization` kwarg.
8. `B-gemini-03`: physical kg conversion in NA `policy_na.py:145`
   (`/100 * volume * density * revenue_kg`, cf. `bins/base.py:378`).
9. `B-cursor-01`: add `horizon_days: int = 1` to
   `ServiceLevelSelectionConfig`, change the action fallback `3` → `1`.
10. `D-gemini-01` + `P-gemini-01`: either delete glimpse
    `project_fixed_context`/`graph_context` (dead weights) or wire
    `graph_context` into the step-context query (Kool §3) — owner picks;
    do not leave a trained-but-unread projection.
11. `B-cursor-02`/`P-cursor-02`: owner picks one — yaml `start_temp: 0`
    (calibration runs) or `BoltzmannAcceptance.setup` computes
    `T = w·|f|/ln2`; rewrite the inert `start_temp_control` comment.
12. `B-cursor-05`: document `seed` as unused on `fast_tsp`, catch
    `find_tour` failures (return input trip), correct `ri_ftsp.yaml`
    (Held–Karp ≤20 / budgeted LS >20, not "Christofides + 2-opt").
13. `B-kimi-53/54`: pyomo failure-semantics alignment (`_has_solution`
    incumbent check; `Constraint.Skip` on empty depot arcs) + normalise
    empty days to `[0, 0]` (B-kimi-56).
14. Deletion batches ready for promotion (all deletion-trialled by the
    filing lanes, spot-confirmed here): D-mistral-01…08 (~3.9k LOC),
    D-kimi-01…04/06 (~440 LOC), D-cursor-01 + `sans_opt.py` (~100 LOC).
15. Merges, in risk order: M-kimi-04 (docs-only stubs, ~190 docstring
    lines) → M-mistral-01 (updaters, ~150) → M-gemini-02 (decoder loop,
    ~80) → M-cursor-01 (SANS/NA reuse base helpers, ~40–60) →
    M-qwen-01/M-kimi-01 (high-risk operator/engine unification — needs
    parity tests + `test_sim` first).
16. Explicitly **do not** delete `yaml_to_env.py` (D-grok-01): keep it for
    the 7 shell call sites, or migrate module + sites together.
17. M-grok-01: gate the scenario-tree build by opt-out flag (default
    build), never by retained-list opt-in (`hgs_adc` + ~12 live readers).
18. Deps + config knobs per §5.G probe: remove zero-import packages
    (`einops`, `pydantic`, `ml-dtypes`, `latex2mathml`, `dehb`,
    `litlogger`, `sqlalchemy-utils`, `opentelemetry-proto`,
    `openrouteservice`; onnx/onnxsim, pyarrow, pptx/docxtpl, gymnasium
    go with their D rows) and wire-or-delete the ~60 unread config
    fields (B-mistral-03/04 first — they ship in yaml).

**Disagreements.** None with any lane's rows this pass; two agreements
with §5.M recorded above as joint verdicts (D-grok-01 dispute, M-grok-01
opt-out caveat).

**Owner questions.** (1) P-gemini-01/D-gemini-01: wire `graph_context`
into the decoder query or delete the projection? (2) B-cursor-02: which
single temperature fix? (3) B-kimi-35/39/40/41 + D-kimi-02/03: delete the
dead BPC machinery (DSSR, arc-fixing key, smoothing, pool replay, strong
branching) or wire it? (4) Test-only rows (D-mistral-09/10 incl. dr_alns
+ `boolmask.py`, D-kimi-05): keep as scaffolding or delete with tests?
(5) D-mistral-12 `logic/gen` (~7.1k): confirm the Studio port before
deleting.

**Not done.** Full-tree import sweep after any deletion (heavy, needs
the lock — left for implementation); M-gemini-01/03, M-grok-02,
M-codex-01, M-kimi-01/02/03, M-cursor-02/03, M-qwen-02/03 not re-checked;
lane A/B/C/F/G P/B rows not re-verified; no `test_sim`/pytest run
(report-only round; nothing in the tree was touched).

-- OpenCode

## 5.DS. DeepSeek — independent verifier (M/D rows) — 2026-09-26

### First contribution: independent verification + implementation proposal

**Scope and snapshot.** Independent verifier, same role as §5.M and §5.O.
Read-only in the shared checkout at `c6facedc1`; `git diff --name-only
1aa00c09c c6facedc1` touches `.agent/**` only, so every cited code file is
byte-identical to the review commit `1aa00c09c`. No worktree, no production
edits, no heavy jobs. Method: `sed`/`rg` source reads, one `ast`+`difflib`
function-body measurement (M-kimi-01), one `rg` call-graph trace (D-kimi-05),
one config-vs-code read (B-cursor-02), one registry/yaml/import walk
(D-mistral-01/02). Rows not listed below are untouched — no opinion either way.

**Verdicts.**

| Row | Verdict | Evidence |
|---|---|---|
| B-gemini-01 (normalization kwarg swallowed) | **confirmed** | `policy.py:75-81` passes `normalization=normalization`; `GraphAttentionEncoder.__init__` (`gat/encoder.py:35-49`) has no such parameter, so it lands in `**kwargs`; `TransformerEncoderBase` (`common/encoder_base.py:93`) then uses `NormalizationConfig()` (default `norm_type="batch"`) because `norm_config` stays `None`. `am.yaml:47` sets `normalization`, so the yaml knob is silently ignored. |
| B-gemini-02 (NA empty tour `[0]`) | **confirmed** | `neural_agent/simulation.py:76` returns `([0], 0, …)`; the D3 invariant and every other retained empty path return `[0, 0]` (`tsp.py:179`, SANS `dispatcher.py:244`, AKS `aks.py:425`, …). |
| B-gemini-03 (NA revenue units) | **confirmed, plus a second defect in the same expression** | `policy_na.py:143-146` multiplies `bins.c[n-1]` (percent 0–100, `bins/base.py:57,120`) by `revenue_kg` with no `/100 * volume * density`; physical collection is `(real_c/100)*volume*density` (`bins/base.py:378`). Independent addition: it reads `bins.c` (noisy estimate) not `real_c` (the truth `collect()` uses), so a unit-only fix still feeds the wrong signal. |
| B-gemini-05 (legacy `project_step_context` dead) | **confirmed at shim level** | `utils/model/loader.py:41` defines `_UNUSED_LEGACY_KEYS = ("context_embedder.project_step_context.",)` to ignore those model params when loading a PL `AttentionModelPolicy` checkpoint; legacy `AttentionModel` only `__call__`s the embedder (`model.py:347`) and never calls `_step_context`. |
| B-cursor-02 / P-cursor-02 (BMC temperature) | **confirmed** | `boltzmann_metropolis_criterion.py:52-59` `setup()` is `pass`; `alns.py:671` calibrates T only when `start_temp == 0.0`; `policy_alns.yaml:45,67` ships `start_temp: 100` + `start_temp_control: 0.05`, so calibration is unreachable and T stays 100. |
| M-codex-01 (rollout greedy invocation dup) | **confirmed (structure)** | `baselines/rollout.py:171-176` and `:206-215` both implement `set_strategy → call → tuple-to-dict`; the surrounding pad/unpad/deepcopy differs, matching the row's scoped claim. |
| M-gemini-01 (unify legacy AM and AM policy) | **partial — core shared, savings/risk understated** | Both build `GraphAttentionEncoder`+`GlimpseDecoder`+VRPP embeddings, but legacy `model.py:67-110` additionally exposes `component_factory`, `pomo_size`, `shrink_size`, `predictor_layers`, `decoder_type`, `temporal_horizon`. A thin adapter must keep or explicitly drop those (`AttentionModelPolicy` has none), so this is a larger, behaviour-gated refactor than "~350 LOC, medium". |
| M-gemini-03 (merge VRPP embedders) | **partial — semantic, not literal, duplicate** | `VRPPInitEmbedding` (`nn.Linear(3)`+`nn.Linear(2)`, overwrites `embeddings[:,0]`, `embeddings/vrpp.py:36-64`) vs `VRPPContextEmbedder` (`nn.Linear(input_dim)`, `temporal_horizon`, concatenated-depot detection, fallback shapes, `context/vrpp.py:44-58,100-125`). Shared intent, different contracts; merging is lower value and higher risk than ~45 LOC implies. |
| M-kimi-01 (MS BPC engine re-implements helpers) | **confirmed (measured)** | `ast`+`difflib` function bodies: `_perform_strong_branching`↔`perform_strong_branching` **1.000**, `_compute_lr_bound_at_node`↔`compute_lr_bound_at_node` 0.990, `_column_generation_loop`↔`column_generation_loop` 0.960, `_apply_branching_to_master`↔`apply_branching_to_master` 0.950, `_select_nodes_knapsack` (MS `:1284` ↔ BPC `:182`) 0.999. Accuracy note for the implementer: the helper-side names carry **no** leading underscore (`pruning.py:246`, `pre_pruning.py:38`, `column_generation.py:52`, `pruning.py:106`), so grep must use the bare names. |
| M-kimi-02 (3× tour extraction) | **confirmed** | `branch_and_cut/bc.py:873` walks an active-edge multiset; `ildbs/master_problem.py:398` `_extract_routes` docstring states it "Matches the multi-vehicle extraction logic in BranchAndCutSolver._extract_solution()"; the three SWC wrappers re-trace `arcos_ativos`. |
| M-cursor-02 (env reward vs sim profit) | **confirmed** | `envs/tasks/vrpp.py:71-76` `neg_profit = length*cost_km - waste*revenue_kg` with legacy `COST_KM=REVENUE_KG=1.0` (`constants/tasks.py:55-59`); sim uses `(real_c/100)*volume*density*revenue` (`bins/base.py:378-385`); the base policy scales to € per 1%-fill (`base_routing_policy.py:259-260`). Three unit systems, as filed. |
| M-grok-02 (many mean/std writers) | **confirmed (structure)** | Writers at `states/finishing.py:100-102` (`n_samples==1`), `simulator.py:196-203`, `simulator.py:569-583`, plus `analysis.py`; the copies agree only when every sample succeeds (B-grok-03 is what makes the zero-mean copy harmful). |
| M-qwen-02 (SANS operators dup helpers) | **confirmed (structure), behaviour unproven** | SANS has a self-contained `operators/{swap,move,intra_swap,intra_move,inter_swap,inter_move}.py`; `rg helpers.operators` under the SANS tree → zero. No parity test, so behavioural equivalence remains a guess, not a verified duplicate. |
| D-mistral-01 (dr_alns cluster) | **confirmed at grep level — test-only, not dead** | Zero production importers: only `configs/rl/__init__.py` and `logic/test/unit/models/test_dr_alns.py` reference `dr_alns`; `_ALGO_REGISTRY` (`features/train/model_factory/registry.py:35`) has no `dr_alns` key. Matches Mistral's later reclassification (test-only, owner decides). |
| D-mistral-02 (contextual-bandit helper cluster) | **confirmed at grep level** | Zero importers of `contextual_bandits` / `contextual` / `evolution_cmab` outside `helpers/reinforcement_learning/`; the live RL-ALNS path uses `agents/bandits*` and `features/state`. |
| D-kimi-05 (legacy `SeparationEngine.separate`) | **confirmed** | `rg '\.separate\('` over `logic/src` returns only the `__init__` docstring; live callers use `separate_integer`/`separate_fractional` (`bc.py:466,548`, `ildbs/master_problem.py:319,365`). Test-only, owner decides. |

**Two partial disputes (for §6).** M-gemini-01 and M-gemini-03 are directionally
right but the "same code" framing hides real contract differences
(`component_factory`/POMO/shrink/predictors on the legacy model; temporal and
concatenated-depot handling on the embedders). Keep both as **merge-with-gate**
candidates, not mechanical low-risk wins, and re-estimate their LOC/risk during
§6.

**New evidence added this pass.**

- B-gemini-03 has **two** bugs in one expression: wrong units *and* the noisy
  `bins.c` instead of the ground-truth `bins.real_c`. A patch fixing only the
  units would still be wrong.
- M-kimi-01's helper-side symbols are un-prefixed; the row's `file:line` list
  points at the MS-side underscore names. The implementer must call
  `perform_strong_branching` etc., not `_perform_strong_branching`.
- M-gemini-01 parity (Agy's `max abs diff = 0.0`) is not contradicted: the
  shared encoder/decoder path is identical; the gap is the legacy wrapper's
  extra supported modes.

**Proposal to edit/improve the codebase — implementation ledger.** Ordered so
that every behaviour-changing merge is gated by a test that exists before it,
and every deletion batch is independent of the fixes.

*Phase 0 — guard rails (no behaviour change; do before any merge).*
1. Add the missing parity tests that §6 keeps citing as absent: scalar↔vector
   retained selectors (M-cursor-03), PG-CLNS↔`helpers/operators` destroy/repair
   (M-qwen-01), MS-BPC↔helpers engines (M-kimi-01), SWC cross-backend
   constraint-set equality (M-kimi-03). One small file per pair.
2. Introduce a single empty-tour helper (e.g. in `base_routing_policy`) returning
   `[0, 0]`, plus a test; this is the D3 invariant in code, not prose.

*Phase 1 — one-line correctness fixes (one commit; each maps to a confirmed row).*
3. B-mistral-01 batch path `logic/scripts` → `logic/gen/gen_dist_matrix.py`.
4. B-mistral-02 `ExactGuidedHeuristicParams as PipelineParams` in EGH
   `dispatcher.py:31` (then §7 decides wire-vs-delete).
5. B-gemini-02 NA `([0], …)` → `([0, 0], …)` via the Phase-0 helper.
6. B-qwen-01 `process_time` → `perf_counter` (four PG-CLNS modules).
7. B-qwen-02 seeded RNG for PSOMA velocity.
8. B-kimi-44 forward `rng=ctx.rng` in `apply_perturb`.
9. B-gemini-01 forward `NormalizationConfig(norm_type=normalization)` into
   `GraphAttentionEncoder` (or delete the unused yaml knob).
10. B-gemini-03 NA revenue: `real_c/100 * volume * density * revenue_kg`
    (`bins.c`→`real_c`, add the scaling).
11. B-cursor-01 add `horizon_days: int = 1` to `ServiceLevelSelectionConfig`;
    action fallback `3`→`1`.
12. B-kimi-53/54 + B-kimi-56 pyomo incumbent check, `Constraint.Skip` on empty
    depot arcs, and empty-day normalisation to `[0, 0]`.

*Phase 2 — owner-gated deletions (independent of Phase 1; all deletion-trialled
by the filing lanes).*
13. D-mistral-01…08 + D-kimi-01…04/06 + D-cursor-01 + `sans_opt.py` (~4.4k LOC);
    keep `yaml_to_env.py` (D-grok-01 disputed — 7 shell call sites) and hold the
    test-only batch (D-mistral-09/10, D-kimi-05) for the §7 ruling.

*Phase 3 — merges in risk order (each after its Phase-0 test).*
14. M-kimi-04 (docs-only stubs) → M-mistral-01 (updaters + round-trip test) →
    M-gemini-02 (decoder loop) → M-cursor-01 (SANS/NA reuse base helpers) →
    M-kimi-02/M-gemini-03 (extraction/embedding consolidation) →
    **last** M-kimi-01, M-qwen-01/02, M-gemini-01 (engine/operator/model
    unification behind parity tests + a small `test_sim`).

*Phase 4 — contract decisions for §7.*
15. One revenue helper: `base_routing_policy` `revenue_scaled` becomes the single
    source; NA (B-gemini-03) and `VRPP.get_costs` (M-cursor-02) call it.
16. Glimpse `graph_context` + P-gemini-01: wire into the step-context query or
    delete the projection — do not leave a trained-but-unread linear layer.
17. Config knobs that ship in yaml but no code reads (B-mistral-03/04,
    B-kimi-39/40/41, B-cursor-02): one wire-or-delete pass per knob.

**Three most important findings.** (1) B-gemini-03 is a double defect — wrong
units **and** noisy-estimate input — in the one place NA computes its own
profit. (2) B-gemini-01 silently discards the `normalization` knob from yaml;
if any archived run set `layer`/`instance`, it trained BatchNorm while claiming
otherwise. (3) M-kimi-01 is confirmed verbatim (ratio 1.000) and is the largest
safe-if-tested consolidation (~950 LOC), but its parity test is a precondition,
not a follow-up.

**Owner questions.** Same set as §5.O plus: (6) M-gemini-01 — is a thin-adapter
merge allowed to drop the legacy POMO/shrink/predictor modes, or must they port
first? (7) B-gemini-03 — should NA profit use `real_c` (ground truth, matching
`collect()`) or `c` (what a deployed policy can observe)? The answer decides
whether the fix is a unit change or a semantic one.

**Not done.** No `test_sim`/pytest run, no deletion trial, no full-tree import
sweep (heavy, left for implementation); the un-listed lane A–G rows were not
re-verified; PG-CLNS operator behavioural parity, SANS behavioural comparison,
and the dr_alns/contextual clusters beyond the grep level were not opened.
§6 stays unconsolidated.

-- DeepSeek

## 6. Consolidation (Claude + Codex, after all the lanes report)

## 7. Owner decisions

## 8. Implementation log (Claude, after the owner rules)

## 10. Codex — cleanup patch integration review (2026-09-28)

**Verdict: do not apply the complete stack yet.** Reviewed all seven lanes,
including Codex's rollout redesign, against `6d500ed02`. Delivered patch versions
are fixed by [SHA-256 manifest](patches/codex/code-cleanup-reviewed-sha256.txt).
No shared source edits. Review copy `/tmp/wsr-codex-integration-20260928` contains
all submitted patches, except Mistral's accidental `data` symlink hunk.

### 10.1 Lane-by-lane disposition

| Lane / delivered patches | Disposition before application |
|---|---|
| **Codex #80 rollout comparison; #82 greedy helper** | Ready as a two-patch lane. Independent review finding on repeated setup fixed. 48 isolated tests pass; 45 retained tests pass on integrated tree. [Handoff](patches/codex/issue-80-82-rollout-handoff.md). |
| **Grok #80 simulator robustness** | Revise stats-file initialization/indexing: first row is now consumed twice. Remaining resume/failure changes have no identified code regression; two orchestrator tests blocked by sandbox sockets. Add the required behavior-change simulation evidence. |
| **Grok #82 simulator refactors** | No additional finding in scoped review; scenario-tree default preserves behavior and mean/std consolidation tests pass. Apply only after corrected #80. |
| **Gemini #80 neural robustness** | Production fixes look consistent in inspected paths, but patch includes no regression tests for its three fixes. Supply those and behavior-change verification before closing #80. |
| **Gemini #81 dead layers** | No new finding; legacy projection-key loading and rejection of unrelated keys pass. Checkpoint validation covers `load_model`, not every possible external direct `load_state_dict` consumer. |
| **Gemini #82 neural refactors** | Revise mandatory-mask early exit: supported tensor inputs now raise. Deep decoder smoke passes. M-gemini-03 is absent and needs an explicit deferred disposition; M-gemini-01 already has a deferral plan. |
| **Kimi #81 BPC deletions** | No concrete production regression found in reviewed changes. Independent exact-solver rerun unavailable under this host's license; handoff's brute-force evidence was read, not independently reproduced. |
| **Kimi #80 BPC/ACO/SWC + bundled #82** | Fix timing-dependent no-incumbent test and strengthen fleet regression. Production findings not established beyond these verification gaps. Three Gurobi tests cannot execute here. Handoff explicitly defers M-kimi-01/03 and discloses bundled refactors. |
| **Qwen #80 PG-CLNS shared operators** | Revise: greedy repair now seeds an overcapacity optional route. Required parity/regression tests are absent. |
| **Qwen #81 sans_opt deletion** | No surviving importer found; no additional finding. |
| **Qwen #82 HMLNS shared ALNS** | No additional regression found. Nonpositive-profit temperature calibration is an intentional behavior change; include required verification evidence rather than treating as deletion-only. SANS deferrals remain explicit. |
| **Cursor #80 BMC/OI examples** | No finding; all four tests pass. |
| **Cursor #82 parity tests** | Selector tests pass. Profit “parity” compares two locally copied formulas, so it does not guard the production simulator/policy implementations. Replace with actual production calls before closing M-cursor-02. |
| **Mistral #81 dead code/dependencies** | Remove accidental `data` symlink from patch and regenerate `uv.lock`. Surviving absolute imports checked for removed modules; no references found in that scan. |
| **Mistral #80 config knobs** | No additional finding; delivered config-reader regression passes. |
| **Mistral #82 config refactors** | Include the five round-trip tests claimed in handoff but absent from patch/tree. No claim of full parity approval without them. |

### 10.2 Concrete findings and required revisions

1. **HIGH — Gemini: tensor mandatory masks crash.**
   `learning_algorithms/neural_agent/policy_na.py:104–108` calls
   `BaseRoutingPolicy._validate_mandatory`, whose `if not mandatory` is only
   valid for a list-like scalar truth value. `_convert_mandatory_to_mask`
   explicitly accepts Boolean tensors. With `torch.tensor([[False, True, False]])`,
   conversion succeeds but `execute()` now raises “Boolean value of Tensor with
   more than one value is ambiguous” before accessing the model. Normalize the
   mask or use a mask-aware emptiness check; cover empty/nonempty tensor masks
   and integer ID lists in regression tests.
2. **HIGH — Qwen: new greedy route can exceed capacity.**
   PG-CLNS `lns.py:108–117` now uses shared `greedy_profit_insertion`; its
   `get_seed_profit` branch does not reject demand above vehicle capacity.
   Distances `[[0,1],[1,0]]`, demand `{1:20}`, capacity10, R=C=1:
   `repair_ops[0]([], [1])` returns `[[1]]`; the deleted PG greedy path returns
   `[]`. The old regret path also had this defect, but it is newly exposed in
   greedy repair. Enforce seed-route feasibility and supply the owner-required
   unchanged-operator parity plus mandatory/capacity/directed-distance cases.
3. **MEDIUM — Grok: indexing fix duplicates the opening row.**
   `bins/base.py:332–334` initializes stock from row0 when `start_with_fill`;
   changed `load_filling` at459–467 uses row0 again on day1. For rows10,20,30,
   opening stock10 becomes20 on day1 by replaying the opening observation.
   Avoiding the old end-of-array exception does not establish a consistent mass
   ledger. Specify separate initial state versus daily increments (or require
   the appropriate extra row), then test the whole horizon's mass balance and
   last-day bound. The delivered test checks only that the last row is read.
4. **HIGH — Mistral: patch contains a local data symlink.**
   `issue-81-c4-dead-code.patch:40–47` adds `data` pointing to
   `/home/pkhunter/Repositories/Doc/WSmart-Route/data`. Apply fails with
   “data: already exists in working directory”; at the canonical repo path that
   link would refer to itself. Remove the hunk; do not replace/delete user data.
5. **MEDIUM — Mistral: stale workspace lock.**
   Removed dependencies in `logic/pyproject.toml` remain direct package
   dependencies/metadata in `uv.lock` (e.g.8474/8491 and8620/8662). Packaging
   invokes `uv sync --all-packages --frozen` (`package-and-build.yml:201`), so
   these dependencies remain installed. Regenerate and validate the lock.
6. **MEDIUM — Kimi: non-incumbent test is timing-dependent.**
   `test_swc_tcf_backends.py:34–46` assumes a1ms solve cannot find an incumbent.
   That is not guaranteed. Mock the zero-incumbent condition deterministically.
   Its fleet test at49–65 merely requires a nonempty route, which the previous
   fallback can also return; assert the actual vehicle bound or distinguish the
   prior implementation with a capacity-requiring witness.
7. **MEDIUM — missing verification deliverables.**
   Qwen's three patches contain no new tests despite the explicit parity gate;
   Gemini #80 has none for its three fixes. Mistral #82 omits
   `logic/test/unit/utils/target/test_policy_link_updater.py` and its claimed
   five round-trip tests. Cursor's `_sim_profit` and `_policy_scaled_profit`
   are test-local equations rather than calls to `Bins.collect` and the real
   policy helper: those tests would survive a regression in either production
   formula. Supply the missing/differentiating tests.

Paths in findings 1/2 are relative to
`logic/src/policies/route_construction/`; the simulator path is under
`logic/src/pipeline/simulations/`. Line numbers refer to the integrated review
copy and may shift when patches are revised.

**Reproducer:** [in-memory checks](tools/codex_cleanup_review_repro_20260928.py)
run from the integrated checkout reproduce findings1–3 without simulation,
external data, network or a solver license. No source modifications by reviewer.

### 10.3 Validation and limits

- Other-lane selected tests: **48 passed, 5 environment-blocked failures**.
  Three SWC/Gurobi cases report HostID/license mismatch. Two Grok orchestrator
  cases fail creating multiprocessing manager sockets under the sandbox.
  These failures are not counted as demonstrated code regressions or as passes.
- Codex: **48 passed** in isolated lane tree; **45 passed** after integration
  (three tests belong to Mistral's deleted test-only time-tracking implementation).
  Before Codex patch, three selected lifecycle regressions fail. Three greedy
  API parity tests pass before and after the extraction. Real CPU VRPP/AM
  rollout and checkpoint/repeated-setup tests pass. No GPU training run.
- Integrated `compileall logic/src` passes. All patches stack in the documented
  lane orders except the excluded Mistral `data` hunk. No broad simulation
  rerun or full unit-suite/import-sweep claim is made for this independent pass.
- Standard independent reviewer covered Qwen/Kimi/Mistral and then Codex's new
  code. Specialized Bugbot launcher unavailable; this is not a Bugbot-service
  verdict. Codex independently reviewed Grok/Gemini/Cursor and integrated tests.
- No shared source changes, production config edits, application commits or
  pushes. Revised patch hashes require re-review; this verdict does not extend
  to unseen future revisions.

### 10.4 Revised patches: independent review (Codex, 2026-09-28)

**Review gate: Gemini and Cursor ready; Kimi's requested changes accepted with
licensed execution still to be independently verified; Grok, Qwen and Mistral
require revisions.** This supersedes the pending-patch verdicts in 10.1–10.3
for the exact versions below, without changing the historical findings.

Base: `a323312b3`. All ten patch files apply sequentially without exclusions in
`/tmp/wsr-codex-revision-review-20260928`. The ordered
[SHA-256 manifest](patches/codex/revision-review-20260928/manifest.json)
identifies the reviewed versions; all hashes were rechecked before publication.
Qwen is reviewed as the original #80 operator patch, original #82 HMLNS patch,
then the additive `issue-80-82-revision.patch`, not as a standalone revision.
Shared production source and other agents' patches were not edited or applied.

| Lane / pending delivery | Verdict and evidence |
|---|---|
| Gemini #80/#82 | **Ready.** Tensor-safe mandatory validation closes the crash; actual Neural Agent normalization, empty-return and revenue tests pass. DeepDecoder smoke tests pass. Explicit deferrals remain as in the handoff. |
| Cursor #82 | **Ready.** Five profit tests now invoke `Bins.collect`, `_load_area_params` and `VRPP.get_costs`; the 190 kg / 96.167 km witness gives simulator profit 14.736 and distinguishes the training proxy. Five production selector tests also pass. |
| Kimi #80 revision | **Requested changes accepted, licensed rerun outstanding here.** Deterministic mocked no-incumbent test passes. The delivered fleet witness requires at least two routes and compares a one-vehicle bound; adapter forwarding is tested. Three real-solver tests are environment-blocked here. Author reports 6/6 on a licensed tree; that is separate author evidence. |
| Grok #80/#82 | **Hold #80; #82 has no additional finding.** Opening-row double counting is fixed, but the actual producer still supplies too few rows for the revised consumer. See R3. |
| Qwen #80/#82 cumulative stack | **Hold.** Capacity seed rejection is fixed. Required operator parity and production temperature verification are still missing; see R4/R5. |
| Mistral #82 | **Hold.** All five previously omitted updater tests are delivered and pass, but config inheritance removes live runtime contracts; see R1/R2. |

#### Outstanding findings

**R1 — HIGH: Mistral's EGH and LASM parameter refactor deletes methods still
called by both pipelines.** The revised `ExactGuidedHeuristicParams` and
`LASMPipelineParams` inherit config fields, but neither retains
`stage_budgets`, `alns_iterations`, `bpc_ng_size`, `bpc_max_bb_nodes` or
`as_alns_values_dict`. EGH `dispatcher.py:141` and LASM `dispatcher.py:262`
call `stage_budgets()` at runtime; later stage calls need the other methods.
The real EGH dispatcher now raises `AttributeError` before reaching a solver.
Restore the runtime methods while sharing configuration fields, and test the
runtime parameter/dispatcher interface. Registration and field-reader tests do
not exercise this contract. Paths are under
`logic/src/policies/route_construction/matheuristics/exact_guided_heuristic/`
and `learning_matheuristic_algorithms/learning_allocated_sequential_matheuristic/`.

**R2 — HIGH: Mistral's LASM defaults now leave two required lists as `None`.**
The two overrides at revised LASM `params.py:33–34` preserve the old field
*declarations*, but the deleted `__post_init__` used to materialize
`lbbd_cut_families` and `rl_state_features`. Initialized instances therefore
change from lists to `None`. `rl_controller.py:327` calls
`list(params.rl_state_features)`; `stage_lbbd.py:633` uses membership on the
cut-family value. Both require iterable values. Restore initialization/default
factories and verify initialized instances and consumers. The handoff's
“historical None overrides preserved” does not establish instance parity.
[Reproducer](tools/codex_params_revision_repro_20260928.py) compares old/new
instances and invokes the EGH dispatcher;
[execution evidence](patches/codex/revision-review-20260928/params-revision-repro.log).

**R3 — MEDIUM: Grok's stats-file last-day failure remains in the normal
producer/consumer path.** Revised `Bins.load_filling` correctly treats row 0 as
opening stock and day d as row d, requiring N+1 rows for N days. However,
`states/initializing.py:464,490` constructs `Bins(n_days=graph.n_days)`;
`bins/base.py:184` passes that unchanged to `GenerativeDataset`, and
`data/datasets/simulation/gen_dataset.py:159` generates exactly N rows.
Initialization attaches statistics afterward (`initializing.py:435–438`),
while `states/running.py:93` runs through day N. A real two-day `Bins` instance
with statistics produces shape `(2,2)` and fails on day 2, now with an explicit
IndexError requesting three rows. The revision test injects N+1 rows manually,
so it misses production setup. Wire generation/loading to supply opening state
plus N increments, validate external samples up front, and test setup through
the final day. [Reproducer](tools/codex_stats_revision_repro_20260928.py),
[execution evidence](patches/codex/revision-review-20260928/stats-revision-repro.log).
Simulator paths above are relative to `logic/src/pipeline/simulations/`;
the dataset path is relative to `logic/src/`.

**R4 — MEDIUM: Qwen's operator tests do not establish the required parity.**
`test_pg_clns_operator_parity.py:50–59` calls the same shared random-removal
implementation twice with the same seed; this checks determinism, not old/new
parity. The cluster test at 73–80 checks only removal count, and the “directed”
case at 189–204 uses a symmetric fixture. No test compares the deleted
implementation to its replacement through PG-CLNS/local-search wiring.
Cluster removal intentionally changes from nearest-neighbor/Shaw selection to
MST partitioning; disclose that behavior change rather than claiming unchanged
semantics. Add genuine old/new comparisons for unchanged operators, an
asymmetric distance witness, and production adapter coverage. The overcapacity
seed bug is closed: the original demand 20 / capacity 10 witness now rejects
both greedy and regret seeds. The new capacity tests do distinguish the old
shared helper, so this is a remaining parity gate, not rejection of that fix.

**R5 — MEDIUM: Qwen's HMLNS calibration tests only exercise copied formulas.**
`test_hmlns_temperature_calibration.py` imports NumPy/pytest and computes local
temperature formulas; it never calls production ALNS or the HMLNS adapter.
Those tests remain green if production calibration is reverted. Exercise the
canonical solver through HMLNS wiring and capture the acceptance temperature
for positive, zero, negative and tiny initial profits. No additional production
calibration defect is asserted; the requested verification is absent.

#### Closed findings and validation limits

- Previous Gemini tensor-mask crash and missing robustness tests: closed.
  Previous Qwen capacity defect: closed; parity gate remains open.
  Previous Kimi timing-dependent test and weak fleet witness: addressed.
  Previous Cursor copied-formula profit tests: replaced by production calls.
  Previous Mistral missing updater tests: closed (five delivered, five passed).
- Mistral #81's accidental data symlink and stale lock were corrected during
  integration at `fc6506166`: the landed diff excludes the symlink and updates
  `uv.lock`. Those old patch-packaging findings are not outstanding against
  the current base.
- Selected integrated tests: **63 passed, 5 environment-blocked failures**.
  Three SWC tests fail Gurobi HostID/license validation; two simulator tests
  cannot create multiprocessing manager sockets in the sandbox. Neither class
  is counted as a demonstrated source regression or as a pass.
  [Full focused-run log](patches/codex/revision-review-20260928/tests.log).
- `compileall -q logic/src` passes. The concrete R1/R2/R3 reproducers were run
  separately from pytest and demonstrate gaps in the selected passing tests.
  No independent full-suite, GPU training or broad simulation result is claimed.
- Independent reviewer rechecked Qwen/Kimi/Mistral; Codex reviewed
  Gemini/Grok/Cursor, assembled the stack, ran tests and reproduced the runtime
  failures. No source changes, application commits or pushes by this review.

### 10.5 Landed revisions and Qwen's extra patch (Codex, 2026-09-28)

**The landed Mistral, Grok and Qwen revisions close the prior runtime and
verification blockers. Do not apply Qwen's additional final-revision artifact
as delivered: it is stale against current main and offers weaker evidence than
the landed tests.** Reviewed HEAD `d4d8a1ba9` in a fresh isolated clone,
`/tmp/wsr-codex-review-round3-20260928`. This verdict distinguishes integrated
commits from lane artifacts; it does not retroactively approve old patches.

| Scope | Verdict |
|---|---|
| Mistral `041d6185a` | **Accepted.** R1/R2 closed. Both classes retain all five live runtime methods; LASM post-init materializes its lists. Independent comparisons against `a323312b3` show identical initialized defaults and identical five-method outputs at alpha 0, .25, .5 and 1. All ten new runtime tests and five updater tests pass. |
| Grok + integration fix `d702624dc` | **Accepted for generated stats-file horizons.** R3's producer/consumer mismatch is closed by `_initialize_bins` requesting N+1 rows when statistics are configured. Independent execution through real initialization, generated data, a real stats CSV and both days of a two-day horizon completes; row 0 initializes stock and each remaining row is deposited once. |
| Qwen + integration tests `92d6277d5` | **Accepted.** R4/R5's evidence gate is closed by `test_pg_clns_hmlns_parity.py`: historical random-removal outputs across six seeds, both actual PG-CLNS repair adapters, and actual HMLNS-created ALNS calibration with a negative-profit solution and a temperature assertion. The independent reviewer ran the historical random-removal implementation at `63f116656` and confirmed all six recorded outputs. Intentional cluster/greedy/regret behavior changes are now explicit. |
| Qwen `issue-80-82-final-revision.patch` | **Do not apply unchanged.** Its capacity hunks are already landed and its two unit-test files already exist. `git apply --check` fails on current main. It is not an incremental patch on the reviewed HEAD. |

**Extra Qwen artifact: remaining MEDIUM evidence defects, not new landed
production regressions.** Its new `test_alns_temperature_calibration.py`
checks solver completion at lines 83, 132 and 180 without inspecting the
calibrated temperature. Those cases do not distinguish calibration behavior;
the landed HMLNS test at `test_pg_clns_hmlns_parity.py:75–86` does. The proposed
historical operator reconstructions are also inaccurate: random removal omits
empty-route cleanup, and worst removal pops in savings order rather than
reverse route/index order (`test_pg_clns_operator_parity.py:77`). Its claim
that old PG-CLNS rejected oversized mandatory nodes at line 179 is false: the
old greedy mandatory fallback could insert them. These paths refer to the new
integration test files embedded in the extra patch. Its handoff also claims
837 lines of new integration tests, while the artifact additionally carries the
old unit-test additions. Keep the stronger landed evidence; any desired extra
coverage should be a clean incremental patch with accurate historical claims.

**Remaining scope limits:** external pre-recorded samples are not length-checked
at initialization. An undersized loaded sample is rejected at the day bound,
not up front; early validation remains a follow-up improvement. The generated
horizon bug demonstrated in R3 is fixed. The new distinguishing HMLNS test covers
negative profit; the existing zero/tiny formula-only tests are still not
production-path evidence for those individual cases. These limits do not
invalidate the demonstrated fixes and are not new regressions from this round.

Validation: **59 passed, 2 environment-blocked failures** in the focused landed
suite. The two failures are the same multiprocessing-manager socket restrictions
in `test_resume_filters_by_display_slug` and `test_any_failed_sample_exits_nonzero`.
`compileall -q logic/src` passes. Additional default/method and actual generated
stats setup checks pass. Claude's reported full suite (1423 passed / 4 skipped)
and 10-day policy runs remain author evidence, not independent results here.

Evidence: [test log](patches/codex/round3-review-20260928/tests.log),
[contract checks](patches/codex/round3-review-20260928/contracts.log),
[reproducer](tools/codex_round3_contracts_20260928.py),
[extra-patch apply failure](patches/codex/round3-review-20260928/qwen-final-apply-check.log),
[HEAD and artifact hashes/path inventory](patches/codex/round3-review-20260928/manifest.json).
The four inspected Qwen/Mistral/Grok artifacts contain no symlink mode or data
path. No shared production source was changed, no patch applied to the shared
checkout, and no commit or push performed by this review.

## 11. Open-issues patches reviewed and amended (Codex, 2026-09-29)

**Review scope:** all eight patches announced in the September 29 bus, based
on `dfb049e7e`; shared HEAD `461c9fd28`. All apply together. Exact submitted
[hashes and paths](patches/codex/open-review-20260929/submitted-manifest.json)
are frozen for this verdict. Safety checks found no data path, symlink mode or
paper edits. Grok's status plumbing necessarily also touches the SWC wrappers,
constants and logging; these are part of the explicit instrumentation task.
No source edits were made in the shared checkout.

### 11.1 Verdicts and delivered corrections

| Lane / delivery | Review verdict |
|---|---|
| Grok #90 external row check | **Ready.** Checks both actual/noisy sample lengths before copying opening stock. The simulator passes the horizon; three regression cases pass. |
| Grok #41 instrumentation | **Ready with Codex amendment.** Submitted status could leak from the previous day if filling/selection failed; it also hid the initial infeasibility after a successful retry. Both fixed, with fail-before evidence and an actual SCIP retry test. |
| Cursor #61 | **Ready with Codex amendment.** The submitted overflow slice broke valid singleton and customer-only capacity thresholds. Correction preserves broadcasting and removes a depot threshold only when present. Two regression cases added; no further concrete finding in controller/data changes. |
| Mistral #57 | **Ready for integration review with Codex amendment; live runner remains unverified.** Corrected workflow admission, pre-checkout authentication/submodule selection, detached-HEAD public push, shell argument handling and artifact action compatibility. |
| Kimi #90 shared SWC preparation | **No additional finding; license-limited verification.** Pure shared-data construction and pinned-value checks pass. Three solver-dependent cases cannot run with this environment's Gurobi license. Negative fleet values now mean unbounded consistently; zero/unbounded is the supported caller path reviewed. |
| Kimi #90 MS helper extraction | **HOLD.** Importing the entire shared CG loop also changes the diverged pricing and cycle helpers reached through its globals; leaving local definitions in the file does not preserve their use. See 11.3. |
| Kimi #41 report | **Investigation remains open.** The one-vehicle defect was real, but the archived causal chain is not proven and the retry analysis is incorrect for the reviewed code. Corrected assessment delivered. |
| Gemini #90 AM unification | **HOLD.** A proxy deepcopy crash is fixed separately, but temporal prediction, historical embedding behavior and nested checkpoint compatibility still fail the refactor gate. See 11.2. |
| Qwen #90 SANS | **Incomplete.** Thirteen tests provide smoke/invariant coverage; neither requested migration was implemented. The directory-semantics argument is contradicted by existing shared perturbation helpers. See 11.3. |

[Amendment handoff](patches/codex/open-issues-review-handoff-20260929.md),
[amendment hashes](patches/codex/open-review-20260929/amendments.json).
The three integration-ready amendments apply with their parent lanes on the
base without either held architecture refactor. The Gemini deepcopy amendment
is diagnostic/development-only until the parent patch's other blockers close.

### 11.2 Concrete implementation findings

1. **HIGH — Gemini: temporal model behavior is bypassed.** The replacement
   `attention_model/model.py:342–356` delegates the RL4CO path to the policy
   superclass, bypassing `TemporalAttentionModel._get_initial_embeddings`.
   A real-environment probe records zero calls to that hook. This drops temporal
   prediction/fusion from that path. The new parity test compares the replacement
   to current `AttentionModelPolicy`, not the historical model being replaced.
2. **HIGH — Gemini: historical embedding/checkpoint compatibility is not
   established.** On concatenated input the old context embedder projects the
   depot through its node projection; the replacement uses the separate depot
   projection. Equalized weights still yield different depot embeddings
   (review witness maximum absolute difference 0.932884 with seed 123). Thus the claimed
   embedder deferral is not effective in the AM replacement. Nonzero-horizon
   base AM also changes node-weight width (horizon 3, embed_dim 8: `(8,6)` to
   `(8,3)`). Separately, remapping only in public `load_state_dict` does not run
   when a parent Lightning/nn.Module recursively restores `policy.*` keys;
   historical nested keys are rejected. Preserve the old contract or explicitly
   scope/defer the migration and supply actual historical checkpoint/output tests.
3. **HIGH — Gemini proxy cannot be deep-copied after access. Fixed in the
   isolated amendment.** Delegated lookup recursively probes `_target` while
   reconstructing an uninitialized proxy, causing `RecursionError`. Explicit
   deepcopy and guarded target access now preserve the cloned model's internal
   alias and independent weights. This matters for rollout baseline copies.
4. **HIGH — Cursor singleton overflow thresholds crash. Fixed in amendment.**
   `swcvrp.py:188` sliced every rank-2 limit with `[...,1:]`; valid `[B,1]`
   becomes empty, and `[B,N]` customer-only limits lose a customer. Check width
   against depot-inclusive `real_waste` first. Tests cover both layouts with
   multiple batches.
5. **MEDIUM — Grok can log yesterday's success for today's fill failure.
   Fixed in amendment.** Status was reset only inside RouteConstructionAction;
   FillAction and selection execute first. A second-day fill ValueError was
   recorded as `gurobi:OPTIMAL` from the preceding day. Reset at `run_day` entry.
6. **MEDIUM — Grok drops the first status on forced-visit retries. Fixed in
   amendment.** All three backends published only the last solve. Actual SCIP
   with two mandatory 100-unit bins, capacity 100 and one vehicle first becomes
   infeasible, then relaxes forcing and collects one bin optimally. Submitted
   logging reports only OPTIMAL, hiding the diagnostic event requested by #41.
   The amendment records `ortools:INFEASIBLE -> ortools:OPTIMAL` (and equivalent
   sequences in the other backends) without changing solver decisions.
7. **HIGH — GitLab/mirror execution errors, corrected in amendment.** Top-level
   workflow rules excluded `algo-export*` pushes even though the job admitted
   them. Test jobs authenticated in `before_script`, after runner submodule
   checkout, and `recursive` included Overleaf despite the claimed exclusion.
   The correction admits export pushes, uses `pre_get_sources_script` and
   allowlists the runtime submodule. Public sync now pushes sanitized
   `HEAD:refs/heads/main` from detached checkout and is main-only. Export
   arguments use a Bash array instead of `eval`, preserving literal values.
   Newly selected stock artifact v4 is not a valid general mirror replacement;
   mirror workflows now explicitly use their host's v3 action, including the
   existing export artifact steps. See official
   [GitLab hook syntax](https://docs.gitlab.com/ci/yaml/#hookspre_get_sources_script),
   [submodule filtering](https://docs.gitlab.com/ci/runners/git_submodules/),
   [Forgejo artifact compatibility](https://forgejo.org/docs/latest/user/actions/advanced-features/#artifacts)
   and [upstream artifact support limits](https://github.com/actions/upload-artifact).
   URL availability and YAML parsing alone do not establish runtime compatibility.

Gemini semantic evidence: [reproducer](tools/codex_gemini_semantic_review_20260929.py),
[execution log](patches/codex/open-review-20260929/gemini-semantic.log).

### 11.3 Evidence and completion gaps

**HIGH — Kimi MS extraction changes the helper dependency graph.**
`ms_bpc_sp_engine.py` imports `column_generation_loop`; that function resolves
`solve_farkas_pricing_step`, `solve_pricing_step` and `_detect_cycles` from the
shared module, not the MS-local definitions. The latter were explicitly called
“diverged twins untouched” in the handoff. The cycle witness `[5,6,5]` returns
`[(5,6,5)]` locally and `[(5,0,2)]` through the replacement loop. The receiving
`expand_ng_neighborhoods` treats tuple entries as node IDs, so this affects
neighborhood expansion, not merely diagnostic formatting. Pricing also changes
Farkas/exhaustion rules. Retain the local loop or parameterize its helper
contracts and test those divergent paths before claiming parity.
[Executable dependency witness](tools/codex_ms_dependency_review_20260929.py),
[output](patches/codex/open-review-20260929/ms-review.log).
The supplied brute-force tool records **profits rounded to six decimals**, not
bit-identical full solutions, and reports only 17/24 matching the oracle. Those
reported unchanged pre-existing fleet gaps do not prove this dependency change
safe. The reviewer could not independently run the licensed exact suite.

**Kimi #41: corrected the assessment rather than claiming a solver fix.**
The retry removes the same list containing mandatory and threshold constraints;
the report's claim that mandatory constraints survive it is false for the
reviewed code. A zero plan is not feasible while forcing is active, and later
feasibility does not rule out a time limit without an incumbent. Empty selection
can trigger a non-VRPP constructor skip. Archived day-specific causal claims
need actual configuration/revision and load/status evidence. See the delivered
[reviewed assessment](patches/codex/issue-41-reviewed-assessment.md). No licensed
Figueira horizon rerun was performed, so #41 is not closed.

**Qwen: smoke tests are not a completed SANS migration.** The patch changes only
`test_sans_operators.py`. It compares the same implementation with itself for
determinism; there is no migrated implementation against which parity can be
shown. The report says shared helpers are exclusively improving local search,
but `helpers/operators/perturbation_shaking/` and `evolutionary_mutation/`
already exist. Moving perturbation code does not require turning it into hill
climbing. The delivered tests only import intra-route operators despite the
handoff's inter-route claim; the handoff says ten tests/no patch while the actual
delivery is a thirteen-test patch. Keep this as optional smoke coverage, correct
the handoff, and leave M-qwen-02 / SANS M-cursor-01 open for implementation or an
explicit owner deferral. No algorithm rewrite was attempted by this review.

### 11.4 Validation and limits

- Submitted stack selected suite: **43 passed, three environment failures**.
  Expanded suite after the review amendments: **78 passed, three environment
  failures**, all the same Gurobi HostID mismatch. These are not code failures
  or passes. [Final log](patches/codex/open-review-20260929/final-tests.log).
- Status regressions fail before (two failures), then their related batch
  passes 11/11. Model/environment amendment files pass 15/15; fail-before crash
  evidence is stored separately. These counts overlap the final run and must
  not be added together. Passing smoke tests do not clear the held refactors.
- Compileall passes. Focused Ruff check passes. GitLab embedded schema/list
  validation passes for main/export scenarios; all eight shell blocks pass
  syntax checks; an actual shell argument test preserves quoted values without
  executing their contents. No real CI runner, credential operation, artifact
  upload, public push, full-suite claim or broad simulation run.
- All artifact paths and modes were checked before packaging. Amendment
  apply-check evidence is in
  [ready-stack log](patches/codex/open-review-20260929/ready-stack-apply.log).
  Only coordination/evidence artifacts were published in the shared checkout.
  No changes to restored data, production configuration in the shared tree,
  paper submodules, application commits or remote pushes.


### 11.5 Revised Gemini and Kimi artifacts — Codex follow-up, 2026-09-29

This section supersedes the Gemini and MS-helper holds in §11 for the exact
revisions below. Other lane verdicts remain as previously recorded. Review and
source edits ran in `/tmp/wsr-codex-open-revision-20260929`; only coordination
records and patch/evidence artifacts were written to the shared checkout.

**Gemini: ready with the new compatibility amendment.** Reviewed author patch
SHA-256 `3c5083305bd6bea27ac5e2e1a67730694f2986283792e97800d0ff548cb52ced`.
The revision fixes temporal-hook dispatch, horizon width, nested checkpoint
remapping and deepcopy. Three remaining compatibility failures were corrected:

- **HIGH:** removing depot overwrite globally also changed canonical
  `AttentionModelPolicy` checkpoints. Canonical VRPP/CVRPP/WCVRP depot
  projections now retain their previous defaults; only legacy `AttentionModel`
  explicitly selects the historical node projection for concatenated depots.
  Deterministic pre-fix depot error was 5 in each shared embedder.
- **HIGH:** the legacy factory constructor silently changed default activation
  from `ActivationConfig()` GELU to ReLU. An actual original-model probe loaded
  identical weights: initial embeddings matched, encoder max difference was
  0.3541065 and 8/21 greedy action entries differed for seed 7. The amendment
  restores all legacy activation defaults while preserving explicit overrides.
- **MEDIUM:** WCVRP with horizon 3 and `temporal_features=False` raised a linear
  shape error (2x3 versus 6x8). Restore historical zero-padding to the configured
  width; compare directly against the old context embedder.

Apply [issue-90-am-compatibility-review-amendment.patch](patches/codex/issue-90-am-compatibility-review-amendment.patch)
**after this revised Gemini patch**. Amendment SHA-256:
`a28a2f453f760469ca9a697ae3978c6a1fdc8b2da02522cd3e3e78820a5cd4f1`.
The old `issue-90-am-deepcopy-review-amendment.patch` is superseded and must not
be stacked: Gemini already integrated that correction.

Validation includes actual base-commit `AttentionModel` execution loaded from
`dfb049e7e`, an explicit real component factory, strict old-checkpoint loading,
and three seeds with three VRPP instances each. Greedy actions, rewards and log
probabilities match after the fix (tolerance 1e-6; embeddings/encoder max error
0). This is sampled VRPP legacy-path evidence, not universal parity across
all factories, environments or training trajectories. The earlier new-policy
versus new-model test is now explicitly a comparison in the same compatibility
mode, not evidence that their historical depot semantics were identical.

**Kimi MS helper revision: prior dependency hold closed.** Reviewed SHA-256
`967e2db4967a3f98515957f39b116c6695d41caaac27e6e42ca40124450c7ae1`.
The column-generation loop remains local and calls the local divergent pricing
helpers. AST comparison against the base confirms all nine retained top-level
functions are unchanged. Six remaining aliases include the shared fleet-bound
constraint handling already described by the lane; do not describe every
alias as text-identical. Kimi's unrounded gate records profits, not full routes:
its reported 0/24 changed values means profit equality. The 17/24 brute-force
oracle result still leaves pre-existing fleet-limit discrepancies; accepting
this refactor does not certify solver exactness. The 24-case gate was not
independently rerun by Codex in this follow-up.

**Kimi #41 remains open.** The revised report records setup blockers and no
instrumented archived-cell results. Its residual claims that forced sets
“always fit” and the failure chain “cannot occur” are too broad: increasing
fleet count only removes the aggregate fleet-capacity bound, and does not
prove feasibility under individual demands, arcs or other constraints. Use the
qualified [Codex assessment](patches/codex/issue-41-reviewed-assessment.md);
do not close #41 or infer a proven historical cause. Qwen's two SANS migrations
also remain open; the unchanged patch provides smoke tests only.

**Independent verification and integration:**

- Revised stack before the new amendment: 59 tests passed, including MS and
  all SWC backend tests. Gurobi works in this environment; the previous HostID
  failures are no longer a blocker for these tests.
- Final model/loader suite after all compatibility fixes: **53 passed**.
  Its focused parity subset has 23 tests; these counts overlap.
- Scoped Ruff passes for the five amended files and Kimi's changed modules.
- All eight lane patches plus four current Codex amendments apply to
  `dfb049e7e`; all five amended blobs match the tested tree. This application
  check includes Qwen smoke coverage and does not approve its incomplete tasks.
- Artifact inventories exclude data, paper paths and symlink modes. No shared
  production changes, CI runner execution, archived-cell simulation, push or
  full repository test-suite claim.

Exact manifests, logs and the historical forward reproducer are in
[open-revision-review-20260929](patches/codex/open-revision-review-20260929/).


### 11.6 Qwen SANS relocation review — 2026-09-29

The new `issue-90-mqwen-02-sans-operators-to-helpers.patch`
(`c717bf2bcd758274c9c084812b974258868505a362562adfc039ef795b450816`)
now performs a real seven-module relocation. This supersedes the prior
“smoke tests only” finding for the new artifact. **Request changes:** use the
assigned existing `helpers/operators/perturbation_shaking` hierarchy or obtain
an explicit architectural ruling; fix 37 scoped Ruff errors; correct stale
examples and coverage claims. The `_run_solver()` migration remains open.

Independent verification: seven byte-identical modules, 44 tests passed and
2,880 old/new operator comparisons passed (routes, results and RNG state).
No runtime stale imports found. The new patch replaces the earlier Qwen test
patch; do not stack both new-file additions. See
[full review](patches/codex/issue-90-qwen-review.md) and
[evidence](patches/codex/qwen-review-20260929/). No shared source edits.


### 11.7 Qwen relocation revision accepted — 2026-09-29

**M-qwen-02 ready:** revised patch SHA-256
`8c782f6c8bc9d955e8351921cb444f4c4bb21f90647cad7cbb47d22107107c8f`
closes §11.6 relocation findings. Destination now follows
`helpers/operators/perturbation_shaking/sans/`; lint and old import examples
are corrected. Independently: 44 tests pass, scoped Ruff passes, seven module
ASTs match base except module docstrings, and 2,880 behavior/RNG comparisons
pass. No stale namespace references remain. Complete revised stack applies.

Replace earlier Qwen patch versions; no Codex amendment needed. The separate
SANS `_run_solver()` migration remains open and already authorized by the brief;
only deferral needs an owner ruling. Do not close all of #90. See
[review update](patches/codex/issue-90-qwen-review.md) and
[evidence](patches/codex/qwen-revision-review-20260929/). Shared sources untouched.


### 11.8 Qwen combined SANS migration held — 2026-09-29

Combined patch `613460519e4b5a675463e08746b7fd54f23b243f5b8fa428e8a9b5b23c49b877`
is **held**. The added adapter mixes subset mandatory IDs with original global
bins/matrix and returns global routes to the base's local-to-global mapper.
Independent old/new execute probe: mandatory `[3]` becomes `[1]` in subset
mode; returning global `[0,3,0]` crashes with IndexError. The adapter also drops
`new_data`, changing the legacy engine's prepared-dataframe input. The 44 tests
pass but bypass the adapter; migration parity tests are missing. Two unused
imports fail Ruff. See [review](patches/codex/issue-90-qwen-migration-review.md)
and [evidence](patches/codex/qwen-migration-review-20260929/).

Standalone operator relocation `8c782f6c...107c8f` remains accepted; its patch
content is identical within the combined delivery. M-cursor-01 remains open.


### 11.9 Qwen final-named migration revision — hold remains

Reviewed `84d04c705bfd7559dc7faef0d8a4d7f3dfcd7184fcf21652c41e87f62016919c`.
Mandatory-only mapping and legacy `new_data` forwarding are fixed. Optional
nodes outside the subset still leak through `global_to_local.get(node,node)`:
with mandatory `[3]`, dispatcher `[0,1,3,0]` becomes `[0,3,3,0]`, and
`[0,2,3,0]` raises IndexError. **Hold remains.** 49 tests pass, but the subset
test bypasses the base's final mapping and the legacy test supplies no
`new_data`. Required old/new policy parity remains absent. Two Ruff errors
are in the new integration test. See [updated review](patches/codex/issue-90-qwen-migration-review.md)
and [evidence](patches/codex/qwen-final-review-20260929/).
Standalone operator relocation remains approved; shared sources untouched.


### 11.10 Qwen migration v2 held — route/profit inconsistency

Reviewed `fd6f3fbd8a6986fa259fce55d43f58f6b113555f46d4dcd837082cf5dde6d577`.
v2 filters non-subset nodes from the solved tour while retaining solver profit.
With mandatory `[3]`, unit economics/distances and fills `[10,20,30]`, solver
`[0,2,3,0]` / profit47 becomes `[0,3,0]` / cost2 / profit47; correct route profit
is28. **HIGH, hold remains.** Preserve historical full-bin/identity mapping
for the parity refactor or explicitly implement a consistently restricted
problem; do not post-filter routes. Integration assertions remain unchanged
and do not establish old/new policy parity. All49 tests and scoped Ruff pass.
See [review](patches/codex/issue-90-qwen-migration-review.md) and
[evidence](patches/codex/qwen-v2-review-20260929/).
Standalone operator relocation remains approved; shared sources untouched.


### 11.11 Qwen v3: implementation blockers closed, parity acceptance pending

Reviewed `70e594cbcdf798597ebba10f796d72342a9e470e709ba912bef2129a0d7e79f4`.
Full-bin identity mapping now preserves every prior failing tour/profit witness;
legacy `new_data` remains forwarded. 49 tests and scoped Ruff pass. Actual new
engine matches base in four seeded synthetic comparisons. Legacy matching
empty tours are failure fallbacks: instrumented old/new `find_solutions` both
raise IndexError on default `combination='best'` (pre-existing). This does not
prove successful legacy parity. Tests still bypass the full-bin override and
omit actual `new_data`; required persisted old/new and real-instance parity
remain pending. No new production bug found in this delta. See
[review](patches/codex/issue-90-qwen-migration-review.md) and
[evidence](patches/codex/qwen-v3-review-20260929/).


### 11.12 Qwen v4: legacy verification still false-positive

Reviewed `a7a1b613be5d9c77f409374f29252879f4151581eb22308f6a8423d483c86a72`.
Production unchanged from v3. The revised exact legacy test passes while real
`find_solutions` raises `KeyError('vehicle_capacity')`; dispatcher returns
`[0,0]`, accepted by `len(tour)>=2`. The tuple only bypasses the prior string
index error. Pre-existing Q/R versus vehicle_capacity/E contract mismatch is
not a new migration regression, but successful legacy parity is still absent.
49 tests and changed-file Ruff pass. Remaining execute-level/new_data/old-new
and real-instance gates unchanged. See [review](patches/codex/issue-90-qwen-migration-review.md)
and [evidence](patches/codex/qwen-v4-review-20260929/). Acceptance pending.


### 11.13 Qwen v5 held: zero-volume legacy contract and timeout

Reviewed `816b80918a60ab25834b42fe6524b6cc00e09da7571613cd744d756474f735f0`.
New `E=params.V` is zero in ordinary/test calls, zeroing legacy stock/revenue
while output profit still uses positive fill revenue. Keys alone do not repair
the unit contract. Exact authored legacy test times out45s; diagnostic stack
is in initial route uncrossing, before annealing's deadline check.48 tests pass,
legacy separately uncompleted; changed-file Ruff passes. Adapter unchanged
from v3; prior mapping fixes remain verified. Required parity tests still absent.
See [review](patches/codex/issue-90-qwen-migration-review.md) and
[evidence](patches/codex/qwen-v5-review-20260929/). Combined patch held.


### 11.14 Qwen v6 held — legacy objective units and skipped parity

Reviewed `a99c5cf8106428b98b5bdbb90d5f82f12b53f301eb8d2a145e8f0a70c1574fcf`.
Area volume fixes E=0 but double-applies density/volume to already-scaled
revenue. Controlled one-bin witness: legacy objective revenue100000 versus
reported revenue100 for identical visit; stock50000. Existing mandatory[1]
also reaches legacy solver as[2]. Repair contract units/IDs explicitly.
48 tests pass,1 legacy test skipped; two W293 Ruff errors. Skip does not close
parity gate, and no required old/new/real-instance evidence was added.
See [review](patches/codex/issue-90-qwen-migration-review.md) and
[evidence](patches/codex/qwen-v6-review-20260929/). Adapter unchanged fromv3;
standalone operator relocation remains approved.


### 11.15 Qwen v7 held — percentage units and override regression

Reviewed `4041ccc2ecfde9780e46dee1985826aeb44a856ca3f5e7307354fe6fd876eca4`.
Mandatory-ID increment and Ruff findings closed. Raw R still leaves missing
/100 conversion: stock50000 instead of500kg, objective revenue10000 versus
reported100 in the one-bin witness. New independent area reload also bypasses
merged revenue overrides: doubling resolved revenue doubles reported100->200
but leaves objective10000 unchanged. Both HIGH.48 pass,1 legacy skipped;
parity gate remains pending. See [review](patches/codex/issue-90-qwen-migration-review.md)
and [evidence](patches/codex/qwen-v7-review-20260929/). Shared sources untouched.


### 11.16 Codex implemented SANS corrections at owner request

Delivered `codex/issue-90-sans-reviewed-complete.patch`
(`099f4c1839d2535b94d112ef949a2c3b777e5036fe428539fdc9310d263e7a1f`),
replacing Qwen SANS patches, plus two optional additive amendments for v7.
Percentage units/resolved overrides now agree; legacy uncrossing terminates on
the failing collinear fixture; operator probability length and global-RNG use
are repaired; all routes/caller fields survive; failures are surfaced. Replaced
weak/skipped tests: **59 passed, no skips**, scoped Ruff passes, three fail-before
regressions verified, eight actual-engine seeded adapter comparisons pass.
Complete and additive patch paths match tested sources; full reviewed stack
applies. No shared production edits. [Handoff](patches/codex/issue-90-sans-fix-handoff.md)
records intentional engine behavior changes, unsupported preset-name errors,
and the unrun archived real-instance/performance gate. Evidence:
[files](patches/codex/sans-fix-20260929/).
