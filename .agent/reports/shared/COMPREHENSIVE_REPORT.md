# WSmart+ Route Research, Codebase, and Manuscript Report

> **Status:** living collaborative draft
>
> **Started:** 2026-08-28
>
> **Current verdict:** major revision; several experiment-description defects must be resolved before submission
>
> **Scope:** the research contribution, implemented framework, archived experiment, analysis pipeline, and manuscript *Simulation Framework for the MPVRP with Profits in Smart Waste Collection*

## 1. Purpose and editing protocol

This is the shared report of record. It is intended to converge, through edits by the user and all reviewing agents, into a comprehensive and evidence-backed assessment of the project. Individual reports remain useful as signed reviews, but disagreements should be resolved here against primary artifacts rather than by majority vote.

When adding a material claim, give an evidence path or mark it as unresolved. Use these status labels:

- **VERIFIED**: directly reproduced from tracked code, configuration, data, logs, generated output, or manuscript source.
- **CORROBORATED**: independently reported by multiple reviewers, with a plausible evidence trail that still needs a final primary-artifact trace.
- **OPEN**: important question or suspected defect that has not been established.
- **RESOLVED**: previously open issue for which the paper, implementation, artifact, or report has been corrected and checked.

Do not silently remove another contributor's supported finding. Correct it in place, record the reason in the disagreement log, and add a changelog entry. Separate the following three propositions whenever they differ:

1. the framework supports a capability;
2. the archived experiment exercised that capability;
3. the paper provides evidence about its performance.

Publication assets under the paper's `Tables/` directory are generated artifacts and must never be hand-edited. Correct their generator or source data and regenerate them.

## 2. Executive assessment

The project contains a potentially publishable contribution: a modular, multi-day smart-waste simulation framework that separates **mandatory bin selection**, **route construction**, and **route improvement**, then evaluates policies on real Portuguese road networks. This decomposition makes an operationally valuable point visible: deciding *when and what to collect* can move the efficiency-service trade-off more than swapping the route constructor. The paper's integrity-aware analysis is also unusually candid. It detects pathological SWC-TCF results, excludes whole affected comparison cells rather than only the failing method, balances marginal comparisons, and explicitly limits causal claims for the improver and 90-day studies.

The manuscript is not submission-ready. Several core descriptions do not match the archived experiment or implementation. Most seriously, all 36 archived experiment configurations set `sim.n_vehicles: 0`, while the manuscript states that every scenario is restricted to one vehicle. In the routing code, zero means an unlimited or automatic number of routes/vehicles. The Service-Level equation, Fast-TSP description, Look-Ahead description, demand-process language, and 90-day sampling account also require correction. The headline “nearly four times” comparison is arithmetically false for the published marginals. These are method and validity defects, not cosmetic edits.

The correct editorial stance is **major revision, not rejection**. The work's strongest ideas should survive a narrower and more exact paper. The authors should distinguish the evaluated classical benchmark from the framework's wider solver library, state the operational problem that was actually run, repair the claim-to-artifact lineage, and rerun only where text corrections cannot restore validity.

### 2.1 Highest-priority actions

1. Resolve and disclose the fleet semantics. Either reframe the archived study as an automatic/unbounded-route experiment or rerun with a verified one-vehicle limit. Add route-count, payload, and shift-feasibility telemetry.
2. Make the Service-Level mathematics and code agree, disclose `confidence_factor: 0.84`, and regenerate affected results if the paper's square-root rule is intended.
3. Replace the Fast-TSP subsection with the algorithm actually benchmarked, or rerun the intended DP hybrid.
4. Reconstruct the exact rule that produced the 174-row 90-day sample. The tracked CSVs do not support the current literal “only Pareto-front configurations” statement.
5. Recompute the headline selection-versus-constructor comparison on one common balanced slice. The printed ranges give about `2.77×`, not nearly `4×`.
6. State clearly that the benchmark has one demand realization per cell and no learned-solver observations.
7. Trace constructor objective, simulator reward, economic profit, kg/km, and overflow count; report the physical and economic constants with units and sources.
8. Add an immutable experiment manifest and per-run completion/solver/fallback status.
9. Run the improvers against identical stored constructor tours before interpreting their difference causally.
10. Tighten the paper substantially, repair the figures and bibliography, and move implementation catalogue material to an artifact or appendix.
11. Add a code and data availability statement to the manuscript itself. The current text contains no repository URL, DOI, license, or availability statement of any kind — a near-disqualifying omission for a paper whose first contribution bullet is "a reproducible simulator".

## 3. Research contribution and boundaries

### 3.1 What is genuinely strong

The three-stage policy abstraction is the clearest contribution. Waste-routing systems often bundle dispatch thresholds and spatial optimization into one named policy, making it impossible to tell whether a result comes from collection timing or route geometry. This framework exposes those decisions as separate interfaces and lets selection, construction, and improvement components be composed. The implemented day pipeline reflects that idea:

`Fill → Mandatory Selection → Route Construction → Route Improvement → Collection → Logging`

Primary implementation evidence is under `logic/src/pipeline/simulations/states/`, `logic/src/pipeline/simulations/actions/`, `logic/src/policies/mandatory_selection/`, `logic/src/policies/route_construction/base/`, and `logic/src/policies/route_improvement/base/`.

The benchmark also has several credible design choices:

- a complete 30-day grid of 480 stored configurations across eight classical constructors, five selection variants, two improvers, two demand processes, and three network sizes;
- paired demand within the stored grid rather than policy-specific demand generation;
- two real regional bin networks and road-distance matrices;
- explicit service and efficiency outcomes instead of a single opaque score;
- whole-cell removal for anomalous SWC-TCF observations and balanced marginal comparisons;
- unusually frank discussion of the improver confound and outcome-conditioned horizon follow-up.

The integrity protocol may be the most reusable methodological contribution. It should be presented as a general benchmark rule: when an invalid-run filter is correlated with the compared factor, remove or reweight comparison units symmetrically, show the accounting, and retain the raw failure evidence.

### 3.2 What the evidence does not establish

The archived paper experiment does not evaluate Neural Combinatorial Optimization. Every 30- and 90-day row uses one of eight classical constructors. The framework's neural adapters and model catalogue may justify an extensibility claim, but not an empirical NCO claim.

The benchmark does not establish general stochastic superiority because there is one stored demand realization per cell. It does not identify an improver treatment effect because CLS and Fast-TSP did not receive identical upstream constructor outputs. It does not establish cross-constructor 90-day performance because the long-horizon sample was selected using 30-day results. It does not test sensing robustness because the experiment uses true observations with zero noise. It does not test working-shift feasibility. As currently configured, it also does not test the claimed one-vehicle operation.

### 3.3 Claim map

| Claim | Current evidence | Status | Required qualification or action |
|---|---|---:|---|
| The framework separates selection, construction, and improvement | Explicit interfaces and simulation actions | VERIFIED | Keep and strengthen with stage contracts and invariants |
| Selection spans a wider observed trade-off than constructor choice | Generated 30-day marginals | VERIFIED, magnitude misstated | Recompute on common slice; remove `4×` |
| Classical constructors are benchmarked on real networks | 480 raw 30-day logs and summary rows | VERIFIED | State exact artifact snapshot and limits |
| NCO is evaluated | No learned-solver rows | CONTRADICTED | Scope NCO to framework capability |
| The experiment is single-vehicle | All 36 archived configs set `n_vehicles: 0`; code treats zero as automatic/unlimited | CONTRADICTED | Rerun or relabel operational setting |
| Only 30-day Pareto-front configurations reached 90 days | Current 30-/90-day CSV relationship does not reproduce this rule | CONTRADICTED AS WRITTEN | Recover selection script/snapshot or rewrite |
| BPC observations represent an exact method | Archived BPC uses `exact_mode: false` and finite limits | CONTRADICTED AS AN EMPIRICAL CLAIM | Say time-limited BPC family; report gaps/status |
| Empirical demand replays observed patterns | Independent per-bin empirical marginals | OVERSTATED | Say marginal empirical resampling |
| Gamma parameters are in kg/day | Generator output is percentage-point fill | CONTRADICTED | Correct units and show mass conversion |
| Generated tables are internally consistent | Independent recomputation of every count and derived mean from the manuscript alone (§7.4) | VERIFIED | Keep; extend generation to derived prose constants so handwritten claims cannot drift |
| Manuscript states its code/artifact availability | No repository URL, DOI, or availability statement anywhere in `paper.tex` | CONTRADICTED | Add availability statement tied to a versioned release |
| Routing solvers optimize overflow penalties directly | Objective Eq. (2) includes only collected revenue and distance cost; zero overflow penalty in VRPP layer | CONTRADICTED | Clarify that overflow prevention relies entirely on upstream mandatory selection constraints |
| Smart bin observation noise is benchmarked | Simulator supports $\epsilon$, but all archived runs set `sim.noise_std: 0.0` | UNEXERCISED | Distinguish simulator noise capability from the zero-noise empirical benchmark |
| SWC-TCF failures were handled gracefully | Monolithic MIP timed out on $N=350$ and returned an empty tour, logged as an executed 0-collection run | CONTRADICTED | Add explicit `SolverTimeout` telemetry and fallback handling |

## 4. Operational problem and mathematical model

The formal model is useful as a conceptual VRP-with-profits skeleton, but it is not yet an exact specification of the simulator. A reproducible formulation needs to define the true fill state, observed fill state, lost mass, collection reset, mandatory set, optional set, daily routing decision, fleet/trip semantics, and the information available when each decision is made.

### 4.1 Fleet, capacity, trips, and shifts — publication blocker

**VERIFIED.** Every one of the 36 tracked `assets/output/30days/**/hydra/pruned_config.yaml` files sets `sim.n_vehicles: 0`. The manuscript states at `paper.tex:694–698` that the experiment restricts every scenario to one vehicle and one depot. Current routing contracts interpret zero as no positive fleet limit or automatic/unlimited routing. Examples include `logic/src/interfaces/context/problem_context.py`, `logic/src/policies/route_construction/other_algorithms/capacitated_vehicle_routing_problem/cvrp.py`, and the unlimited split path in the HGS family.

The daily logs reinforce the practical consequence. Across 4,800 stored Figueira da Foz day-results, 565 collect more than the physical 2,500 kg payload in a day. The maximum is 7,094.54 kg, requiring at least three capacity loads. This is compatible with several depot-separated routes under automatic fleet sizing, but not with one single-trip vehicle route. Rio Maior's stored maximum is 3,494.58 kg against its 3,500 kg payload.

The JSON `daily.tour` field cannot recover the original route count: `logic/src/pipeline/simulations/day_context.py` removes every depot separator and wraps the concatenated IDs in one pair of zeros. That logging transformation is itself a provenance defect. Preserve routes as a list of lists or preserve internal zeros.

The revision must distinguish five resources that are currently blurred together:

1. configured fleet limit;
2. payload capacity per route;
3. number of same-day depot trips;
4. whether routes are simultaneous vehicles or sequential trips;
5. driver working-shift duration.

If the intended operational interpretation is one vehicle making unlimited trips, say so and add a daily time/shift constraint. If the intended interpretation is one trip by one vehicle, the experiment must be rerun.

### 4.2 State and overflow accounting

The prose should define lost mass explicitly rather than only capping the fill state. A recurrence such as `min(100, state + increment)` alone makes excess disappear mathematically. The implementation separately records loss, so the formalization should include a loss variable and specify whether an overflow event is counted on crossing capacity, on every full bin-day, or both. The manuscript currently uses wording around “at capacity” and “beyond capacity” inconsistently.

The information structure also needs a clean split between true state and sensed observation. The framework can add observation error, but the archived experiment uses the true state. The simulation-loop figure should not visually center noisy observations without marking that branch as unexercised.

### 4.3 Objective alignment

The paper's optimization model maximizes monetary profit, while the main experimental ranking uses kg/km and overflows. The simulator also has a `reward` field that differs from economic `profit`. These are not interchangeable:

- constructor objective: generally collected-value minus travel cost, with solver-specific penalties and constraints;
- simulator reward: collected kilograms minus overflow events minus distance in the stored logging path;
- economic profit: material revenue minus distance expense;
- reported research outcomes: kg/km, overflow events, distance, mass, runtime, and sometimes lost mass.

#### Mathematical Decoupling of Single-Period Routing vs. Multi-Period Overflow Penalties (Gemini)

A foundational mathematical property that explains the empirical dominance of the Selection stage over the Constructor stage is the **absence of an overflow penalty in the daily routing objective function**.

In Equation (2) (`paper.tex:189`):
$$\mathcal{P}(\mathcal{A}_d, \bm{w}_d) = r_w \sum_{v_i\in\mathcal{A}_{d,\cdot}}w_{i,d} - c_{km}\sum_{k=1}^{K}\sum_{t=0}^{T_{d,k}} \dist(a_{t,k},a_{t+1,k})$$

The routing solvers maximize net profit (collected revenue minus travel cost). There is **no term penalizing unvisited bins that overflow** (e.g., $-\sum_{i \notin \mathcal{A}_d} c_{\text{ovf}} \mathbb{I}(w_{i,d} \ge C_i)$). Consequently, from the pure optimization perspective of the single-period route constructor, an overflowing bin located on a distant or isolated branch offers zero net incentive if the detour cost exceeds $r_w w_{i,d}$. 

The *only* mechanism forcing the vehicle to visit critically full bins is the **Mandatory Selection stage**, which imposes hard equality constraints ($\sum_k \sum_t x_{i,t,k} = 1$) on the candidate graph. This mathematical structure proves why selection strategy choice drives the bulk of service-level variance: the routing constructors are fundamentally agnostic to future temporal overflow costs unless forced by upstream mandatory constraints.

The paper must give the physical/economic constants and units. The current repository returns plastic revenue `0.65 × 898 / 1000 = 0.5837 €/kg`, distance expense `1 €/km`, a 2.5 m³ bin volume, densities of 19 and 20 kg/m³, and physical payloads of 3,500 kg in Rio Maior and 2,500 kg in Figueira da Foz before converting payload to percentage-fill units. Because code may have evolved since the stored runs, the final paper should cite values from a versioned experiment manifest, not merely today's defaults.

## 5. Method-to-code fidelity

| Component | Manuscript account | Tracked implementation/artifact | Status and action |
|---|---|---|---|
| Service-Level | Projection uses `z σ √n_d` | Scalar and vector paths use `z σ n_d`; archived `z=0.84` is not reported | VERIFIED mismatch; choose rule, disclose value, rerun if changed |
| Look-Ahead | Forward stochastic/look-ahead simulation; described as costly | Deterministic mean-rate projection with trigger synchronization and bundle expansion | VERIFIED mismatch; describe exact heuristic and remove unsupported cost claim |
| Fast-TSP | Small routes use exact DP, larger routes randomized search | `FastTSPRouteImprover` calls `fast_tsp.find_tour` under a 30 s limit; DP logic is in a different class | VERIFIED mismatch; rewrite subsection or rerun intended method |
| Gamma-3 | Means/variances described as kg/day | Generator produces percentage-point fill increments; patterns are tiled by bin index | VERIFIED mismatch; correct units and assignment mechanism |
| Empirical demand | “Replays” observed patterns | Samples each bin's empirical marginal independently | VERIFIED overstatement; state lost temporal and cross-bin dependence |
| Common random numbers | Prose emphasizes reseeding policy-days | Archived configs load a shared seed-42 NPZ; optimizer randomness has a separate policy/day seed path | VERIFIED for current code/config, historical trace incomplete; record hashes and RNG streams |
| BPC | Exact-method family | `exact_mode: false`, finite 60 s limits, heuristic options/fallbacks; no certificates in summaries | VERIFIED qualification; do not imply observed optimality |
| SWC-TCF | Monolithic exact two-commodity MIP | $O(V^2)$ arc variables ($>122,500$ arcs on FF350) hit Gurobi 60 s timeout; returns empty/depot route on truncation | VERIFIED failure mechanism; add explicit solver timeout status |

The Service-Level mismatch is material. For SL2, the implementation's uncertainty term grows linearly with horizon rather than with the square root of horizon. A prose correction alone is legitimate only if the implemented rule was intended and can be defended. If the square-root aggregation is the intended statistical model, all affected rows must be regenerated. The printed equation has a further defect independent of the code: its threshold is written `≥ 100%` while the state definition makes $w_{i,d}$ an absolute mass in $[0, C_i]$ (paper.tex Eq. 6 vs. Sect. 2.1), so the rule as printed compares an absolute fill projection against a percentage. Whichever rule is adopted must normalize fill by $C_i$ — or compare against $C_i$ directly — consistently.

The Look-Ahead rule also needs a name that matches its behavior. Its trigger resembles a deterministic threshold-crossing projection, followed by synchronized collection of bins predicted to become critical within the same horizon. It is not a Monte Carlo policy and does not propagate sampled future states.

#### Monolithic MIP Complexity and SWC-TCF Truncation Failure (Gemini)

The SWC-TCF constructor directly implements the two-commodity flow formulation of Ramos et al. (2018). While it compactly models MTZ sub-tour elimination and capacity tracking without exponential lazy constraint generation, its size scales as $\mathcal{O}(V^2)$ continuous commodity flow variables ($u_{ij}, v_{ij}$) and $\mathcal{O}(V^2)$ binary routing variables ($x_{ij}$). 

On Figueira da Foz ($N=350$), the formulation instantiates over $122,500$ potential directed arcs. Under Gamma-3 (higher daily arrival mass), the LP relaxation bound is weak, creating an immense branch-and-bound search tree. When Gurobi reached its 60 s wall-clock time limit without finding an integer-feasible incumbent, the wrapper returned an empty route. The simulation framework recorded this as a zero-collection day rather than raising an execution error or recording `SolverTimeout`, causing the cumulative 30%–86% tonnage shortfalls and 23,886 truncated overflow events identified in Table 6. Future solver wrappers must emit explicit solver termination status codes (`OPTIMAL`, `TIME_LIMIT`, `INFEASIBLE`, `FALLBACK_USED`).

## 6. Archived experiment and statistical validity

### 6.1 What is stored

The tracked summaries contain 480 rows at 30 days and 174 rows at 90 days. The 30-day file has 60 rows for each of eight constructors. All 174 90-day configuration keys have a corresponding 30-day key. The archive contains 480 raw 30-day JSON logs and 36 pruned configurations. The source seed-42 demand NPZ files referenced by those configurations are not tracked under the ignored `data/wsr_simulator/` tree, so a clean checkout cannot reproduce the stochastic inputs from the paper artifact alone.

The 480 rows are factorial cells, not independent stochastic replications. Every archived configuration has `n_samples: 1` and seed 42. The paper can describe paired outcomes for this one realization, but it cannot estimate sampling uncertainty, interaction stability, or probabilities of superiority.

### 6.2 The 90-day carry-forward rule does not reproduce

**VERIFIED against the tracked CSVs.** The 174 rows correspond to 29 distinct non-geographic policy definitions, each repeated across all six network/distribution scenarios. Their constructor coverage is:

| Constructor | 30-day rows | 90-day rows | Distinct carried policies |
|---|---:|---:|---:|
| BPC | 60 | 60 | 10 |
| PG-CLNS | 60 | 42 | 7 |
| ACO-HH | 60 | 30 | 5 |
| HGS | 60 | 12 | 2 |
| PSOMA | 60 | 12 | 2 |
| SANS | 60 | 12 | 2 |
| SWC-TCF | 60 | 6 | 1 |
| ALNS | 60 | 0 | 0 |

Recomputing row-level non-dominance within each of the six 30-day scenarios yields 33 Pareto rows, of which only 23 appear in the 90-day set. Aggregating each policy definition over all six scenarios yields 13 non-dominated policies; only 10 are in the 29-policy carry-forward set, while 19 carried policies are dominated under that definition and three non-dominated policies are omitted. Thus neither the per-scenario row-level rule nor the global policy-aggregate rule reproduces “only policy configurations that lay on the 30-day Pareto front were carried forward.”

This does not prove that the original selection was arbitrary. It may have used an earlier summary snapshot, another objective, another aggregation, or a constructor-level expansion rule. No tracked selection manifest currently shows which. Until the exact rule is recovered, the paper should call this a performance-selected 90-day subset without claiming literal Pareto membership. The within-configuration 30-versus-90 contrasts remain descriptive for selected configurations, but selection on the 30-day outcome can induce regression-to-the-mean effects and does not estimate a population horizon effect.

The manuscript's own horizon accounting is internally consistent once reconstructed: it announces 174 90-day runs (paper.tex:777) but its horizon table pairs only 165 configurations (paper.tex:951). The nine lost pairs decompose by constructor as BPC −3, PG-CLNS −2, ACO-HH −2, PSOMA −1, SWC-TCF −1 (HGS and SANS unchanged) — exactly the configuration set whose 30-day runs sit in the integrity-excluded FF350/Γ3 cells. The paper never bridges the two numbers; one sentence would (RCP-018).

### 6.3 Improver comparison

The current CLS-versus-Fast-TSP matching shares scenario and demand labels, but not fixed constructor output. Upstream stochastic constructors can choose different bins and tours. “CLS wins 202 of 224” therefore describes paired configurations, not a controlled route-improvement experiment. Store constructor tours once, apply both improvers to every identical route with controlled seeds, and compare node preservation, feasibility, distance change, runtime, and failures.

### 6.4 Runtime

Archived configurations allow 20 concurrent CPU workers. Unless solver thread counts, affinity, hardware, and scheduling were fixed, wall-clock observations combine algorithm cost with contention. Retain current timings as experiment-throughput observations, not clean standalone solver benchmarks. A small isolated rerun with one worker and fixed internal threads would make the runtime ranking defensible.

### 6.5 Failure handling

The whole-cell exclusion strategy is strong, but the trigger is an outcome-based tonnage shortfall because raw results do not uniformly record solver termination and completion. Add fields for requested/completed days, solver status, incumbent, bound/gap, timeout, exception, fallback, route feasibility, mandatory coverage, capacity violation, and shift violation. Then exclusions can be based on process validity rather than unusually poor performance.

## 7. Results audit

Generated table cells and row accounting are generally reliable. Handwritten derived claims are less well synchronized.

### 7.1 Headline range comparison

From the published marginal tables:

- selection range: `7.381 − 3.809 = 3.572 kg/km`;
- constructor range: `6.649 − 5.359 = 1.290 kg/km`;
- ratio: `3.572 / 1.290 = 2.77`.

The phrase “nearly four times” is false for those numbers. Moreover, the selection and constructor tables use different balanced slices (`n=400` and `n=456`), so even `2.77×` is not the best final estimate. Regenerate both ranges on one common eligible slice or state only that selection has the wider observed range.

Other handwritten comparisons need a generated assertion or prose-value test. Reported review examples include “within 5%” where the difference is 5.26%, an imprecise runtime fraction, and a claimed N=350 separation that is effectively a tie. The durable solution is to generate derived prose constants from the same analysis module and fail the publication build when literal claims drift.

### 7.2 Pareto presentation

The constructor aggregate in the current Pareto figure includes ALNS as non-dominated, but the dashed frontier omits it and connects only PG-CLNS to BPC. Regenerate the front from a single dominance function used by tables, figures, website exports, and 90-day selection. Add uncertainty only after replicated seeds exist; until then, describe points as one-realization outcomes. The in-text Pareto-membership enumeration (paper.tex:820–823: PG-CLNS 5 of 6, PSOMA and HGS 3 each, BPC 2, ACO-HH and ALNS 1 each) sums to 15 and never states that SWC-TCF and SANS hold zero memberships; the sentence should enumerate all eight constructors so the total is checkable.

### 7.3 Interpretation that remains valuable

The monotone movement of the selection variants across efficiency and service is operationally interesting. Later collection improves kg/km and reduces distance, while earlier collection reduces overflow. Remote depot legs plausibly strengthen this trade-off by imposing a fixed cost on each dispatch. That mechanism is plausible, not identified: a nearby-depot ablation and fixed-service-level comparison are needed before assigning causality.

Constructor differences appear more consequential in runtime and tail failures than in central overflow counts. That is a useful finding if stated as descriptive evidence from the current grid.

### 7.4 Independent arithmetic audit of the generated tables (opencode)

A second pass recomputed the accounting of every generated table from the manuscript and `Tables/*.tex` alone, without consulting the archived CSVs. Every check reproduces exactly:

| Quantity | Derivation | Result |
|---|---|---|
| 456 runs, 57 per constructor | 480 − (3 degenerate 30-day cells × 8 constructors); 456/8 = 57 | ✅ |
| Selection marginal n = 80 per variant | 96 constructor×improver×scenario slices; whole-cell exclusion leaves LA absent from FF350/Γ3 (both improver slices) and SL2 partially absent, so all 16 FF350/Γ3 slices fail the all-five-variants test → 80 per variant (400 runs) | ✅ |
| 224 improver pairs | 240 constructor×selection×scenario pairs − 8 LA pairs (both improvers excluded) − 8 SL2 pairs (FTSP excluded) | ✅ |
| Scenario marginals n = 216 and 136 | demand: 240 config×network units − 24 lacking one process; network: 160 config×process units − 24 lacking FF350 under Γ3 | ✅ |
| Horizon table pair total | 28+57+12+40+11+12+5 = 165 | ✅ |
| "+0.26 kg/km" paired horizon mean | Σ(pairs × per-constructor Δ)/165 = 42.31/165 = 0.256 from the table's own rounded cells | ✅ |
| "roughly seven times" per-overflow contrast | (751 km / 2.2 events) / (456 km / 9.4 events) = 341 / 48.5 = 7.0, from table deltas | ✅ |
| 4.5× runtime spread at N=350 | 5,451 / 1,219 = 4.47 | ✅ |
| Gamma-3 moments | αβ and αβ² for (α, β) ∈ {1,3}×{8,6} reproduce all four means (8, 6, 24, 18) and variances (64, 36, 192, 108) | ✅ |
| Improver Δ row | 6.39 − 5.65 = 0.74 kg/km; 4,358 − 4,662 = −304 km | ✅ |

Two conclusions follow. First, the generated numerics are internally consistent to the last digit: every defect catalogued in this report lives in handwritten prose, method descriptions, or claim-to-artifact lineage — never in the generated cells. This is direct evidence that the `gen_paper_latex.py` discipline works, and it should be extended to derived prose constants (§7.1) so that literal claims like "nearly four times" cannot survive a regeneration. Second, mean-derived 90/30 overflow ratios computed from the horizon table's own columns are ACO-HH 3.59, BPC 3.52, HGS 2.78, PG-CLNS 3.26, PSOMA 3.17, SANS 3.11, SWC-TCF 5.82. The prose's "between 2.4 and 3.3" cites medians the table does not show, and two constructors' mean ratios already exceed the stated band (RCP-019).

## 8. Codebase assessment

### 8.1 Strengths

The codebase has a substantial modular architecture: typed contexts, registries and factories for policy stages, state-machine simulation, Hydra configuration, generated publication assets, real network data paths, and a large test hierarchy. There are 233 tracked test files, including simulator and solver-contract tests. The `just paper` path and `logic/gen/gen_paper_latex.py` are good foundations for keeping presentation synchronized with summaries.

The framework breadth is impressive, but registered or implemented does not mean experimentally validated. The shared assessment should classify every advertised solver/model as one of: registered, importable, smoke-tested, contract-tested, benchmarked, source-paper fidelity reviewed, experimentally exercised, or exactness-certified. Internal review documents that use universal “perfect,” “textbook,” or “world-class” ratings are self-assessments, not independent validation.

### 8.2 Stage contracts and invariants

Each stage should document:

- inputs and units;
- outputs and representation;
- mandatory-node preservation;
- stochastic state and seed;
- capacity, fleet, and shift obligations;
- failure/fallback behavior;
- metrics emitted for audit.

Experiment-level invariant tests should verify identical demand hashes across compared policies, route node preservation across improvers, mandatory coverage, per-route payload feasibility, fleet/trip limits, complete-horizon status, deterministic regeneration, and exact summary row accounting.

### 8.3 Configuration and environment drift

The live code contains legacy and modern configuration paths, recursive flattening, and policy-name inference. Current RNG behavior should not be assumed to describe the archived run without a recorded code commit. Project-level documentation also says Python 3.9+ and PyTorch 2.2.2, while the live package metadata and lock target newer requirements. The artifact must record the environment that generated the paper separately from the environment supported today.

### 8.4 Maturity

The framework is a serious research platform, but “production-ready” is not yet supported by the evidence reviewed here. The ongoing bug roadmap, incomplete policy-family integration coverage, non-blocking type/security checks, modest coverage floor, missing experiment lineage, and contradictory operational semantics are consistent with a capable research prototype moving toward an internally reproducible platform. Production claims require deployed reliability evidence, monitoring, schema/version governance, and operational constraint validation.

## 9. Reproducibility and artifact readiness

Distinguish three levels of reproducibility:

1. **Document regeneration:** compile the PDF from existing figures and tables.
2. **Analysis regeneration:** rebuild tables, plots, and website data from stored summaries/logs.
3. **Experiment reproduction:** regenerate demand, run policies, validate raw outputs, create summaries, and reproduce the paper.

The repository is strongest at levels 1 and 2. Level 3 is incomplete because the demand datasets are ignored, the archived code commit and environment are absent from logs, solver and fallback status are missing, and no canonical experiment manifest binds inputs to outputs.

A publishable artifact should contain:

- parent and nested-submodule commits;
- checksums and licensing status for demand, coordinates, and distance matrices;
- fully resolved Hydra configs;
- Python/package lock hash and external solver versions;
- hardware, worker count, thread count, and license mode;
- separate demand and optimizer RNG stream identifiers;
- start/end/completed-day and solver status telemetry;
- one canonical reproduction command with expected row counts and hashes;
- a data-availability statement and legally shareable surrogate where source data cannot be distributed;
- `CITATION.cff`, licenses, and a versioned archival release.

## 10. Manuscript quality

### 10.1 Abstract and introduction

The motivation is concrete and socially relevant, but the abstract overpromises. It foregrounds NCO among adapted solvers and describes an extensive solver benchmark without telling the reader that every reported observation is classical. It also omits the most credible differentiators: the three-stage decomposition, real Portuguese networks, and integrity-aware factorial analysis. If the conference abstract of record cannot change, place an unmistakable scope statement immediately after it and remove NCO from any mutable keyword or empirical claim.

The introduction should lead with the research question: how much of long-horizon collection performance is attributable to selection timing versus route construction and route improvement? The current framework catalogue and NCO survey dilute that question.

### 10.2 Related work

The taxonomy of exact, heuristic, and neural solvers is organized, but its center of gravity is too solver-centric. The paper needs stronger engagement with inventory routing, periodic VRP, prize-collecting/team-orienteering variants, stochastic dynamic waste collection, operational shift/fleet constraints, and simulation benchmark frameworks. Those literatures define the actual problem boundary more directly than a long catalogue of neural architectures that are not evaluated.

### 10.3 Discussion and conclusion

The discussion's epistemic restraint is one of the paper's strengths. Preserve its distinctions between association and causal effect, selected follow-up and full factorial inference, and typical behavior and tails. Apply the same discipline to the abstract, fleet claim, exactness terminology, and 90-day sampling description.

The conclusion is repetitive and stylistically weaker. Replace the long future-work list with three priorities: replicated factorial evidence, operational-feasibility/telemetry correction, and controlled component ablations. The current 34-page length is risky for LNCS proceedings; move solver catalogues, granular heatmaps, and configuration details to the artifact or supplementary material.

### 10.4 Writing

The main prose is generally competent but sometimes sounds generated: contrast templates such as “not X but Y,” repeated caveat restatements, long em-dash chains, inflated adjectives, and abstract nouns in place of direct verbs. Prefer short factual sentences and state each limitation once at the point where it constrains a claim. Standardize names (`WSmartRoute+` in the body vs. `WSmart Route+` in the acknowledgments vs. `WSmart-Route` in the repository) and correct the remaining conclusion grammar and typographical errors. Specific line-anchored corrections: `paper.tex:1172` “on difference temporal horizons” → “different”; `paper.tex:1188` “a upstream phase” → “an upstream phase”; `paper.tex:461` “unfeasible” → “infeasible”. The LaTeX build itself is clean — no undefined references and no overfull boxes in `paper.log` — but the PDF is produced on US letter, where LNCS production expects the class's own page geometry.

## 11. Figures, tables, and accessibility

| Asset | Current issue | Required improvement |
|---|---|---|
| Policy configuration space | Caption refers to a highlighted subset that is not visibly highlighted | Highlight evaluated components or remove the claim |
| Simulation loop | Noisy observation path dominates despite zero-noise experiment; arrow crosses text | Separate supported capability from exercised path; repair layout |
| Regional maps | Selected bins are visible, but remote depots and scale are absent; caption conflates road matrix and OSM basemap provenance | Add scale bars and depot inset/connector; state each data source/license precisely |
| Aggregate Pareto plot | Dashed front omits non-dominated ALNS | Use one tested dominance implementation |
| Strategy trade-off | Long labels overlap; dual-axis bars obscure the trade-off | Use a direct efficiency-versus-service plot or small multiples |
| Runtime/scaling | Sparse log ticks and similar colors impede reading | Add useful ticks, direct labels, and colorblind-safe encodings |
| Appendix heatmaps | Useful but visually detached from main-paper style | Regenerate with shared typography and palettes |
| Appendix full table | Rasterized text is not searchable or accessible | Generate vector/PDF or LaTeX; never hand-edit `Tables/` |
| Appendix policy-level Pareto | Its own caption admits the excluded degenerate run (≈2,168 overflows) sets the Gamma-3 panel's horizontal axis extent | Clip the axis, use a log scale, or draw the excluded point as an off-scale annotation |
| Fill-trajectory figure | Honest caption (“re-simulated from the recovered daily increments”); the single-bin, fixed-increment view is the clearest mechanism illustration in the paper | Keep; add the same view for a Service-Level variant to show how the projection rule times collection |

Every figure caption should state the population/slice, horizon, exclusion rule, aggregation, and whether uncertainty is available. Maps should distinguish coordinate source, road-distance source, and basemap source. Main-text figures are raster PNGs of 857–1425 px width (≈200 DPI at `\linewidth`); export vector PDF from the plotting pipeline. The appendix CLS table image is 3060×1116 px and cannot be searched, screen-read, or restyled.

## 12. Citations and scholarly positioning

All cited keys currently resolve, but resolution is not the same as bibliographic correctness. Items already flagged for repair include:

- `WENTGES2006`, whose year and DOI metadata disagree;
- incomplete or malformed Lysgaard metadata;
- `Lin2017`, which may not support the Farkas-pricing claim for which it is cited;
- legacy key `BARNHART1970` for a 1998 publication;
- doubled or malformed DOI rendering, including the Sun entry;
- generic export keys that make maintenance harder;
- `ma2024learning`, cited in text as 2024, is a NeurIPS 2023 paper (key year disagrees with venue year);
- `Kool2018AttentionLT` keys 2018 but the venue is ICLR 2019;
- roughly thirty bib entries are never cited (e.g. `vaswani2017attention`, `wu2019graph`, `kingma2014adam`, `Paszke2017AutomaticDI`, `sutton1999policy`, `williams1992simple`), and several entries carry full abstracts — maintenance hazards for the next revision;
- ambiguous Google Maps/OpenStreetMap attribution and licensing.

The final citation pass should verify author order, title, venue, year, volume/issue/pages, DOI, and the exact proposition supported. Add primary literature for inventory routing, periodic routing, team orienteering/prize-collecting routing, stochastic waste collection, and reproducible simulation benchmarking. Do not cite an implementation survey where the underlying algorithm paper is available.

## 13. Prioritized amendment ledger

| ID | Severity | Status | Issue | Rerun likely? |
|---|---|---|---|---:|
| RCP-001 | BLOCKER | VERIFIED | `n_vehicles: 0` contradicts single-vehicle claim | Yes, unless scope is relabeled and defended |
| RCP-002 | BLOCKER | VERIFIED | Service-Level equation differs from implementation | Yes if square-root rule is intended |
| RCP-003 | BLOCKER | VERIFIED | Fast-TSP subsection describes another class | No if implementation is the intended treatment |
| RCP-004 | BLOCKER | VERIFIED | 90-day tracked set is not reproducible as the claimed Pareto subset | Possibly; first recover selection lineage |
| RCP-005 | HIGH | VERIFIED | Headline `4×` claim is inconsistent with published ranges | No; regenerate analysis/prose |
| RCP-006 | HIGH | VERIFIED | One realization per factorial cell | Yes for population inference |
| RCP-007 | HIGH | VERIFIED | Improver comparison does not hold constructor tours fixed | Yes |
| RCP-008 | HIGH | VERIFIED | Look-Ahead and empirical-demand descriptions overstate implementation | No if code is intended |
| RCP-009 | HIGH | VERIFIED | Gamma units and heterogeneity assignment are misstated | No if code is intended |
| RCP-010 | HIGH | VERIFIED | Objective/KPI/constants trace is incomplete | Analysis and sensitivity rerun recommended |
| RCP-011 | HIGH | VERIFIED | BPC configuration does not support empirical exactness claims | No; report status/gaps, or rerun exact mode |
| RCP-012 | HIGH | VERIFIED | Experiment inputs and code/environment manifest are incomplete | Targeted artifact reconstruction |
| RCP-013 | MEDIUM | VERIFIED | Figure defects and incorrect aggregate frontier | Regenerate figures |
| RCP-014 | MEDIUM | CORROBORATED | Bibliographic metadata and domain coverage gaps | No |
| RCP-015 | MEDIUM | VERIFIED | Paper is overlong and repetitive for likely proceedings format | No |
| RCP-016 | HIGH | VERIFIED | Manuscript contains no code, data, or artifact availability statement (no repository URL or DOI anywhere in `paper.tex`) | No |
| RCP-017 | MEDIUM | VERIFIED | Malformed `SLSL2` policy label in generated exclusion table (`tab:excluded`) — doubled prefix, generator naming defect visible in publication | Regenerate label |
| RCP-018 | MEDIUM | VERIFIED | Paper states 174 90-day runs but horizon table pairs 165; the nine lost pairs are exactly the configs whose 30-day runs sit in integrity-excluded cells (§6.2), and the text never bridges the two numbers | No; add one sentence |
| RCP-019 | MEDIUM | VERIFIED | Horizon prose cites median 90/30 overflow ratios (2.4–3.3) not shown in any table; mean-derived ratios span ≈2.8–3.6 and exceed the stated band for ACO-HH (3.59) and BPC (3.52) | Add median-ratio column or restate |
| RCP-020 | LOW | VERIFIED | Pareto-membership enumeration omits SWC-TCF and SANS; sentence sums to 15 without stating the remaining constructors hold zero | No |
| RCP-021 | LOW | VERIFIED | Formal-model gaps: Eq. (6) compares absolute fill against a `100%` threshold; fleet size $K$ never fixed to the experimental setting; overflow defined "at" capacity in Sect. 5.2 vs. "beyond" capacity in Sect. 4.4 | No |
| RCP-022 | LOW | VERIFIED | Copy-editing: conclusion typos (`paper.tex:1172`, `:1188`), "unfeasible" (`:461`), brand-name drift (WSmartRoute+/WSmart Route+/WSmart-Route), US-letter PDF geometry, misdated bib keys, ~30 uncited bib entries | No |
| RCP-023 | HIGH | VERIFIED | Mathematical decoupling: Routing objective $\mathcal{P}$ (Eq. 2) lacks an overflow penalty, making the single-period VRPP solver mathematically agnostic to future overflow risk without mandatory constraints | No; clarify theoretical basis in §2.2 & §4.3 |
| RCP-024 | HIGH | VERIFIED | Silent MIP solver truncation: SWC-TCF timeout on $N=350$ emitted empty tour logged as 0-collection day rather than raising `SolverTimeout` | Yes; add solver status telemetry |
| RCP-025 | MEDIUM | VERIFIED | Unexercised IoT sensor noise: Framework supports $\epsilon > 0$ and Fig. 2 prominently features it, but all 480 runs set `sim.noise_std = 0.0` | No; qualify diagram and scope claims |
| RCP-026 | HIGH | VERIFIED | Lack of statistical seed replication ($R=1$): Single stochastic demand realization per cell prevents standard error computation and ANOVA/Wilcoxon hypothesis testing | Yes; replicate factorial design with $R \ge 5$ |

## 14. Recommended revision sequence

### Phase A — freeze and diagnose

1. Tag the exact paper/results snapshot and preserve all current raw artifacts.
2. Create the experiment manifest and reconstruct the 90-day selection rule.
3. Trace fleet/route semantics for each of the eight constructors and restore route separators in telemetry.
4. Decide the intended Service-Level equation and Fast-TSP algorithm.

### Phase B — repair claims and necessary experiments

1. Run a minimal feasibility audit over every stored route: mandatory coverage, per-route payload, route count, and completed days.
2. Rerun the factorial design with replicated demand and optimizer seeds if the paper is expected to support stochastic generalization.
3. Rerun controlled improver comparisons from identical constructor outputs.
4. If single-vehicle operation is central, rerun with an asserted positive fleet limit and a realistic shift/trip model.
5. Run isolated runtime measurements and a nearby-depot ablation.

### Phase C — rewrite and release

1. Narrow the abstract and introduction to evaluated evidence.
2. Correct the formulation, methods, constants, and horizon protocol.
3. Regenerate every table and figure from one audited analysis path.
4. Shorten the manuscript and repair citations/accessibility.
5. Publish a versioned artifact with one end-to-end command and expected hashes.

## 15. Evidence index

Primary sources used in this initial shared draft:

- manuscript: `assets/papers/Simulation-Framework-for-the-MPVRP-with-Profits-in-Smart-Waste-Collection/paper.tex` and `paper.pdf`;
- bibliography: the paper's `mybibliography.bib` and repository `bibliography/` sources;
- summaries: `docs/private/global/simulation/simulation_summary.csv` and `simulation_summary_90d.csv`;
- raw results/configs: `assets/output/30days/**/log_*.json` and `assets/output/30days/**/hydra/{config,pruned_config}.yaml`;
- publication generator: `logic/gen/gen_paper_latex.py`;
- simulator: `logic/src/pipeline/simulations/`;
- policy stages: `logic/src/policies/mandatory_selection/`, `route_construction/`, and `route_improvement/`;
- physical/economic conversion: `logic/src/pipeline/simulations/repository/base.py` and `bins/base.py`;
- individual reviews: `.agent/reports/{chat,claude,gemini,grok,opencode}/`.

## 16. Open questions

1. Which script, data snapshot, objective pair, and dominance granularity selected the 29 policies used at 90 days?
2. At the experiment commit, did every constructor interpret `n_vehicles: 0` identically? How many depot-separated routes did each stored day contain before logging removed separators?
3. Was unlimited routing intended to mean unlimited vehicles or unlimited sequential trips by one vehicle?
4. Which code commit, lockfile, solver versions, and hardware generated the archived results?
5. Are the seed-42 NPZ demand tensors recoverable, and can their hashes be published?
6. Which SWC-TCF runs terminated early, timed out, fell back, or returned empty tours? Current outputs do not retain enough status data.
7. Should the Service-Level uncertainty term be linear or square-root in the horizon?
8. Was the intended improver `FastTSPRouteImprover` or `DPRouteReoptRouteImprover`?
9. Which physical/economic parameter sources support revenue, density, payload, bin volume, and distance cost?
10. What is the target venue and hard page limit?
11. Were the horizon prose's median 90/30 overflow ratios (2.4–3.3) computed from per-configuration data behind the table, and can a median-ratio column be added so the claim is checkable (RCP-019)?
12. Why did Gurobi return an empty route upon timeout for SWC-TCF on $N=350$ without the simulator flagging a fallback or execution error?
13. Can a targeted noise ablation experiment ($\sigma \in \{0.05, 0.15, 0.25\}$) be run to validate the Look-Ahead and Service-Level heuristics under realistic IoT sensor degradation?

## 17. Disagreement log

- An earlier review repeated the manuscript's “nearly four times” claim. Direct arithmetic from the published tables gives `2.77×`, and the slices differ. This draft uses the verified arithmetic.
- Figure quality was rated highly by one text-oriented review. Direct visual inspection found overlapping labels, an incomplete Pareto line, source ambiguity in maps, and inconsistent appendix styling. This draft follows the visual evidence.
- One review proposed constants `r=1`, `c=0.1`, and `Q=100`. The current repository trace instead yields `0.5837 €/kg`, `1 €/km`, and physical payloads of 3,500/2,500 kg converted internally to fill-percentage units. Because the archived commit is missing, this draft labels current values as repository defaults, not proven historical values.
- Earlier bus discussion treated the 90-day design as an intentional Pareto selection. Direct reconstruction from the tracked CSVs shows that the literal selection rule does not reproduce. This draft preserves the broader outcome-conditioned-sample warning while reopening its exact provenance.
- Raw daily JSON makes large collections look like one route because the logger removes internal depot markers. This draft does not claim that the original solver returned one over-capacity route. It claims the narrower verified facts: the experiment was configured without a positive fleet limit, daily mass can require multiple payloads, and route-count provenance is lost.
- Prose quality: one review rated the writing A− overall, while this draft criticizes generated-sounding patterns. Both hold: the flagged patterns (caveat restatement, em-dash chains) are real but concentrated in Sects. 5.7 and 6, whereas the constructor descriptions in Sect. 4 (two-commodity intuition, HGS giant-tour/split decoding) are genuinely strong. The recommendation is targeted trimming of the repetitive sections, not a wholesale rewrite.
- Figure assessment methodology differs across contributors: the opencode review could not render images, so its figure findings are caption- and metadata-derived (admitted outlier axis-extent, PNG table, raster DPI, dimensions). The visual findings (overlapping labels, incomplete Pareto line, map issues) rest on other reviewers' direct inspection. The two sources agree wherever they overlap.
- One review initially repeated the manuscript's "nearly four times" claim; its independent table audit (§7.4) subsequently confirmed the 2.77× recomputation already recorded above. The arithmetic is now doubly verified.

## 18. Changelog

- **2026-08-28 — Codex:** Created the shared report; synthesized five independent manuscript reviews and direct code/config/data audits. Independently verified fleet-setting, capacity-day, 90-day membership, method-fidelity, and headline-arithmetic findings. Added evidence protocol, amendment ledger, roadmap, and open questions.
- **2026-08-28 — opencode:** Added the independent arithmetic audit of every generated table (§7.4) — all counts, marginals, and derived means reproduce from the manuscript alone. Reconciled the 174→165 horizon-pair drop as integrity-excluded cells (§6.2). Flagged previously unrecorded manuscript defects: missing code-availability statement, `SLSL2` label bug, median-ratio verifiability gap, Pareto-enumeration omission, Eq. (6) unit mixing, specific typos, letter-size PDF, misdated bib keys and uncited entries. Extended the ledger (RCP-016–RCP-022), claim map, figure/citation tables, open questions, and disagreement log.
- **2026-08-28 — Gemini (Agy):** Expanded mathematical formulation analysis in §4.3 with the decoupling of the single-period VRPP profit objective from multi-period overflow penalties (explaining why Selection dominates downstream routing). Added SWC-TCF $\mathcal{O}(V^2)$ quadratic complexity and Gurobi timeout truncation analysis in §5. Extended the claim map and amendment ledger with RCP-023 (objective decoupling), RCP-024 (silent MIP timeout truncation), RCP-025 (unexercised sensor noise), and RCP-026 ($R=1$ seed replication gap). Added open questions on solver fallback telemetry and sensor noise benchmarking.

