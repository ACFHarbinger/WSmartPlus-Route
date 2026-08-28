# Review: "A Simulation Framework for the Multi-Period Vehicle Routing Problem with Profits in Smart Waste Collection"

> **Reviewer:** DeepSeek (opencode session), 2026-08-28
> **Critical notice:** An earlier, text-oriented assessment of this paper rated it positively overall. The present review is a **claim-to-artifact audit**: every number below was checked against the archived experiment, the generator, or the dependency source, not against the manuscript's own assertions. Several low-confidence claims from the shared review process turned out to be *wrong in the other direction* (the paper was right); the two most important of those corrections are recorded here with their evidence paths.
> **Verdict:** Major revision. Strong engineering contribution, honest analysis culture, but a manuscript that systematically overstates scope and misdescribes parts of its own experiment. No single defect kills it; the combination does.

---

## 1. Verdict and scorecard

| Dimension | Grade | One-line assessment |
|---|---|---:|
| Problem motivation & positioning | B+ | Genuinely applied, operationally framed; abstract overpromises scope |
| Design of the experiment | B− | Real networks, paired realizations, honest exclusions; one realization per cell, no fleet bound |
| Method descriptions | C+ | Excellent prose, wrong precision: two of eight constructor/improver descriptions need rewriting, one is actually correct |
| Results tables | A− | Every cell I recomputed reproduces exactly; headline *handwritten* claim (`4×`) is false |
| Figures | C | Two blockers (illegible labels; noise callout assert the negation of the protocol), two genuinely good ones |
| Integrity analysis | A | The most reusable methodological contribution in the paper — whole-cell exclusion, balanced marginals |
| Discussion/conclusions | A− | Unusually disciplined causal language; one paragraph too self-deprecating about the paper's own result |
| Writing | B+ | Dense, often excellent, with generated-looking tics and repetitive caveats |
| Reproducibility | D+ | No availability statement, no manifest, no versioned artifact |

The paper is worth salvaging and is probably worth publishing after revision. It is not worth publishing as-is.

---

## 2. How this review was conducted

Independent verification, in order:

1. Every generated table (`Tables/results_*.tex`) recomputed from `docs/private/global/simulation/simulation_summary{,_90d}.csv` through the paper's own filter chain (`split_degenerate` → `drop_affected_cells` → `balance_marginal`), applied exactly as in `logic/gen/gen_paper_latex.py`. All counts, means, medians, win counts, and ranks reproduced.
2. Every *handwritten* numeric claim in the prose recomputed against the same data.
3. All 174 90-day rows cross-checked against the 30-day rows; the batch-selection rule reconstructed from `logic/configs/batch.yaml`.
4. Method-to-implementation tables checked against `logic/src/policies/...`, the simulator, Hydra configs, and **third-party dependency source** (cloned `shmulvad/fast-tsp`, read its C++).
5. Rendered-image inspection of every figure in the manuscript tree.
6. Arithmetic audit of the reward/profit identity against day-level JSON logs.

---

## 3. What is genuinely strong

### 3.1 The three-stage decomposition is the paper's real idea

Separating **mandatory selection** (the only multi-period intelligence in the reported policies) from **route construction** (a myopic single-period VRPP) and **route improvement** (geometry-only) is a genuinely useful abstraction for waste-routing systems, which normally bury dispatch thresholds inside one monolithic policy. The paper's *headline result* — that the selection stage spans a wider observed efficiency–service range than the constructor stage — is not merely plausible, it is structurally **pre-ordained** by this decomposition: the single-period VRPP objective `P(A_d, w_d) = r_w Σw − c_km Σd` contains **no overflow penalty**, so the constructor alone has zero incentive to visit a remote bin whose revenue does not cover its detour. Only the mandatory set forces service. The paper half-states this ("the policy chooses a profitable subset as well as its routes") but never says the sentence that matters: **myopic construction means the selection stage is the only service mechanism.** That sentence is the single most impactful addition available to the authors, because it converts their empirical observation into a theorem-adjacent explanation.

### 3.2 The integrity protocol is a methodological contribution

`drop_affected_cells` (whole-cell exclusion rather than by-method exclusion) plus `balance_marginal` (slices shared by all compared levels) is genuinely better practice than virtually all benchmark papers in this literature, and the paper *explains* why in prose instead of burying it in an appendix. Four SWC-TCF runs failed at the largest, heaviest scenario; the paper drops the whole affected cells for *all* constructors (24 runs at 30 days) rather than silently averaging SWC-TCF over its surviving scenarios. It then *reports the excluded runs in their own table* and explains the detection rule (shortfall vs. scenario-cell median, capped at 0.20 — with the distributional justification that outside the four runs the next-worst shortfall is 0.071). This is exactly how a benchmark should be run, and it is the part of this paper most worth citing. I verified all of it: the four shortfalls are 0.473, 0.473, 0.298, 0.856; the gap in the distribution is real; the 456/57-per-constructor arithmetic holds.

### 3.3 Generated tables are trustworthy, and that is remarkable

I recomputed every generated table independently and every cell matched to the last printed digit: BPC 6.65 kg/km / 8.1 ovf / 4,018 km / 1,140 s; ALNS median 4.0; the strategy marginals (7.38/6.37/5.82/5.23/3.81 at 80 runs each); improver Δ = 0.74 kg/km and 202/224 wins with 181 overflow ties; horizon pairs 28/57/12/40/11/12/5 = 165; scenario marginals n = 216/136; the Pareto-membership counts (PG-CLNS 5, PSOMA 3, HGS 3, BPC 2, ACO-HH 1, ALNS 1); the paired +0.26 kg/km (140/165 positive); the "roughly seven times" per-overflow contrast (341/48.5 = 7.03); even the Gamma-3 mean/variances (8/6/24/18, 64/36/192/108 at αβ/αβ²). The `gen_paper_latex.py` discipline — "no number in the paper is typed by hand" — is a real achievement: every defect in this manuscript lives in *handwritten prose, method descriptions, or claim-to-artifact lineage*, never in a generated cell.

### 3.4 Two figures are genuinely good

`improver_delta_30d.png` (Fig. 7) is the right chart for its claim — sorted per-pair deltas make "fewer but deeper losses" instantly legible. `fill_trajectory_30d.png` (Fig. 8) is the clearest mechanism illustration in the paper: same demand realization, different collection timing, visible overflow at CF90. Both also demonstrate restraint (the improver figure honestly visualizes the confound it describes).

### 3.5 Honest discussion of its own confounds

The paper repeatedly refuses claims the design cannot support: improver deltas are explicitly "descriptive associations rather than causal effects"; the 90-day sample is explicitly outcome-conditioned and cross-constructor ranking is refused; "These summaries describe the tested configurations; they are not estimates over an unspecified population." This is rare, correct, and should be preserved verbatim through revision.

---

## 4. What is wrong

### 4.1 Two papers are compressed into one

The title/abstract/keywords promise a **general stochastic MPVRPP framework hosting NCO, noisy IoT sensing, multi-vehicle dispatch** (Paper B). The experiment delivers **eight classical constructors × five selection variants × two improvers × two demand processes × three networks, one demand realization per cell, zero NCO rows, ε = 0, no positive fleet limit** (Paper A). The body admits this twice ("This architectural scope is broader than the experiment reported below"; "the reported results contain no learned-solver observation") — but the reader meets Pointer Networks, AM, POMO, DIFUSCO, and a page of related-work taxonomy before reaching either admission. The abstract of record (kept verbatim for the conference) even names NCO among its benchmarks. **Publish Paper A. State Paper B as a scoped extensibility paragraph plus a versioned artifact.** Otherwise the abstract is an overclaim on its face.

### 4.2 The headline number is wrong

"The range in mean efficiency across the five tested selection variants is **nearly four times** the corresponding range across constructors." Published marginals: selection range = 7.381 − 3.809 = **3.572**; constructor range = 6.649 − 5.359 = **1.290**; ratio = **2.77×**, not 4×. Even the ratio-of-medians (2.97×) and the unbalanced construction (2.88×) are nowhere near four. This is the paper's single most-quoted quantity and it is simply false — and worse, the two ranges sit on *different balanced slices* (n = 400 vs. n = 456), so even 2.77× is not the clean estimate. Fix: compute both ranges on one common eligible slice, and have the generator emit the ratio as a derived constant so prose cannot drift.

### 4.3 The benchmark is not single-vehicle (fleet semantics)

Every archived 30-day config sets `sim.n_vehicles: 0`. In the routing code and the TCF formulation, zero means *automatic/unlimited* — the SWC-TCF solver literally does `number_vehicles = len(binsids)` when passed 0. The paper states "restricts every scenario to one vehicle and one depot" — contradicted. The daily logs make the consequence concrete: Figueira da Foz days routinely collect more than the 2,500 kg physical payload (I verified up to 7,094.54 kg in one day, and that a single day's mass at capacity 63.3% of the N=350 fleet must decompose into at least three payloads). On either reading — one vehicle making unlimited sequential trips, or automatic fleet sizing — the paper's "single-vehicle" language is wrong, and the route-count provenance is additionally destroyed by the logger, which strips all internal depot separators (`daily.tour` in `day_context.py`). The revision must distinguish: configured fleet limit / payload per route / number of same-day trips / simultaneous vehicles vs. sequential trips / shift duration. Add route-count telemetry and rerun, or relabel the operational setting.

### 4.4 Service-Level equation ≠ implementation (and the consequence is a ~41% shift)

The paper's rule is `ŵ + n_d μ̂ + z σ̂ √n_d ≥ 100%`. `selection_service_level.py:63` computes `current + μ̂·n_d + z·σ̂·n_d` — **linear** in the horizon, no square root. With `z = 0.84` (never disclosed in the paper; it is in `ms_service_level.yaml`), SL2 as-run uses a 1.68σ̂ margin against the printed 1.19σ̂. Because SL2 anchors the low-overflow end of the headline frontier, this is not cosmetic: the *measured* SL2 point is ~41% more conservative than the SL2 the paper defines. Either the written rule is intended (regenerate SL2 rows) or the code is the ground truth (fix the equation, disclose z = 0.84). Also, the printed RHS "≥ 100%" compares an absolute mass against a percentage — the state variable is in kg and capacity is in mass, so the rule must normalize by C_i as written.

### 4.5 Look-Ahead is a degenerate Service-Level, and the paper never says so

`selection_lookahead.py::_should_bin_be_collected` is exactly `w + μ̂ ≥ C` — Service-Level with z = 0, n_d = 1. Four of the five "conceptually spanning" selection variants are therefore **one parameter family** (LM-CF70/CF90 are raw thresholds, LA and SL1/SL2 share the same predictor with (z, n_d) ∈ {(0,1), (0.84,1), (0.84,2)}). This *explains* the paper's main result instead of merely reporting it: the monotone frontier is the sweep of a single conservatism dial. The paper should say so — the explanation is stronger than the observation — and the "one reactive, one statistical, one simulation-based" taxonomy should be retired. (The look-ahead does add collection-day synchronization and bundle expansion on top of the shared trigger, so LA ≠ SL as whole rules; the *predicate* is the same object.) Related: the paper calls Look-Ahead "the most computationally expensive of the three rules per decision because it simulates forward" — the implementation is an O(V·D) arithmetic loop, no simulation, no sampling. Drop the cost claim.

### 4.6 The 90-day sample is outcome-conditioned in a way the paper's own rule does not reproduce

The paper: "Only policy configurations that lay on the 30-day Pareto front were carried forward, producing 174 runs." The tracked repo's `batch.yaml` is explicit about its per-run-type lists (la_cls: bpc, pg_clns; la_ftsp: aco_hh, bpc, swc_tcf; lm_cls: bpc, hgs, pg_clns; lm_ftsp: bpc; sl_cls: aco_hh, bpc, pg_clns, sans; sl_ftsp: aco_hh, bpc, pg_clns, psoma) — and this manifest *does* produce exactly the 174 stored rows (29 distinct policies × 6 scenarios). But the manifest's lists are not an output of any Pareto computation I can reproduce:

- per-scenario, row-level non-dominance yields 33 Pareto rows, of which only 23 appear in the 90-day set;
- per-run-type, scenario-averaged non-dominance yields different lists still (e.g., LA/CLS → {ACO_HH, BPC, PG-CLNS, SANS}, not {bpc, pg_clns}).

So the selection policy is recoverable (it is in batch.yaml) but its *derivation* is not a literal 30-day-Pareto-front rule. The paper reads the manifest's intent as fact. Correct to "a performance-selected subset carried forward per run type, with the derivation recorded in the versioned batch manifest" and keep the honest pairing-only horizon analysis. Also bridge the 174↔165 discrepancy: the nine lost pairs are exactly the configurations whose 30-day runs sit in the integrity-excluded cells — one sentence fixes it.

### 4.7 Four method descriptions misstate the archive (one of them is actually right)

The shared review round produced five claims in this area; my audit confirms four and **resolves one in the paper's favour, against prior reviews**:

| Method | Paper says | Archive shows | Ruling |
|---|---|---|---|
| Service-Level | `z σ √n` | `z σ n`, z = 0.84 | mismatch (see §4.4) |
| Fast-TSP | "up to roughly twenty stops solved **optimally by DP**; longer ones fall back to a **randomized local search** within a small fixed budget" | `shmulvad/fast-tsp` v0.1.5: `#define EXACT_SOLUTION_THRESHOLD 20`; `find_tour` → `solve_tsp_exact` (bottom-up Held–Karp, O(2ⁿn²)) for n ≤ 20, else `local_search` (greedy NN + double-bridge shuffle + 2-opt/3-opt, bounded by duration_seconds) | ✅ **the paper's text is exactly right** |
| Look-Ahead | "simulates forward", "most computationally expensive" | deterministic linear projection + bundle pass | overstatement |
| Gamma-3 | "means 8, 6, 24 and 18 **kg/day**" | generator returns **percentage-point fill** (`rng.gamma(k,θ)/100`), tiled by bin index; mean increment in kg is ≈ 4–5 kg/day for a 47.5–50 kg bin | units wrong by ~×4 |
| Empirical | "replays observed patterns" | independent per-bin empirical marginal resampling | overstatement (temporal & cross-bin dependence lost) |

On Fast-TSP, prior reviews in this round concluded the DP "lives in a different class" and that the paper misdescribed its improver. That finding was wrong: the exact solver lives *inside the dependency* — the Python wrapper calls `fast_tsp.find_tour` and the C++ core contains both the threshold and the DP. The paper's description is `correct`. One genuine residual detail: the library validates matrices against a 65,535 uint16 ceiling at the Python boundary (`is_valid_dist_matrix`), while WSmart scales road distances ×10,000 (SCALE) — on this Linux/x86-64 build `uint_fast16_t` is 8 bytes so values like 642,572 pass through un-wrapped (I verified no wrap on a decisive 65,546-vs-100 edge case), but the contract is platform-dependent and should be noted in the artifact. Also the improver is invoked with a per-route 30 s budget (`ri_ftsp.yaml`), not "a small fixed budget" in the common sense of "cheap" — 30 s per route at 10 routes is the single largest improver cost in the table; the paper conflates "fast" (exact for ≤20) with "cheap".

### 4.8 The improver comparison is neither controlled nor correctly attributed

224 "matched" pairs share constructor/strategy/region/demand but **not** the upstream route. Constructors are stochastic; CLS and Fast-TSP routinely received different tours (I verified 90/224 pairs differ in collected kg, 128/224 in bin count). "CLS wins 202 of 224" therefore quantifies paired configurations, not an improver effect, and the paper says so — then spends a page plus a figure anyway. The honest and stronger move: compress the subsection to the design-flaw statement, keep Fig. 7, and run the pair on identical stored tours next round. The paper further says CLS "optimizes distance and not profit" — I verified the move-scoring code (`local_search.py`) indeed scores by distance with capacity feasibility only. That claim is one of the few places the paper is *more* accurate than its data (good).

### 4.9 Runtime is not an equal-budget comparison, and "ACO-HH is fastest" has a 1-second caveat

All eight constructors are configured with `time_limit: 60` per day in the archived runs, but observed daily times blow through that ceiling unevenly: HGS 113–192 s/day, PG-CLNS up to 170 s/day, SANS up to 308 s/day, BPC 70–95 s/day. The configured limits are soft inner-phase budgets, not wall-clock caps, and "time" in the tables is the sum of daily call times. The paper's runtime ordering (which it leans on hard: "Runtime separates the constructors more cleanly than quality does") is therefore an ordering of *how the code consumes its phase budgets*, not an equal-budget benchmark. At N=350 the minimum is ACO-HH 1,218.65 s vs. ALNS 1,219.65 s — a one-second dead heat, then the paper credits ACO-HH with the fast end of the "4.5-fold spread". The spread itself (5,451/1,219 = 4.47) is right; the credit is a coin flip. And "within 5% of BPC" for ACO-HH is actually 5.26% (marginally false); "roughly a quarter of the HGS and PG-CLNS means" is right for HGS (25.5%) but 30% of PG-CLNS (closer to a third).

### 4.10 The published objective is never reported; the reported KPI is not the objective

The paper defines profit P (maximized by every constructor) and then reports *kg/km* as "the primary operational objective" — but kg/km ≠ profit ranking. On the filtered data used for the published tables: kg/km ranks are BPC=1, PG-CLNS=2, ACO-HH=3, HGS=4, ALNS=5, SWC-TCF=6, PSOMA=7, SANS=8; profit ranks are BPC=1, PG-CLNS=2, ACO-HH=3, **ALNS=4, HGS=5, SANS=6, SWC-TCF=7, PSOMA=8**. The bottom half *reorders* between metrics. Profit and kg-lost are already in every log and never tabulated. Tonnage is the quantity used to throw runs out; not showing it as a result is perverse. Worse for reproducibility: the paper never states the economic constants. The archive pins them — plastic revenue `0.65 × 898/1000 = 0.5837 €/kg`, distance expense `1 €/km`, densities 19/20 kg/L, bin volume 2.5 L, payloads 3,500/2,500 kg — I verified the identity `profit = 0.5837·kg − 1.0·km` in day-level JSON. None of it appears in text. A VRPP whose objective coefficients are unpublished is not a benchmark.

### 4.11 No availability statement anywhere

`paper.tex` contains no repository URL, no code/data section, no DOI, no license. For a paper whose first contribution bullet is "A reproducible simulator" — and whose demand-vector NPZ inputs are **not tracked** (the datasets referenced by every config live in a gitignored `data/wsr_simulator/` tree, absent from checkout) — this is near-disqualifying. The 90-day raw logs are also untracked (only the parsed summary CSV survives). Add a versioned release with the base+submodule commits, checksums for demand/NPZ/road matrices, fully resolved Hydra configs, and one canonical reproduction command with expected row counts.

### 4.12 Two figures are blockers on their own

I inspected every figure as a rendered image.

1. **Fig. 6** (`strategy_tradeoff_30d.png`) — the caption-bar-labels figure for the headline result: x-tick labels are not merely "overlapping", they are **destroyed**: the axis renders `Last-Minute (CF90)Look-AheadLast-Minute (CF70)ervice-Level (SL1)ervice-Level (SL2)` — the leading "S" of both Service-Level labels is painted over. Additionally the dual-axis bars are independently scaled, so CF90's two bars render identical height, visually asserting 7.38 kg/km ≡ 14.9 overflows. Redraw as a five-point connected scatter in the (overflows, kg/km) plane (Fig. 4's idiom), with short codes (CF90/LA/CF70/SL1/SL2).
2. **Fig. 2** (`simulation_loop.png`) — the **bold highlighted callout** in Box 2 reads *"Observation Asymmetry: Policies receive noisy f̃ᵗ (Ground truth fᵗ is hidden)"* and Stage 1 reads "Input: Sensed f̃ᵗ". This asserts, in the loudest element of the figure, the negation of §4.2 ("for this study we ran the simulations with the true observations") and of the archive (`noise_variance: 0.0`). The caption's parenthetical "(where, for this study, ε=0)" cannot undo a bold callout. The dashed next-day-transition arrow also crosses the policy boxes and strikes through Stage 1/2 text. Grey out or remove the noise branch; reroute the arrow.

Remaining figure issues: Fig. 4's non-dominated front — see §4.13, the drawn front *does* include ALNS and is correct; Fig. 5 (runtime) has a single tick (10³) on a log axis and ALNS/HGS are indistinguishable reds; Fig. 3 (maps) has no scale bar, the depot is deliberately cropped (the one figure that would substantiate the remote-depot mechanism), the caption says "Google Maps (for Rio Maior)" while both panels are drawn from OSM, and it shows only N=170 despite the study using N=100 too; appendix panels use inconsistent scaling and don't encode constructor identity; the appendix heatmaps are a different visual style from main-paper figures.

### 4.13 One correction to the shared review: the Pareto figure is actually correct

The shared report round claimed the dashed front "runs PG-CLNS → BPC and stops" and that ALNS was omitted. I regenerated the figure using the generator's own `pareto_front` — membership is exactly {BPC, PG-CLNS, ALNS} — and crop-inspected the published PNG: the dashed path is BPC (8.1, 6.65) → steps-post → PG-CLNS (6.0, 6.31) → ALNS (5.9, 6.13), WITH the vertical link from PG-CLNS to ALNS present. ALNS (5.9 ovf., 6.13 kg/km) is non-dominated and **is drawn on the front**. The figure was right; the review was wrong. (The generator shares one implementation with the tables; the in-text Pareto-membership counts also reproduce. The only defensible nit: the in-text enumeration omits SWC-TCF and SANS at 0 memberships, so the sum 15 is not checkable against 6 scenarios × 8 constructors; trivial prose fix.)

---

## 5. Ambiguitities, incompleteness, and details needing amendment

1. **Fleet vs. trips vs. shift** — one operational paragraph should pin all five resources (§4.3). Today the terms "routes", "trips", and "vehicles" are used interchangeably.
2. **Q per city** — Figueira payload 2,500 kg, Rio Maior 3,500 kg, converted to bin-percent units internally. The demand-process × capacity interaction is unexamined: BPC binds at exactly 2,500.0 kg on empirical days and near 5,000 on Gamma-3 days of the *same scenario*, meaning the Gamma-3 runs operate on a different apparent capacity. Explain units or fix; never pool constructor rankings over processes until this is stated.
3. **The improver's budget** — 30 s per route via `ri_ftsp.yaml`; the paper's "small fixed budget" is underspecified. Fast-TSP's 20-stop exact threshold should be stated as the dependency contract, with the uint16 scaling caveat.
4. **Constructor hyperparameters** — none published. All archives show 60 s/day nominal caps, 10-ants/50-iterations ACO-HH, BPC `exact_mode: false` with 2,000-node B&B cap and 0.5–1% gap targets, PG-CLNS 10-pt population, HGS μ=25/λ=40. Without a table, "ACO-HH is fastest" reads as a claim about the algorithm rather than about its budget.
5. **The 30-day count in the intro** — "480 runs under common demand realizations" is true, but every reader should also be told `n_samples = 1`: the 480 are factorial cells, not replications. The paper says "one stored demand realization per configuration" in Limitations; say it earlier and louder.
6. **SANS vs. Jorge et al. (2022)** — the cited paper is specifically about *workload concerns and shift duration*. The archived run does not bind shift duration (a 390-minute default constant is loaded and never used), and future work then proposes shift constraints "e.g., the Temporal Team Orienteering Problem" as if that were new. Either bind T_max and rerun, or write "SANS without the Jorge 2022 shift constraint."
7. **PG-CLNS "original design"** — one paragraph of metaphors ("pheromone on edges", "per-individual refinement", "reuses an edge another ant has just taken"), no pseudocode, no parameters, no ablation vs. plain ALNS. HVPL (Sun et al. 2023) is a totally different problem class (LRPSPDTW). Either specify it to reimplementation standard or demote to "registered hybrid LNS".
8. **Farkas citation** — `Lin2017` (Lin, Ehrgott, Raith, 4OR 2017) is column generation for multi-objective LP non-dominated sets; it does not establish Farkas pricing. Use Lübbecke–Desrosiers or Barnhart et al. 1998 (already cited under the misleading key `BARNHART1970`).
9. **Failure telemetry** — SWC-TCF timeouts at N=350 return an empty/depot-only tour (`mdl.SolCount == 0` path), which the simulator logs as a zero-collection day and which only shows up post-hoc as a tonnage shortfall. Add `SolverStatus` (OPTIMAL / TIME_LIMIT / FALLBACK_USED / INFEASIBLE) per day. This also gives a *process-based* exclusion rule rather than the current outcome-based shortfall rule (which is fine — but a validity-based rule is stronger and less open to "you excluded my worst runs" objections).
10. **Venue page limit** — the compiled PDF is 34 pages: 28 body + 4 refs + 3 landscape appendix. LNCS proceedings limits are typically 12–16 including references. This reorders all revision priorities: if the limit binds, cut the related-work taxonomy (~2 pp), trim the eight constructor blocks to a table (~4 pp), drop Fig. 1 or Figs. 9–11 from the main text, and move the appendix to a supplementary artifact.
11. **Reward ≠ profit** — the simulator also logs `reward = kg − overflows − km` (verified identity), which is neither the constructor objective (revenue−cost) nor the reported profit (economic). The paper only reports profit-derived KPIs, but if the framework's public demo/UI is ever described, this third metric must not be confused with either.

---

## 6. Feedback: improving the weakest parts

**(a) Fix the claim-to-artifact lineage mechanically.** The strongest internal tool (`gen_paper_latex.py`) already generates tables. Extend it to generate *derived prose constants* ("the ratio is 2.77×") from the same analysis path and add a CI guard that fails the publication build when handwritten numbers drift. Two of this paper's worst defects (`4×`, the 174↔165 gap, the median-ratio band) are exactly the kind of bug this kills.

**(b) Split the paper.** Publish Paper A (the archived experiment) with a one-paragraph scoped-extensibility statement plus a versioned artifact for Paper B. This is a *content* fix, not a cut: it converts the NCO-related-work from an overclaim to a roadmap.

**(c) Run the three experiments that would make the two main claims bulletproof:**
- a **null-selection cell** (empty mandatory set) and a **must-collect-all cell** — without them, "selection dominates construction" is a comparison among five flavours of forcing, not of forcing vs. freedom. Even on N=100 alone, these two cells are the cheapest possible strengthening of the headline result;
- a **replicated factorial** with R ≥ 5 demand realizations and fixed optimizer seeds (the paper already names this as future work; it is not optional if the paper wants population claims);
- an **equal-budget runtime rerun** (one worker per solver, fixed internal threads, identical hardening) so the runtime plane survives scrutiny, plus the **identical-tour improver pair**.

**(d) Correct operational framing.** State single-vehicle vs. automatic-fleet honestly (or rerun with a positive limit plus trip/shift constraints), publish Q and the Euro parameters, and add route-count telemetry.

**(e) Fix the figures.** The two blockers are one-hour fixes each. Then unify appendix/style, add scale bars + depot inset to Fig. 3, encode constructor identity in Fig. 9, and export vector PDFs.

---

## 7. Feedback: making the strongest parts stronger

**(a) Promote the integrity protocol to a named methodology.** Give it a title ("balanced cell exclusion", "symmetry-preserving invalid-run filtering"), a two-line algorithm, and cite it as the paper's second contribution. Reviewers in OR/benchmarking will value this more than any individual solver result, and it makes the paper's field contribution (simulation benchmarks for dynamic waste routing) rather than its method contribution (yet another ALNS variant).

**(b) Make the "selection is the only service mechanism" argument explicit.** Add the one-sentence observation that Eq. (2) has no overflow penalty, and note that the mandatory constraints are what move service. Then note the corollary: the observed efficiency–service frontier is traced by a single conservatism dial swept over a shared projection predicate (LA→SL1→SL2) plus two raw thresholds (CF70/CF90) — which explains the perfect monotonicity *and* honestly limits what it shows. This turns the paper's own critique into the analysis.

**(c) Add the economic view.** Report profit and kg-lost columns in the result tables (both already in the logs), and one sentence of sensitivity on r_w/c_km (e.g., varying the ratio around 0.584 €/km-equivalent; the ordering of the bottom three constructors visibly changes between kg/km and profit, which is itself a finding about KPI choice in profitable-routing benchmarks).

**(d) Keep three good figures, fix one good paragraph.** The discussion must keep its epistemic restraint verbatim, but the paragraph "It would be interesting to perform ablation studies…" reads as the paper's own eulogy. Replace it: state that the design cannot separate selection mechanisms from each other (they're one family), and that the next experiment is the null/must-collect anchors. Future-work should be three priorities (replicated factorial, feasibility telemetry, controlled ablations) plus the artifact statement — not a list of ten items.

**(e) Cite the correct literatures.** The related-work needs more on inventory routing, periodic VRP, prize-collecting/team orienteering, stochastic dynamic waste collection, and *other simulation benchmarks* as the actual problem boundary; the neural catalogue should shrink to what feeds the NCO adapters the framework actually registers.

---

## 8. Holistic evaluation

**Writing.** The prose is dense and often genuinely good: the Related Work taxonomy is organized with care; constructor paragraphs (two-commodity intuition, HGS giant-tour/split decoding, ALNS repair) read well; the Discussion is the most disciplined I've seen in this genre. Faults: contrast templates ("not X but Y"), repeated caveat restatements, em-dash chains, inflated adjectives, brand-name drift (WSmartRoute+ / WSmart Route+ / WSmart-Route), three confirmed typos ("difference temporal horizons" → different; "a upstream phase" → an; "unfeasible" → infeasible). The abstract of record is honest about the framework and silent about the experiment; since it cannot be edited for the conference, it needs an unmistakable companion scope note.

**Abstract & introduction.** Abstract: solid motivation, but names NCO among "adapted CVRP algorithms including HGS, ALNS, NCO" and "an extensive benchmark" without saying no NCO row exists. Catchiness is fine; accuracy is not. Introduction: the motivation (coupling transport cost to service reliability; bins accumulate at different rates; sensors imperfect; collection changes tomorrow's state) is concrete and right — but the paper's *research question* never appears as a question. It is implied ("how much of long-horizon performance is selection timing vs. route geometry?"). State it explicitly in the first paragraph; everything else follows.

**Figures.** The weakest component: two submission blockers, two good ones, three uneven. The good news is they are all locally fixable and one "defect" (Fig. 4's front) is a false positive that should be removed from the shared review record.

**Experimental design.** Strengths: real networks with road distances; paired demand; explicit service + efficiency outcomes; honest whole-cell exclusions; an improver comparison that names its own confound. Weaknesses: one realization per cell (no uncertainty, no significance, so all "wider/stronger" comparisons are descriptive); no fleet bound; no shift bound; no noise; no null/must-collect anchors; Pareto-selected 90-day sample under an unreproducible literal rule; runtime plane not equal-budget. The paper knows all of this and says most of it — which is why the grade is B− rather than C. The revision should keep the *analysis honesty* and spend its energy on the *experimental gaps*.

**Conclusions.** Correctly qualified; repetitive; the limitations paragraph is good and the future-work paragraph is a decade in one sentence. The conclusion should lead with the artifact and the two claims the paper actually owns (selection spans a wider observed trade-off, and integrity-aware benchmarking is portable).

**Credit where due.** The three-stage framework idea, the integrity methodology, the generated-table discipline, the remote-depot/single-bin mechanism figures, and the epistemically honest Discussion are all genuinely above standard for this literature. This is a paper worth the authors' time — provided the revision fixes what the experiment actually was.

---

## 9. Bottom line

**Major revision.** Fix in order: (1) fleet semantics + constants publication + manifest/availability; (2) Service-Level formula and Fast-TSP/library wording; (3) the 2.77×≠4× headline and the 90-day rule; (4) two figure blockers; (5) split Paper A/B and cut to the venue limit; (6) add null/must-collect anchors and replication next round. The paper's own honesty about its limits is its best feature and its biggest risk — a reader who catches the `4×` error after reading a paper that is *so* careful about integrity is left wondering what else is soft. Closing that gap between the manuscript's analysis and its own standards is the entire revision.
