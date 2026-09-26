# Technical and editorial review of “A Simulation Framework for the Multi-Period Vehicle Routing Problem with Profits in Smart Waste Collection”

**Review date:** 2026-08-28

**Recommendation:** Major revision

**Overall assessment:** Promising framework and unusually candid analysis, but not yet publication-ready because several central method descriptions do not match the implementation that produced the results.

## 1. Bottom line

This paper has the ingredients of a strong applied optimization/framework contribution: an important municipal problem, real road networks and fill records, a clean three-stage policy abstraction, a broad classical benchmark, and a notably serious attempt to prevent biased aggregation after failed runs. The best idea in the paper is not any individual solver. It is the separation of *when to require collection*, *how to construct the profitable routes*, and *how to improve their geometry*. The strongest empirical observation follows directly from that design: among the tested levels, mandatory selection spans a much wider efficiency-service range than route construction. That is useful, operationally intelligible, and easy to miss in a construction-only benchmark.

The paper is also more honest than most benchmark papers about its evidence. It identifies the SWC-TCF failures, removes whole affected cells rather than quietly deleting only the losing method, balances marginals, refuses to rank constructors from the performance-selected 90-day subset, and explicitly says that the improver comparison is not causal. Those are substantial strengths.

However, the manuscript currently has four classes of problem:

1. **Some central descriptions are factually inconsistent with the code.** The benchmarked Fast-TSP improver is not the exact-up-to-20-nodes method described in the paper; the Service-Level equation uses a different horizon scaling from the implementation; the Look-Ahead algorithm is described as a different decision rule from the one in code; and the two demand processes are misstated in ways that affect their scientific interpretation.
2. **The experiment has one demand realization per scenario.** The 480 rows are factorial configurations, not stochastic replications. They support a detailed case study under seed 42, not uncertainty estimates or broad performance claims.
3. **The mathematical problem, optimization objective, and reported evaluation objective are not aligned.** The formal model maximizes profit, while the paper ranks methods mainly by kg/km and overflows, without reporting profit or the economic constants. The formal model also omits several constraints and state variables central to the implemented framework.
4. **The framework claim is broader than the demonstrated artifact.** NCO is prominent in the abstract and keywords but absent from all 480 benchmark rows; sensor noise and multiple vehicles are supported but untested; and no public artifact/version/data-availability statement makes “reproducible” independently checkable.

These are not cosmetic objections. In particular, the Fast-TSP, Service-Level, Gamma-3, and Empirical-process discrepancies must be corrected before another external review. A reader cannot accurately interpret or reproduce the central selection and improver findings from the current Methods section.

My recommendation is therefore **major revision**, not rejection. The architecture, data discipline, and main operational result are worth preserving. With method-code reconciliation, replicated scenarios, a controlled improver experiment, and a tighter manuscript, this could become a persuasive framework paper. Without those changes, the paper overstates what was run and understates what remains uncertain.

## 2. Evidence reviewed

I reviewed the full 34-page PDF and the complete LaTeX source, all generated result figures and tables, the bibliography database, the 30- and 90-day summary CSVs and analysis generator, saved Hydra configurations from the experiment, and the implementation paths needed to verify the selection rules, demand generation, seeding, route improvement, and physical/economic conversions. In particular:

- manuscript: `assets/papers/Simulation-Framework-for-the-MPVRP-with-Profits-in-Smart-Waste-Collection/paper.tex` and `paper.pdf`;
- results: `docs/private/global/simulation/simulation_summary.csv` (480 rows) and `simulation_summary_90d.csv` (174 rows);
- generated analysis: `logic/gen/gen_paper_latex.py` and the generated `Tables/` fragments (read only; none were edited);
- saved run configuration: representative `assets/output/30days/*/*/*/hydra/pruned_config.yaml` files;
- selection implementations: `logic/src/policies/mandatory_selection/selection_service_level.py` and `selection_lookahead.py`;
- demand implementations: `logic/src/data/distributions/statistical_gamma.py`, `statistical_empirical.py`, and `logic/src/pipeline/simulations/wsmart_bin_analysis/export/grid.py`;
- improver implementation: `logic/src/policies/route_improvement/fast_tsp.py` and the called TSP wrapper;
- state/accounting and physical units: `logic/src/pipeline/simulations/bins/base.py` and `logic/src/pipeline/simulations/repository/base.py`.

The existing PDF has no unresolved citations or references. Its log contains only minor underfull-box warnings. The problems below concern scientific content and presentation, not a broken LaTeX build.

## 3. Qualitative scorecard

These scores are a compact summary, not a claim of measurement precision.

| Aspect | Score / 5 | Assessment |
|---|---:|---|
| Importance and practical motivation | 4.5 | High-value municipal problem with clear operational stakes. |
| Framework architecture | 4.0 | The three-stage decomposition is clean, useful, and extensible. |
| Novelty positioning | 3.0 | Plausible contribution, but the framework gap is asserted more than demonstrated. |
| Formal problem definition | 2.0 | Too incomplete to reproduce the implemented decision problem. |
| Method/implementation fidelity | 1.5 | Several material mismatches, including one plainly incorrect improver description. |
| Experimental breadth | 3.5 | Broad factorial coverage across policies, networks, and demand regimes. |
| Experimental validity | 2.0 | One stochastic realization, confounded improver comparison, selected 90-day sample. |
| Data-integrity analysis | 4.5 | Excellent balancing logic and transparency, although failure classification needs telemetry. |
| Reproducibility | 2.0 | Automated tables and saved configs are strong internally; external artifact details are absent. |
| Results and discussion | 4.0 | Nuanced, mechanism-oriented, and appropriately cautious in many places. |
| Writing and organization | 3.0 | Generally readable, but overlong, repetitive, and uneven in the conclusion and algorithm survey. |
| Figures and tables | 3.0 | Consistent visual language and improved maps, but several figures are too dense or visibly cramped. |
| Citations | 2.5 | No missing keys, but there are metadata errors, thin framework positioning, and weak support for some implementation claims. |

## 4. The strongest parts

### 4.1 The three-stage policy model is the paper’s real contribution

Separating mandatory selection, route construction, and route improvement is conceptually strong. It exposes a source of variation that conventional VRP comparisons often hold fixed without discussion. It also creates a sensible software boundary: selection can evolve without rewriting solvers, constructors can be compared on common mandatory sets, and geometry-only improvers can be isolated from profitable-subset decisions.

This part can be made even stronger by showing the actual interface contract, not only the conceptual blocks. A compact table should state each stage’s input, output, invariants, stochastic state, and permitted decision authority. For example, a constructor must visit every mandatory bin, may visit optional bins, and must return a feasible route plus a status. An improver must preserve selected-bin membership unless explicitly tagged as profit-aware. Those contracts would also make failure detection principled rather than outcome-based.

### 4.2 The selection-stage result is important and well interpreted

The observed ordering from LM-CF90 through LA, LM-CF70, SL1, and SL2 produces a clear efficiency-service frontier. The paper correctly avoids calling the fourfold range difference a general variance decomposition. The discussion of remote-depot geometry provides a plausible operational mechanism: deferral amortizes a large fixed access leg over more collected waste.

This is the strongest empirical contribution. It should be elevated into the abstract and tested more directly. A synthetic or real network with an in-area depot would provide the missing geographic ablation. A factorial interaction analysis would also show whether the selection ordering is stable across constructors, demand processes, and networks rather than only in balanced marginal means.

### 4.3 The integrity section is unusually good

The whole-cell exclusion argument is thoughtful. Deleting only SWC-TCF’s failed rows would indeed give that constructor a favorable scenario mix. The additional balancing of selection and improver marginals is even better: it recognizes that removing a constructor-balanced cell can still unbalance another factor. Central generation of tables and figures is exactly the right practice.

The section can be stronger by replacing inferred failure status with explicit telemetry: horizon completed, solver status, timeout, incumbent availability, mandatory bins missed, route feasibility, exception, and fallback invoked. With those fields, exclusions can be declared before looking at outcomes. The paper should also report a sensitivity analysis that treats failures as failures or censored outcomes rather than simply omitting the affected cells from every view.

### 4.4 The Results and Discussion show good scientific restraint

The authors distinguish means from medians, interpret constructor differences as tail behavior where appropriate, describe the improver result as an association, and restrict the 90-day claims to within-configuration comparisons. The discussion connects statistics to plausible mechanisms without pretending that the current design identifies those mechanisms. That restraint deserves credit.

The best way to strengthen this section is to remove repeated caveat prose and use the recovered space for uncertainty, interaction plots, and one controlled ablation. The current manuscript often states the same limitation in the design, result, table note, discussion, and conclusion. One precise limitations table could do that work more efficiently.

### 4.5 The regional maps are now informative

The coordinate-based Rio Maior and Figueira da Foz panels are a large improvement over an empty or schematic map. They establish that the experiment uses real service geometry, show density and topology differences, and make the network-size labels tangible.

Their main weakness is that the remote depots are omitted even though remote-depot geometry is central to the paper’s explanation. Preserve the detailed service-area extent, but add a small inset or annotated connector showing depot direction and distance. Also add a scale bar and make the data-source attribution and legend legible at final print size.

## 5. Publication-blocking factual discrepancies

### 5.1 Fast-TSP is described as the wrong algorithm

This is the clearest factual error in the paper.

The manuscript says that Fast-TSP solves routes of roughly 20 stops exactly by dynamic programming and uses randomized local search for longer routes. That description corresponds much more closely to the separate `DPRouteReoptRouteImprover`, which has a default `dp_max_nodes=20` and uses Held-Karp-style dynamic programming.

The benchmarked `FastTSPRouteImprover` does something else. It splits the plan into routes and calls `fast_tsp.find_tour` for every route with a wall-clock duration. The saved experiment configuration supplies 30 seconds. There is no route-length threshold, no exact dynamic-programming branch, and no exactness guarantee. A `seed` argument is accepted by the wrapper but is not forwarded to `fast_tsp.find_tour`.

This matters because the paper’s explanation of why Fast-TSP should differ from CLS is built partly on a false exact-versus-incremental contrast. It also affects reproducibility and the improver’s stochastic behavior.

**Required amendment:** rewrite the Fast-TSP subsection from the implementation actually invoked by the saved configuration. State the package and version, the 30-second budget, whether the budget is per route or per day, the absence or presence of deterministic seeding, and the returned-solution guarantees. If the intended method really was the DP improver, rerun the experiment with that registered method and regenerate every affected result.

### 5.2 The Service-Level equation does not match the implementation

The manuscript defines the uncertainty term as

`z * sigma_i * sqrt(n_d)`.

Both the scalar mandatory-selection implementation and its vectorized counterpart use

`z * sigma_i * n_d`.

This difference is immaterial for SL1 but material for SL2: the paper describes a multiplier of approximately 1.414, while the experiment uses 2. The saved configuration also gives the missing constant: `confidence_factor: 0.84`. The paper merely says that `z` is fixed.

Because selection is the paper’s main empirical factor, this is not a minor notation problem. The reported SL2 result belongs to a different rule from the one readers are told was tested.

**Required amendment:** decide which rule is intended. If the code is authoritative, change the equation to linear horizon scaling, state `z=0.84`, explain what probability or calibration that value represents, and avoid calling the expression a conventional independent-increment confidence bound. If square-root scaling is scientifically intended, fix the implementation, rerun SL2, and regenerate all affected tables and figures.

### 5.3 Look-Ahead is materially mischaracterized

The paper describes LA as a simulation-based rule that advances each bin to its last safe collection day and makes it mandatory then. The implementation is a deterministic synchronization heuristic using online mean rates:

1. identify bins whose current fill plus mean increment reaches capacity by the next day;
2. simulate those bins after collection to find their earliest next collection date;
3. add other bins that would overflow before that date.

If no bin is initially urgent, it selects none. It does not independently schedule every bin on its last pre-overflow day, and it does not sample forward scenarios. Calling it “simulation-based” invites the wrong mental model.

**Required amendment:** give short pseudocode matching the implementation and explain the synchronization mechanism. If the Jorge et al. rule differs, explicitly label this implementation as an adaptation and state the differences.

### 5.4 Gamma-3’s reported units are wrong

The paper gives Gamma means of 8, 6, 24, and 18 **kg/day**. In the code, these are Gamma means in **percentage points of bin capacity per day**. Generated values are used as percentage fill increments. Collection converts percentage fill to mass using area-specific bin volume and density.

For plastic, a full bin is 47.5 kg in Rio Maior (`2.5 L * 19 kg/L`) and 50 kg in Figueira da Foz (`2.5 L * 20 kg/L`). The corresponding mean mass increments are therefore approximately 3.8, 2.85, 11.4, and 8.55 kg/day in Rio Maior, and 4, 3, 12, and 9 kg/day in Figueira da Foz, not the values printed in the paper.

The heterogeneity assignment also deserves exact disclosure. The implementation tiles an alpha pattern `[1,1,1,1,1,3,3,3,3,3]` and a scale pattern `[8,6]` across bin indices. It is not a fitted spatial grouping in the usual sense. Unless bin ordering has a defensible meaning, this creates an arbitrary index-order dependence.

**Required amendment:** correct the units, state the Gamma parameterization as shape/scale, list the exact assignment pattern or assignment algorithm, and justify why index-based tiling is an appropriate heterogeneous scenario. Ideally randomize or stratify the assignment within each replicated seed.

### 5.5 “Empirical replay” overstates what is preserved

The paper says that the Empirical process “resamples directly from historical daily-fill records” and “replays observed patterns.” The implementation forms a per-bin empirical frequency table and samples each bin’s marginal distribution independently for each simulated day. It does not resample complete historical days or temporal blocks.

Consequently, it does not preserve day-of-week effects, seasonality, temporal autocorrelation, or cross-bin correlation. It is empirical-marginal sampling, not replay and not a multivariate historical bootstrap. This distinction is scientifically important because spatially correlated high-fill days and persistent accumulation regimes can drive vehicle capacity and overflow tails.

**Required amendment:** rename the process to “independent per-bin empirical marginal resampling” and state exactly what dependencies it discards. If the authors want to claim replay of observed patterns, implement a date-wise or moving-block bootstrap and rerun the affected experiments.

### 5.6 The common-random-number explanation is technically inaccurate

The paper says that the simulator “re-seeds its waste generation per policy-and-day combination.” The saved runs instead load a pre-generated `.npz` demand sample keyed by distribution, network, horizon, sample id, and seed 42. That is a good way to share demand, but it is not the mechanism described. Separately, the daily runner creates policy-specific RNG seeds for solver isolation.

**Required amendment:** document the two RNG streams separately:

- a scenario/demand stream or immutable dataset shared across every policy in the block;
- an optimizer stream, including the exact key used to derive seeds for each constructor, strategy, improver, day, and replicate.

Add the actual seed and `n_samples=1` to the experimental table. Also test and document that every compared policy loads byte-identical daily demand arrays.

## 6. Major scientific-design weaknesses

### 6.1 The design is factorial but not replicated

The 480 runs equal 8 constructors × 5 selection variants × 2 improvers × 2 demand processes × 3 network sizes. That is a complete factorial over configured levels, but each cell is observed under one stored demand sample (`n_samples=1`, seed 42). The 480 rows are therefore not 480 independent stochastic observations.

Common random numbers improve the precision of within-scenario contrasts. They do not show whether an ordering survives another demand history. Constructor randomness also remains. Means and medians over configurations summarize the tested grid, but they are not uncertainty estimates over operational days, seeds, networks, or algorithm runs.

The paper is commendably cautious about this in the 90-day discussion, but the central 30-day claims still read more confidently than one realization warrants. The Pareto fronts are particularly visually definitive despite having no uncertainty.

**Best remedy:** replicate the entire 30-day factorial over independently generated demand scenarios and independent optimizer seeds. Five seeds is a bare exploratory minimum; ten or more would give more credible interval estimates, especially for heavy-tailed overflow counts. Treat realization as a blocking factor. Report cluster/bootstrap intervals or paired within-seed contrasts, and do not treat configuration rows as independent replicates. A mixed-effects or blocked factorial analysis can then estimate main effects and interactions without pseudo-replication.

**If reruns are impossible:** explicitly call this a seed-42 case study, remove population-sounding claims, avoid hypothesis tests, and put the result’s conditional nature in the abstract.

### 6.2 The primary metric is not the stated optimization objective

The mathematical model maximizes monetary profit: revenue per kilogram minus travel cost. The experiments call kg/km the “primary operational objective” and rank constructors chiefly on that ratio plus overflow counts. Profit is present in the raw logs but not reported, and the paper never states the values of `r_w`, `c_km`, vehicle capacity, bin volume, or density.

This disconnect is serious. Maximizing a ratio is not equivalent to maximizing profit. A high kg/km policy can provide less service or collect less total waste; a high-profit policy depends on the chosen economic coefficients. Overflow and lost waste are external service criteria, not terms in the displayed objective.

**Required amendment:** choose and state the scientific objective clearly.

- If this is a profit benchmark, report profit and its parameter values, and use kg/km as a secondary efficiency metric.
- If this is a multiobjective service-efficiency study, formalize it as such: profit or distance, overflow events, lost kilograms, and possibly service frequency. Then describe Pareto analysis as the primary evaluation rather than presenting a single-period profit objective that the Results largely ignore.

Also report raw collected kilograms and collection days beside ratio metrics so readers can detect ratio gaming.

### 6.3 The failure rule is principled but still post hoc

The four SWC-TCF outputs are clearly separated in tonnage shortfall, and the 90-day case visibly terminates at day 13. The decision to drop whole cells is preferable to dropping only SWC-TCF. Still, “more than 20% below the scenario median” is an outcome-based diagnostic, not direct evidence of failure. The paper asserts that the runs “failed to complete,” but the stored summaries do not include a uniform completion/status field for every solver invocation.

For a constructor that must serve all mandatory bins, large tonnage differences can indeed indicate invalid or missing routes. That invariant should be checked directly. Otherwise, an unusually poor but valid policy could be labeled corrupt because it performed poorly.

**Required amendment:** add explicit status telemetry and a prespecified validity rule. Report the four runs as failures in a denominator-based reliability table, not only as excluded observations. Include a sensitivity view in which failure is penalized or treated as censored. The appendix should not let invalid points set axes for otherwise valid data.

### 6.4 The improver experiment does not identify an improver effect

The manuscript correctly admits this. The 224 pairs share demand and configuration labels, but upstream constructor outputs differ, including selected-bin counts and tonnage. The “CLS wins 202 of 224” title is therefore too strong even if the surrounding prose is cautious. Those are configuration-level associations, not wins by the improver.

This is also an avoidable design flaw. Serialize each constructor’s pre-improvement route once, clone it, and pass the identical route to CLS and Fast-TSP with independent but recorded improver seeds. Verify that selected-bin membership is unchanged. Then report paired distance and runtime deltas. Because the improvers are intended to preserve membership, overflow and tonnage should be exactly tied; any difference becomes an invariant failure rather than an effect estimate.

### 6.5 The 90-day study is exploratory and currently overallocated space

Only 174 configurations selected on 30-day Pareto performance were rerun; 165 survive matching and integrity checks. The paper handles the resulting selection bias responsibly, but the follow-up cannot estimate an unbiased horizon effect. It adds descriptive evidence for a selected subset and little more.

Explain exactly why 174 raw rows become 165 matched pairs. Consider moving most 90-day material to the appendix until a replicated, prospectively specified horizon sample exists. If retained in the main text, call it “exploratory selected-policy follow-up” in the subsection title.

### 6.6 Runtime comparisons are not reproducible or fully fair

The paper reports precise runtime rankings but omits hardware, operating system, Python and solver versions, thread counts, CPU/GPU use, parallel execution, and exact time budgets. Saved configs show 60-second constructor budgets and 30-second route-improvement budgets in representative runs, while the paper only says “per-run time budget.” BPC is configured with `exact_mode: false`, heuristic RCC separation, finite node and pricing limits, and nonzero gaps.

State whether budgets apply per daily call, per route, or per multi-day run. Separate constructor time from improver and simulator overhead. Run methods on the same hardware and thread allocation, or normalize the comparison. Report timeout frequency and optimality gaps for mathematical solvers.

### 6.7 External validity remains narrow

The benchmark has two municipalities, three sizes, one waste stream, remote depots, one vehicle, zero sensor noise, and no deployment comparison. The paper acknowledges most of this. It should also acknowledge that the N=100 and N=170 cases are nested or related Rio Maior instances rather than independent regions, and that both demand models are generated/resampled rather than a held-out operational replay.

The next experimental expansion should prioritize new realizations and topology/depot ablations before adding many more algorithms. More algorithms on the same single scenario would create breadth without stronger evidence.

## 7. Formal problem definition

The current mathematical section is too short for the role it plays. It gives a fill transition, a route sequence, a no-duplicate condition, and capacity constraints, but it does not formalize the actual simulation/decision problem.

Missing or ambiguous elements include:

- the observed state versus latent true state and the policy information set;
- nonanticipativity in the stochastic multi-period objective;
- mandatory-set variables and the constraint requiring service of mandatory bins;
- optional-visit variables and their link to route arcs;
- route feasibility, vehicle-use variables, depot flow, and the meaning of “up to K routes”;
- overflow and lost-waste state variables, even though they are central evaluation outcomes;
- the relationship between capped fill and separately recorded lost material;
- fallback behavior and what happens when mandatory service is infeasible;
- whether K denotes vehicles, trips, or routes, particularly since the experiment is single-vehicle but may use repeated depot trips;
- the actual economic and physical constants;
- the connection between the expected-profit objective and the kg/km/service Pareto analysis.

The duplicate-visit constraint is also awkwardly quantified and route-sequence notation switches between `vec w` and `bm w`. The transition caps fill at capacity, which makes excess waste disappear unless a lost-waste variable is introduced.

A better formalization would define latent fill, observed fill, lost waste, collection/visit variables, mandatory sets, route feasibility, and a policy adapted to observation history. If full arc-flow detail is too long, give a compact simulator-level Markov decision process plus a clearly referenced single-period VRPP subproblem. The current hybrid is neither a complete mathematical program nor a complete stochastic control model.

## 8. Abstract, title, introduction, and positioning

### Abstract

The abstract is competent but not memorable. It spends most of its space defining the problem and none on a quantitative result. A reader reaches the end without learning that the experiment contains 480 classical runs, that selection spans roughly four times the constructor efficiency range, or that PG-CLNS is the most consistently non-dominated constructor in the tested cells.

More importantly, it says the adaptation methodology includes HGS, ALNS, and NCO, then says an extensive benchmark “evaluates the solvers.” All 480 benchmark rows are classical. NCO is in the keywords but contributes no observation. The body eventually discloses this, but the abstract strongly implies broader empirical coverage than exists.

The source marks the abstract as an immutable conference “abstract of record.” That administrative fact will not protect the paper from a reviewer. If it truly cannot change, add an immediate scope statement in the first paragraph after it and ensure submission metadata or a cover note explains that the full paper evaluates only the classical subset. If the venue permits any revision, replace NCO as an evaluated item with “adapters for future NCO evaluation” and include the main quantitative result.

### Introduction

The opening paragraph is good: compact, concrete, and motivated by temporal state. The introduction also gives the exact factorial size and states the main result early. These are strengths.

The novelty gap is less convincing. The claim that progress is hindered by a lack of standardized multi-period environments is not supported by a comparison of existing simulators, benchmarks, digital twins, or routing libraries. Related Work mostly surveys algorithms. For a paper whose title foregrounds a simulation framework, the missing framework landscape is conspicuous.

Add a comparison table with existing work as rows and capabilities as columns: multi-period state, stochastic generation, sensed versus latent state, mandatory/optional separation, real road networks, classical solvers, learned solvers, common random numbers, multi-vehicle support, public code, and standardized output. This would make the novelty claim testable.

The introduction could also be catchier. A strong version would lead with the operational tension and then state the headline: in this case study, changing when bins become mandatory spans about four times the efficiency range observed from changing route constructors. That is a much better hook than a generic promise of a unified framework.

### Title and scope

The title is accurate at a high level, but “MPVRPP” can be confused with multi-vehicle terminology. Define it immediately and consistently as *multi-period* VRPP. If the experiment cannot be replicated before submission, a more honest subtitle would be “Framework and single-realization case study on Portuguese waste networks.”

## 9. Related work and algorithm descriptions

The taxonomy is readable but too long relative to the missing framework comparison. It reads partly like a compact textbook chapter. The paper devotes substantial space to NCO architectures that are not evaluated and to detailed descriptions of eight constructors, yet gives little systematic evidence about comparable simulation frameworks or MPVRPP benchmarks.

Recommended restructuring:

1. prior smart-waste and multi-period/profit routing formulations;
2. existing simulation/benchmark environments and the uncovered gap;
3. solver families represented in this benchmark, summarized in a compact table;
4. only the implementation details needed to understand adaptations and fairness.

The constructor subsections need a parameter table and clearer novelty boundaries. For every method, state source, implementation provenance, VRPP adaptation, time/iteration budget, stochastic seed, fallback, and guarantee. PG-CLNS is called an original design but is not given pseudocode, an ablation, or a sharp statement of what is novel relative to HVPL, ACO, and ALNS. That is not enough to assess or reproduce an original algorithm.

“The one true hyper-heuristic in the benchmark” is needlessly theatrical. Use “the only hyper-heuristic evaluated.” Similar claims such as “current open-source reference point” should be time-qualified and sourced.

The BPC subsection should distinguish an exact algorithmic family from the actual experimental configuration. The saved run sets `exact_mode: false`, uses finite pricing and node limits, permits heuristic separation, and returns a heuristic fallback. The Results appropriately note the lack of certificates, but placing it under “Exact Methods” without a configuration-level qualifier still risks misunderstanding. Call it a time-limited BPC implementation and report bounds/gaps where available.

## 10. Experimental results and conclusions

The 30-day results are strongest when they remain descriptive:

- the tested selection variants trace a service-efficiency frontier;
- constructor means differ more in tails and runtime than in typical overflow count;
- PG-CLNS is frequently non-dominated under the paper’s chosen aggregate criteria;
- SWC-TCF has a clear reliability/scaling problem in the largest heavy-load case.

Those claims should survive, subject to corrected method descriptions and explicit conditioning on seed 42. Claims that should not survive in their current form include:

- any implication that NCO was benchmarked;
- causal claims that CLS is better than Fast-TSP;
- a general claim that efficiency is horizon-invariant;
- treating the Empirical process as a replay preserving observed temporal patterns;
- treating BPC’s observed solution as exact;
- treating the 20% tonnage threshold alone as proof of solver failure.

The discussion is thoughtful but repetitive. The remote-depot mechanism, three forms of selection/filter bias, improver confounding, and future ablations are each repeated several times. Consolidate limitations into a table with columns for limitation, affected claim, direction of possible bias, and required experiment. This would cut pages while increasing precision.

The conclusion is the weakest prose in the paper. Its opening sentence is long and awkward (“be it during … as well as”), contains “difference temporal horizons” instead of “different temporal horizons,” and later says “without a upstream phase.” It repeats the improver ablation in multiple forms and turns future work into a laundry list. Phrases such as “the obvious next experiment” sound conversational rather than deliberate.

Rewrite the conclusion around three short points:

1. **What was built:** a modular multi-period simulator with separate selection, construction, and improvement stages.
2. **What this experiment showed:** under one common realization on the tested remote-depot networks, selection choice spans the larger service-efficiency trade-off; constructor choice mainly changes runtime and tails.
3. **What remains unresolved:** stochastic generalization, controlled improver effects, noisy sensing, multi-vehicle operation, and selected-sample horizon bias.

Then prioritize future work rather than listing everything: replicated factorial first, controlled improver replay second, explicit failure telemetry third, depot/noise/multi-vehicle ablations next, and learned/matheuristic expansion after the experimental foundation is sound.

## 11. Figures, tables, and visual communication

The generated figures use a consistent palette and generally have informative captions. The maps, factor-space diagram, Pareto view, runtime scaling, and fill trajectory cover complementary aspects rather than duplicating one another. Tables clearly disclose sample sizes and exclusions.

Specific weaknesses:

- **The strategy trade-off figure has overlapping x-axis labels.** “Last-Minute (CF90),” “Look-Ahead,” and adjacent labels visibly collide. Rotate, wrap, abbreviate, or switch to a horizontal dot plot. The dual y-axis also makes a trade-off look more exact than it is; a scatter/frontier plot would be cleaner.
- **The simulation-loop figure is too dense at LNCS width.** Its smallest text is difficult to read. Reduce it to the five-state loop in the main paper and move interface details into a larger appendix figure.
- **The fill-trajectory figure is too small.** Its three panels and legend do not survive final-page scaling. Enlarge it or remove it if it does not carry a central inference.
- **The aggregate Pareto plot lacks uncertainty.** Once replicated, add confidence regions or seed-level points. Until then, say “seed-42 aggregate” in the caption.
- **The improver-delta title is too causal.** Replace “CLS wins 202 of 224” with “Observed configured difference (CLS minus Fast-TSP)” and foreground the uncontrolled input-route caveat.
- **Raw degenerate points distort appendix axes.** Show a filtered analytical panel and a separate failure panel or inset. A caveat in a long caption does not recover the lost visual resolution.
- **The regional maps omit the depots.** Add an inset/connector because depot remoteness is central to the interpretation.
- **Several appendix heatmaps are too small to interrogate.** Provide them as supplementary high-resolution artifacts or interactive data, not only full-page raster panels.
- **The paper is 34 pages in an LNCS-style layout.** Confirm the venue’s page limit. If this is a proceedings paper, the current length is likely untenable. The algorithm survey and raw appendix are the first candidates for compression or supplementary material.

All figures should be checked in grayscale and for color-vision accessibility. Marker shape already provides some redundancy and should be retained.

## 12. Citation audit

The good news is straightforward: all 33 citation keys used by the manuscript exist in the bibliography, the PDF shows no unresolved citation markers, and the LaTeX log has no undefined-reference warnings.

The bibliography still needs a careful metadata and relevance pass:

- `WENTGES2006` renders as 2006, but its DOI and journal volume correspond to 1997. Correct the year and verify all remaining fields.
- `SUN2023111004` stores `doi = https://doi.org/10...`, causing the rendered reference to contain `https://doi.org/https://doi.org/...`. Store the bare DOI.
- The Lin reference renders the same DOI twice through `doi` and `url`; remove redundant URLs where the style already prints a DOI.
- Several keys are opaque or generic (`f386...`, `inbook`, `inproceedings`, `BARNHART1970` for a 1998 article). Keys do not affect readers, but they make maintenance and review unnecessarily error-prone.
- The Gu “Complexity” reference points to Semantic Scholar rather than a publisher record/DOI. Use a primary bibliographic record where available.
- Farkas pricing is supported by a multiobjective column-generation paper rather than an obvious canonical reference. Verify that this source directly supports the implementation claim and add a more specific citation if needed.
- Custom cut families and the original PG-CLNS design need citations or explicit novelty claims plus algorithms. At present, some details are asserted without enough provenance to distinguish literature method from repository-specific invention.
- The paper should cite the source and terms for Google Maps road distances and OpenStreetMap data, not only place tiny OSM attribution in the image.
- Most importantly, add literature on simulation frameworks, benchmark environments, and software artifacts. The current bibliography is algorithm-heavy and does not establish the framework gap on which the paper’s novelty claim depends.

The NCO references are individually relevant, but their collective weight is disproportionate to an experiment with no learned solver. Either add an NCO baseline or compress that survey and frame NCO strictly as interface scope.

## 13. Writing and style

The manuscript is strongest when it is direct: “A useful evaluation must therefore follow a policy over time instead of scoring an isolated route.” It is weakest when it accumulates qualifications into long sentences or uses stylized contrasts repeatedly.

Recommended editorial changes:

- reduce the heavy use of spaced em dashes; many can become commas, parentheses, or separate sentences;
- remove repeated “not X but Y” and “rather than” constructions where a positive statement is clearer;
- replace theatrical phrases (“one true hyper-heuristic,” “the obvious next experiment”);
- define `run`, `configuration`, `scenario`, `scenario cell`, `realization`, and `replicate` once and use them consistently;
- distinguish elapsed horizon days from collection days whenever `days` appears;
- state that reported kilometer values are cumulative over the 30-day horizon;
- standardize terminology for true/latent fill, sensed/observed fill, and zero-noise observation. “True observations” is contradictory;
- shorten the route-constructor descriptions and move parameter detail to a table or supplement;
- remove caveat duplication across prose, captions, notes, discussion, and conclusion;
- fix local errors including “difference temporal horizons” and “a upstream phase.”

The introduction and Results already sound mostly human and technically grounded. The conclusion and parts of Related Work have the strongest “LLM-assisted” signature: comprehensive but indiscriminate enumeration, repeated balanced contrasts, excessive em dashes, and long sentences that say the same thing twice.

## 14. Recommended revision plan

### Priority 0: correct the record before further circulation

1. Reconcile Fast-TSP, Service-Level, Look-Ahead, Gamma-3, and Empirical-process descriptions with the code and saved configs.
2. State all experiment constants: seed, samples, noise, vehicle capacities, revenue/cost, physical bin parameters, solver/improver budgets, gaps, threads, hardware, software versions, and fallback behavior.
3. Decide whether the paper evaluates profit or a multiobjective service-efficiency problem, then align the formal model, metrics, tables, and conclusions.
4. Reframe NCO as untested framework scope or add an actual learned baseline.
5. Add a data/code availability statement with repository URL, immutable commit/tag or archive DOI, license, exact reproduction command, and privacy/licensing status of municipal data.

### Priority 1: repair the experimental evidence

1. Replicate the full 30-day factorial across independent demand and optimizer seeds.
2. Run a controlled improver experiment from serialized identical pre-improvement routes.
3. Add explicit completion, timeout, fallback, feasibility, and mandatory-service telemetry; regenerate exclusions from statuses rather than outcome thresholds.
4. Add a depot-location/topology ablation to test the main proposed mechanism.
5. Report profit, total collected mass, lost mass, collection days, and failure rate in addition to kg/km and overflow events.

### Priority 2: sharpen the contribution

1. Add a framework-comparison table and a compact API/invariant table.
2. Add interaction analysis for selection × constructor × network × demand.
3. Validate at least one small instance against a known optimum or exhaustive reference to show that adapters preserve feasibility and objective semantics.
4. Calibrate or validate the demand simulator on held-out data, including whether marginal, temporal, and spatial properties are preserved.
5. Give PG-CLNS pseudocode, novelty boundaries, and ablations.

### Priority 3: editorial and visual pass

1. Rewrite the abstract if permitted and completely replace the conclusion.
2. Compress Related Work and move granular raw plots to supplementary material.
3. Fix crowded labels and dense diagrams; add depot insets and scales to maps.
4. Correct bibliography metadata and add missing framework/data-source citations.
5. Rebuild and inspect every figure at final print size.

## 15. Final verdict

There is real value here. The paper’s modular decomposition and integrity-aware analysis are stronger than the typical “many solvers, one table” benchmark. The result that selection policy dominates the observed service-efficiency spread is both plausible and operationally useful, and the paper deserves credit for exposing the limitations of its improver and horizon comparisons rather than hiding them.

At the same time, the current Methods section is below the standard required for a technical paper because readers are told that several different algorithms and stochastic processes were run from those actually implemented. The Fast-TSP description is plainly wrong for the benchmarked class; the SL2 equation describes a different policy; Gamma-3 uses the wrong units; and “Empirical replay” claims dependencies the sampler does not preserve. Coupled with one demand realization and no controlled improver input, these issues prevent strong conclusions about comparative solver performance.

The right response is not to discard the work. It is to narrow unsupported claims, correct the factual record, replicate the experimental blocks, and center the paper on what is genuinely distinctive: a modular multi-period testbed showing that collection timing can matter more than the choice among competent route constructors. If those revisions are made, the strongest parts will become much more credible. If they are not, an exacting reviewer would be justified in rejecting the paper despite its good architecture and unusually transparent analysis.
