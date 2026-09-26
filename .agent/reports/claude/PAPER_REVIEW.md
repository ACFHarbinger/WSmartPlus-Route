# Independent Review — *A Simulation Framework for the MPVRP with Profits in Smart Waste Collection*

**Reviewer:** Claude (lead)
**Date:** 2026-08-28
**Artifact reviewed:** `assets/papers/Simulation-Framework-for-the-MPVRP-with-Profits-in-Smart-Waste-Collection/` @ `36d7c7df6`
(paper.tex 1301 lines, 34 pp. compiled: 28 body + 4 refs + 3 appendix; 6 generated tables; 8 figures)
**Cross-checked against:** `docs/private/global/simulation/simulation_summary{,_90d}.csv`, `logic/gen/gen_paper_latex.py`,
`logic/src/policies/mandatory_selection/`, `logic/configs/policies/other/ms_service_level.yaml`, `mybibliography.bib`

This review is deliberately independent of `.agent/reports/gemini/PAPER_EVALUATION_REPORT.md`. I read that report
only after finishing my own verification. Where we disagree I say so explicitly — there are two such places, and
in one of them the prior report repeated an error rather than catching it.

---

## 0. Verdict

**This is a good paper with an unusually honest analytical core and three defects that must be fixed before
submission — one of which is a wrong number in the paper's own headline claim, and one of which is a stated
equation that does not match the code that produced the results.**

| Dimension | Grade | One-line |
|---|---|---|
| Problem framing & motivation | **A−** | Genuinely well-motivated; the temporal-coupling argument is real and clearly made |
| Related work | **B** | Excellent taxonomy, thin coverage; two whole adjacent literatures are missing |
| Formalisation | **B−** | Clean state dynamics; ambiguous on *K*, silent on every numeric constant |
| Method description | **C+** | Well-written, but one strategy's stated equation is wrong and another's cost claim is unsupported |
| Experimental design | **C** | The 30-day factorial is solid; the improver and 90-day arms are structurally unidentified — the paper knows this |
| Data-integrity handling | **A** | The best thing in the paper. Genuinely exemplary. |
| Results reporting & honesty | **A−** | Disciplined refusal to over-claim; docked for the 2.77× erratum |
| Figures | **C−** | Two figures have defects a reviewer will see immediately; one contradicts the text |
| Writing | **B+** | Strong, with a visible register break in recently-inserted passages |
| Abstract | **C** | Promises a benchmark of NCO that the body states does not exist |

**Recommendation:** *Major revision.* The science is sound and the honesty is above the norm for this literature.
The blockers are correctable in days, not months — except the page budget, which is a structural decision.

---

## 1. What is genuinely strong (credit where it is due)

**1.1 The data-integrity section is better than most published practice.** §5.5 and the surrounding
machinery are the paper's real contribution to methodology, and I want to be unambiguous about this because
the rest of this review is critical. Four runs failed. The naive response — drop the four bad rows — would
have introduced exactly the bias the paper is warning about, because all four are SWC-TCF's. The authors
instead dropped the entire scenario cell for **all eight** constructors, at a cost of 21 additional healthy
runs, to keep every constructor averaged over an identical scenario set. Then they noticed the *second-order*
version of the same problem — that dropping a cell removes all constructors together but only one selection
variant and one improver, because those stages *identify* the cell — and restricted every other marginal
accordingly. I verified both filters reproduce the published tables exactly (§2 below). I have reviewed a lot
of benchmark papers; almost none of them get the second-order case, and many do not get the first.

The detection rule is also defensible rather than arbitrary: the shortfall distribution has median 0.00 and
tops out at 0.071 outside the flagged runs, which sit at 0.30/0.47/0.47/0.86. That is a genuine gap, not a
threshold chosen to produce a desired answer, and the paper shows it. The explicit note that collection-day
count *cannot* diagnose truncation (because good policies bundle collections) is the kind of detail that only
appears when someone actually tried it.

**1.2 The paired-demand design.** Re-seeding waste generation per policy-and-day, rather than letting each run
consume randomness in its own order, is the right call and is what makes "collected tonnage is comparable by
construction" true rather than aspirational. It is also what makes the integrity check possible at all. This
is well-executed and well-explained.

**1.3 The generation pipeline.** No number in the tables is typed by hand; `gen_paper_latex.py` derives every
cell from the summary CSVs and applies the exclusion rules centrally so the paper, the markdown reports and
the website cannot disagree. I was able to reproduce Tables 1 and 3 to the printed digit from the raw CSVs in
under ten minutes *because* of this. That is exactly the property a benchmark paper should have. It is also,
pointedly, what let me find the one number that *was* typed by hand and is wrong (§2.1).

**1.4 The Related Work taxonomy.** §3's organising principle — that methods separate by "what they guarantee
against what they cost to run", with heuristics differing "chiefly in how they escape local optima:
trajectory methods perturb a single incumbent, population methods recombine several" — is a better framing
than most surveys manage, and it earns its place by making the eight constructors legible as a designed
spread rather than a grab-bag. The decision to describe Jorge et al. and de Morais et al. as methods that
"enter it directly rather than as external points of comparison" is the correct and honest framing of
reimplementing prior work.

**1.5 The Discussion's refusal to over-claim.** Passages like "the design cannot identify that mechanism",
"does not establish a common cause", and "these are descriptive contrasts between configured variants, not a
controlled comparison of their mechanisms" are doing real work. §5.6's "one methodological hazard appears in
three forms" is the intellectual high point of the paper: it correctly identifies that the 90-day Pareto
conditioning, the degenerate-run exclusion, and the marginal-balancing problem are *the same error* wearing
three costumes, and generalises it ("such benchmarks invite this error because runs fail selectively where
the instance is hardest and follow-up experiments are naturally conditioned on what looked promising first").
That paragraph is more valuable than several of the results it qualifies.

---

## 2. Confirmed errata — verified against the source data

I recomputed the marginals from `simulation_summary.csv` using the paper's own filters
(`split_degenerate` → `drop_affected_cells` → slice-balancing). My reproduction of Table 3 matches the
published values to the last printed digit (7.38 / 6.37 / 5.82 / 5.23 / 3.81 kg/km, n=80 each), so the
following comparisons are against a faithful replication, not a re-analysis.

### 2.1 ❌ **BLOCKER — the headline quantification is wrong.** §5.3.2, line 853

> "The range in mean efficiency across the five tested selection variants is **nearly four times** the
> corresponding range across constructors."

| Quantity | Value |
|---|---|
| Selection range (Table 3) | 7.381 − 3.809 = **3.572** kg/km |
| Constructor range (Table 1) | 6.649 − 5.359 = **1.290** kg/km |
| **Ratio** | **2.77×** |

Not "nearly four". Not even three. No alternative reading rescues it: on medians the ratio is 2.63×; on
max/min it is 1.56×; on the standard deviation of the level means it is 2.93×.

This is the single most consequential error in the paper because it quantifies the sentence the Abstract,
the Introduction (line 128), the Discussion (line 1155) and the Conclusion (line 1176) all rest on. A
reviewer who checks one number will check this one, because it is the paper's claim to a finding. Finding it
overstated by 44% invites the reviewer to distrust everything else — including the parts that are right.

**The fix is not simply "change four to three."** The two ranges are computed on *different* balanced slices:
the constructor marginal uses 456 runs (57 each) and the selection marginal uses 400 (80 each), because the
balancing rules differ for a stage inside `CELL_KEYS` versus one outside it. Comparing their ranges as
printed is therefore already slightly apples-to-oranges. Two defensible repairs:

- **(preferred)** Recompute both ranges on a single common slice and report the ratio with the slice stated.
- **(minimal)** Drop the numeric multiplier entirely and keep the qualitative claim — "spans a materially
  wider observed efficiency range than route construction (3.57 vs 1.29 kg/km)" — which is true, checkable,
  and sufficient for every downstream use the paper makes of it.

> **Disagreement with the prior review.** `.agent/reports/gemini/PAPER_EVALUATION_REPORT.md` (§2.1, "Generic
> Title") states the paper's most impactful finding is that "mandatory selection policies drive **4×** more
> efficiency variance than route construction algorithms" and recommends putting it in the title. That
> report reproduced the paper's error instead of catching it, and would have promoted a wrong number to the
> title. Do not act on that recommendation as written.

### 2.2 ❌ **BLOCKER — Eq. (4) does not describe the code that produced the results.** §4.1, line 397

The paper states the Service-Level rule as:

> ŵ_{i,d} + n_d·μ̂_i + **z·σ̂_i·√n_d** ≥ 100%

The implementation, `logic/src/policies/mandatory_selection/selection_service_level.py:60-64`:

```python
predicted_fill = (
    context.current_fill
    + (context.accumulation_rates * horizon_days)
    + (context.threshold * context.std_deviations * horizon_days)   # linear in n, not sqrt(n)
)
```

The σ term scales **linearly in n_d**, not as √n_d. The vectorised twin
(`logic/src/policies/vector/selection/service_level.py:83`) does the same. I traced the full wiring to rule
out a second code path:

- `logic/configs/policies/other/ms_service_level.yaml` defines the two benchmarked variants as
  `service_level1: {confidence_factor: 0.84, horizon_days: 1}` and
  `service_level2: {confidence_factor: 0.84, horizon_days: 2}`.
- `logic/src/pipeline/simulations/actions/node_selection.py:170` passes `horizon_days` into
  `SelectionContext` from those YAML params.
- `SelectionContext.horizon_days` is a real field (`selection_context.py:71`), so the value reaches the
  selector. SL1 and SL2 are genuinely different, and they differ only in `horizon_days`.

**Consequence.** At n_d = 1 the two forms coincide, so SL1 is unaffected. At n_d = 2 the code's safety margin
is `2 × 0.84σ̂ = 1.68σ̂` where the paper's stated equation gives `0.84 × √2 σ̂ = 1.188σ̂` — **the SL2 actually
run is ~41% more conservative than the SL2 the paper defines.** SL2 is the variant that anchors the
low-overflow / low-efficiency end of the frontier (3.81 kg/km, 1.2 overflows) — i.e. it is load-bearing for
the headline result in §2.1. A reader reimplementing Eq. (4) will not reproduce Table 3.

Two aggravating details:

1. The paper's √n_d is the *statistically correct* form for a sum of n iid daily increments, and the
   codebase knows it — `selection_multi_day_prob.py:70` uses `np.sqrt(horizon_days)` for exactly this
   quantity. So this is a divergence between two implementations in the same repo, and the paper documents
   the one that was not used.
2. The paper's follow-on sentence — *"holding z constant, so the two variants differ in how far ahead they
   look and not in how conservative they are at a given distance"* — is specifically an argument that the √n
   scaling makes conservatism comparable across variants. Under the linear-n code that argument is simply
   false: SL2's margin is exactly 2× SL1's at fixed σ̂.

**Required action.** Decide which form is intended, then either (a) fix the code and re-run the SL2 cells, or
(b) correct Eq. (4) to `+ z·σ̂_i·n_d`, delete the "not in how conservative they are" clause, and note the
deviation from the standard √n form. **(b) is cheaper and honest; (a) is better science.** Either way this
cannot ship as-is.

### 2.3 ⚠️ **z = 0.84 is never stated in the paper.** §4.1

Eq. (4) introduces *z* as "a fixed safety coefficient" and the text says it is held constant, but its value
appears nowhere. It is 0.84 (≈80th percentile) in `ms_service_level.yaml`. One of the three benchmarked
selection strategies is therefore not reproducible from the paper. Same class of omission as the missing
r_w / c_km / Q constants noted below (§4.2) — but worse, because *z* is the only free parameter of the rule.

### 2.4 ⚠️ **The Look-Ahead cost claim is unsupported and the paper's own table points the other way.** §4.1, line 407

> "It is the most computationally expensive of the three rules per decision because it simulates forward
> instead of evaluating a closed form."

No per-decision timing is reported anywhere. The only timing evidence in the paper is Table 3's Time column,
which orders CF90 892 s < **LA 914 s** < CF70 1,171 s < SL1 1,376 s < SL2 1,941 s — LA second *cheapest*.

I want to be precise about what this does and does not show: that column is whole-run wall-clock, dominated
by constructor work and by how many days a variant collects on, so it does **not** refute a per-decision
claim. But the paper asserts a cost ordering with zero measurement while its only published timing points the
opposite way, and a reviewer will notice the tension. Either instrument the selection stage and report
per-decision cost, or soften to a structural statement ("advances state day by day rather than evaluating a
closed form").

The mechanism claim is also loose. `selection_lookahead.py` advances at the *estimated mean* accumulation
rate — a deterministic arithmetic projection unrolled as a loop, functionally `(C − w)/μ̂`. Calling that
"simulates forward" oversells it, and the accompanying claim that LA "assumes no parametric demand
distribution" does not distinguish it from SL, which also uses only online moment estimates. The real
difference is that LA uses the mean alone and SL adds a σ̂ term.

Relatedly — and I think this is interesting rather than a defect — LA's mandatory trigger
(`_should_bin_be_collected`: `w + μ̂ ≥ C`) is **exactly SL with z = 0 and n_d = 1**. That is a clean
structural relationship which would explain why LA lands between CF90 and CF70/SL1 on the frontier, and the
paper would be stronger for stating it. *Scope note:* I verified the trigger; `_calculate_next_collection_days`
adds collection-day synchronisation on top, so the equivalence is at the trigger only, not the whole rule.

### 2.5 Minor numeric errata

| # | Location | Claim | Actual | Note |
|---|---|---|---|---|
| a | line 810 | ACO-HH "within 5% of BPC" | (6.65−6.30)/6.65 = **5.26%** | Narrowly false. Say "within about 5%" or "5.3%". |
| b | line 811 | ACO-HH "roughly a quarter of the HGS and PG-CLNS means" | HGS 589/2309 = 25.5% ✓; PG-CLNS 589/1967 = **30.0%** | A quarter of HGS, a *third* of PG-CLNS. Split the claim. |
| c | line 826 | "from ACO-HH at 1,219 s to SANS at 5,451 s" at N=350 | ACO-HH **1,218.65**, ALNS **1,219.65** | A 1-second dead heat, not a win. The 4.5× spread is correct; naming ACO-HH the fastest at N=350 is a coin flip. |

Verified-correct, for the record: the 456/57-per-constructor balance; every cell of Tables 1, 3 and 6; the
Pareto-membership counts (PG-CLNS 5, PSOMA 3, HGS 3, BPC 2, ACO-HH 1, ALNS 1 — exact); the 4.5× N=350 runtime
spread; BPC at 2,430 s; the 456 km / 49 km-per-event and 751 km / 340 km-per-event selection increments and
their ~7× ratio; the monotone ordering of all four measures across the five variants; 174 runs at 90 days;
the +0.26 kg/km paired change; HGS as the only constructor whose efficiency falls (−0.47). The generated
tables are trustworthy. **It is specifically the hand-written prose numbers that are unreliable** — which is
the strongest possible argument for extending the generator to emit the derived quantities in §5.3.1–5.3.2
as macros too.

---

## 3. Figures — the weakest part of the paper

> **Disagreement with the prior review.** Gemini scored "Figures & LaTeX Typography 8.5/10" with "minor font
> sizing issues". I looked at every rendered PNG. That score is not defensible: Fig. 6 has unreadable axis
> labels, Fig. 4 draws an incorrect Pareto front, and Fig. 2 asserts in bold the opposite of what §4.2 says
> was done. Those are not font-sizing issues. **C−.**

### 3.1 ❌ Fig. 6 (`strategy_tradeoff_30d.png`) — x-axis labels overlap into illegibility

The five category labels collide: the axis renders as
`Last-Minute (CF90)Look-AheadLast-Minute (CF70)ervice-Level (SL1)ervice-Level (SL2)` — the "S" of both
Service-Level labels is literally overprinted away. **This is the figure illustrating the paper's headline
result.** It cannot be submitted in this state. Fix: use the short codes (CF90 / LA / CF70 / SL1 / SL2) that
the rest of the paper already uses, or rotate 30°.

Beyond the rendering bug, the *chart type* is the wrong choice. A dual-axis bar chart with independently
scaled axes lets the two CF90 bars render at identical height, visually asserting an equivalence between
7.38 kg/km and 14.9 overflows that means nothing. The caption promises a "trade-off" the geometry cannot
show. Fig. 4 already establishes efficiency-vs-overflow scatter as this paper's visual idiom for exactly this
trade-off; **redraw Fig. 6 as a five-point connected scatter in that same plane.** The frontier shape becomes
visible, the monotonicity claim becomes readable off the figure, and the paper gains visual consistency.

### 3.2 ❌ Fig. 4 (`pareto_30d.png`) — the drawn non-dominated front is wrong

The caption says the dashed line is "the non-dominated empirical front". It runs PG-CLNS → BPC and stops.
But **ALNS (5.9 overflows, 6.13 kg/km) is non-dominated**: it has the minimum overflow count of all eight
constructors, so nothing dominates it, and PG-CLNS (6.0, 6.31) does not — PG-CLNS has *more* overflows. ALNS
should be the leftmost point on the staircase. The correct front is ALNS → PG-CLNS → BPC.

This matters beyond the geometry: §5.3.1 credits ALNS with "the fewest mean overflows (5.9)", so the text and
the figure disagree about whether ALNS is a frontier method. Fix the front computation in
`gen_paper_latex.py` (it looks like a strict-inequality / tie-handling bug at the boundary).

### 3.3 ❌ Fig. 2 (`simulation_loop.png`) — contradicts §4.2, and the arrow occludes text

Two separate problems, both serious.

**(a) The figure asserts the opposite of what was run.** Box 2 reads "IoT fill-level sensors add noise
ε ~ N(0,σ²)" and carries a bold highlighted callout: **"Observation Asymmetry: Policies receive noisy f̃_t
(Ground truth f_t is hidden)"**. Stage 1 reads "Input: Sensed f̃_t". But §4.2 (line 682) states plainly:
*"for this study we ran the simulations with the true observations being given as input to the routing
policies."* The LaTeX caption patches this with a parenthetical "(where, for this study, ε=0)" — that is not
enough. A reader who looks at the figure, as readers do, will come away believing sensing noise was studied.
The bold callout is the most emphatic element in the figure and it is false for this paper.
**Redraw the box for ε=0**, and if the noise capability is worth showing, show it greyed as "supported, not
exercised" rather than as the operative path.

**(b) The dashed next-day-transition arrow cuts diagonally across the whole diagram and strikes through the
Stage 1 and Stage 2 text**, partially obliterating "Input: Sensed f̃_t". Route it around the policy box or
below it.

**(c) Notation drift.** The figure uses `f_{i,t}`, `H`, `D` for fill / horizon / demand distribution; the body
uses `w_{i,d}`, `D`, `P_i`. `D` means *horizon* in the text and *demand distribution* in the figure. Unify on
the body's notation.

### 3.4 ⚠️ Fig. 3 (`networks.png`) — omits the one thing the paper's central mechanism needs

Three issues:

1. **No scale bar**, on two panels at explicitly different extents. The results section invokes spatial
   density directly ("the two cities differ in spatial density as well as scale") and this figure is the
   only evidence offered for it — without a scale bar it offers none.
2. **The depot is deliberately cropped out.** The paper's principal mechanistic explanation for its principal
   finding is that the depot sits ~5× the median inter-bin distance away (§5.1, §5.6, and the whole
   future-work item at line 1218). The one figure that could substantiate that removes the evidence to
   "preserve detail". At minimum add an inset, or a broken-axis arrow annotated with the 52.9 / 46.6 unit
   distances. This is a missed opportunity rather than an error, but it is a large one.
3. **Caption/figure contradiction on data source.** The caption says the geometry is "Google Maps (for Rio
   Maior) and OpenStreetMap (for Figueira da Foz)", but both panel legends read "OSM roads" and the credit
   line is "© OpenStreetMap contributors". The *distance matrix* for Rio Maior is Google; the *drawn basemap*
   is OSM for both. The caption conflates them. Rewrite.

Also worth stating: Rio Maior is drawn at N=170 only, though the study uses both N=100 and N=170 there.

### 3.5 ⚠️ Fig. 5 (`runtime_scaling_30d.png`) — one tick on a log axis

The y-axis is log-scaled and labelled with a **single** tick (10³). A reader cannot read any value off this
plot. Add minor-tick labels. Also: ALNS and HGS are two near-identical reds and are not separable in the
legend or the plot — recolour. And for a figure titled "runtime scaling", log-log would let a reader read off
the scaling exponent, which is the actual quantity of interest; log-linear with three x-points does not.

### 3.6 ⚠️ Fig. 1 (`policy_configuration_space.png`) — a full-width figure carrying one sentence

Three boxes listing names that the adjacent prose already lists, with a large empty band at the top. The
caption claims "the experiments in Sect. 5 cross **the highlighted subset** exhaustively" — nothing is
highlighted, and there is no visual distinction between the framework's larger registry space and the
benchmarked subset, which is the one thing the figure exists to convey. Either make the framework/benchmark
contrast visible (grey out the unexercised registry entries, show the counts of registered vs. benchmarked)
or cut the figure and recover a page.

### 3.7 ⚠️ Appendix figures use a different design language

`appendix_*.png` are heavy-bold presentation-deck exports; the main-text `Generated/*.png` are a light
seaborn style. Within one document the mismatch reads as unfinished. Additionally, in Fig. 9
(`appendix_pareto_30d.png`) the two panels use different x-scalings (the left shows a symlog axis with 0
and 10⁰ both ticked, the right a plain log from 10⁰) and different y-ranges, while being placed side by
side for visual comparison — and constructor identity, the factor a reader most wants at policy level, is
not encoded at all.

### 3.8 ✅ Fig. 7 (`improver_delta_30d.png`) and Fig. 8 (`fill_trajectory_30d.png`) are good

The sorted per-pair delta plot is exactly the right chart for its claim and makes "fewer but deeper losses"
immediately visible. The fill-trajectory panels isolate the temporal mechanism cleanly. Two notes: the
red/green encoding in Fig. 7 is a deuteranopia hazard (add a hatch or shift hue); and Fig. 8's "three
representative bins" needs a stated selection criterion — "representative" without a rule invites a
cherry-picking objection, and the caption's "re-simulated from the recovered daily increments" deserves a
sentence establishing the reconstruction is faithful, since this is the one figure not taken from stored
simulator output.

---

## 4. Structural and scientific issues

### 4.1 ❌ The Abstract promises a benchmark the body says does not exist

The abstract says the paper proposes "a methodology to adapt CVRP algorithms, including Hybrid Genetic
Search (HGS), Adaptive Large Neighborhood Search (ALNS), and **Neural Combinatorial Optimization (NCO)**"
and that "an extensive benchmark evaluates the solvers". §3 of the body says the opposite in terms:
*"Every stored 30- and 90-day benchmark row, however, uses one of the eight classical constructors … so the
reported results contain no learned-solver observation."*

I understand the constraint (issue #53: the abstract is the byte-identical abstract of record from the
conference; where the two disagree, the body moves). Within that constraint this is unfixable for the
proceedings version, and the right response is what the body already does — state the exclusion plainly and
early. But I will register the assessment the task asked for: **as a piece of writing, the abstract does not
sell this paper's actual contribution, and it sets up the reviewer's single most likely objection before the
paper has said a word in its own defence.**

The paper's real contribution — a reproducible multi-period simulator plus a *balanced, integrity-audited*
480-run classical baseline with an explicit account of what its design cannot identify — is more interesting
and more defensible than "we benchmarked some solvers including NCO". For any journal extension, rewrite the
abstract around: (i) the three-stage decomposition with common demand realisations, (ii) the finding that
selection outweighs construction on efficiency spread while construction dominates runtime and overflow
tails, and (iii) the integrity methodology. On catchiness specifically: the current abstract's first three
sentences are throat-clearing about municipal sanitation that any reader of this venue already accepts. The
hook is in sentence four ("routes in each period affect subsequent periods") and should be in sentence one.

### 4.2 ❌ No numeric constant of the objective appears anywhere

Eq. (3) introduces r_w (revenue/kg), c_km (cost/km) and Q_k (capacity) algebraically. Their values are never
given. Neither is *z* (§2.3), the ALNS decay λ and segment length φ, the annealing schedules, the population
sizes, the ng-neighbourhood size, or — most importantly — **the per-run time budget for BPC and SWC-TCF**,
which §4.2.1 says both exact methods run under and which determines whether their results mean anything.
The paper explicitly instructs the reader to read exact-method results as "best solution found within
budget" while never disclosing the budget.

This is a straightforward reproducibility failure and the cheapest blocker on the list to clear: one
appendix table of constants and per-constructor hyperparameters, plus the time budget inline in §4.2.1.

### 4.3 ⚠️ *K* is ambiguous: vehicles or trips?

Eq. (3) says the daily policy produces "up to K routes" and constrains `∑ w ≤ Q_k ∀k`, calling Q_k
"**vehicle** k's capacity". §4.2 says "operation is single-vehicle and single-depot … which restricts every
scenario to one vehicle and one depot". The paper never reconciles these. Either K = 1 for the study (in
which case say so, and the multi-route notation is dead weight), or K indexes *trips* by the single vehicle
within a day (in which case Q_k is not "vehicle k's capacity" and the formulation needs one sentence
defining a multi-trip period).

This is not pedantry: the CLS improver is described as able to "exchange route tails between vehicles" and
"move a bin between vehicles" (§4.3), which under a single-vehicle experiment is either impossible or means
something different from what it says. That description is doing interpretive work — it is the stated
mechanism for CLS's advantage over Fast-TSP — so the ambiguity propagates into a results claim.

### 4.4 ⚠️ Internal contradiction on sensing

- §4.1 line 373: *"All three strategies decide from **sensed** fill levels, **never from the true ones**"*
- §4.4 line 682: *"for this study we ran the simulations with the **true observations** being given as input
  to the routing policies"*
- Fig. 2: bold callout asserting policies receive noisy readings (§3.3)
- §6 Limitations: correctly says the noise question was not explored

Three of these four cannot all be right. Line 373 is the one to fix — it should say the strategies read the
*observed* fill signal, which in this study equals the true level, and that no strategy is given the
generating distribution or its parameters (which is the actually-important point that sentence is making).

### 4.5 ⚠️ Related work: two adjacent literatures are absent

33 references for a paper that reviews exact methods, trajectory and population meta-heuristics,
hyper-heuristics, matheuristics *and* NCO is thin, and the gaps are not random:

- **Inventory routing.** A problem where stock accumulates at customers between visits, service resets it,
  and the planner chooses whom to visit across periods to trade routing cost against stock-out risk **is
  structurally an inventory-routing problem**. The paper's Eq. (1) is an IRP state transition with a
  maximisation objective. Not engaging that literature at all is the omission a routing reviewer is most
  likely to raise, because it is the closest classical analogue and it has decades of multi-period
  methodology the paper could be building on or distinguishing itself from.
- **Orienteering / prize-collecting routing.** The optional-service objective — choose a profitable subset
  and route it under a resource budget — is the orienteering and prize-collecting TSP/VRP family. The paper
  cites the Archetti–Speranza–Vigo VRP-with-profits chapter and stops there. Given that the future-work
  section explicitly names the Capacitated Team Orienteering Problem, the absence is conspicuous.

Name the families and pick representative surveys; I am deliberately not reconstructing specific citations
from recall.

### 4.6 The design limitations the paper already knows about

I have nothing to add to §5.6 and the Limitations paragraph on substance — the paper's self-diagnosis is
correct and is tracked as issues #49 and #41. Three things worth stating as a reviewer:

- **R = 1 is the binding constraint.** One demand realisation per configuration means no dispersion estimate,
  no significance test, and no way to separate the horizon effect from regression to the mean. The paper
  says this. It is the right first item in Future Work and it should stay there.
- **The improver arm should probably be demoted, not defended.** §5.3.3 spends a page and a figure on a
  comparison the same section then says cannot be interpreted ("the stored runs expose an experimental-design
  problem rather than an isolated improver effect"). The honest and stronger move is to compress it to a
  short subsection stating the design flaw and what a controlled rerun would cost, and let the recovered
  space go to the selection and constructor results, which *are* identified. As written, the reader spends
  effort on a Δ = 0.74 kg/km headline that the surrounding prose withdraws.
- **The 90-day arm is correctly fenced.** The refusal to emit a cross-constructor 90-day aggregate — enforced
  in the generator, not just promised in prose — is the right call and should be said more loudly in the
  Introduction's contribution list.

### 4.7 ❌ Page budget: 28 body pages in LNCS

Compiled: 34 pages total, 28 of body before references. Typical LNCS proceedings limits are 12–16 pages
including references. Unless this venue has granted an extension, **this is a desk-rejection risk that
outranks every scientific point above**, and it is the one issue on this list that cannot be fixed in an
afternoon.

If a cut is needed, the material that costs least:
- **§3 Related Work taxonomy** (~2 pp). Excellent writing, but the four-family exposition is textbook
  material for this audience. Compress to one paragraph per family with the surveys carrying the load.
- **§4.2's eight constructor prose blocks** (~4 pp). Each runs 150–250 words of algorithm exposition for
  methods that are all published elsewhere and cited. A table of {method, family, source, VRPP adaptation
  mechanism, key parameters} plus one paragraph on the adaptation *pattern* they share would carry the same
  information in a page and would actually make the "the adaptation is uniform across the families" claim
  visible rather than asserted eight times.
- **Fig. 1** (§3.6) and the compressed improver subsection (§4.6) recover another page.

That is ~6 pages without touching a result.

---

## 5. Writing

**Strong overall — B+.** The prose has a distinctive, controlled register: short declaratives carrying real
content, minimal hedging where hedging isn't warranted and precise hedging where it is. Sentences like
"Waste collection introduces a different coupling: service resets a bin's state, while deferral raises its
future collection value and overflow risk" do a paragraph's work in a line. The Discussion's structure —
each stage yields a *different kind* of finding, and the paper says which kind — is genuinely good scientific
writing.

**The register breaks in recently-inserted passages.** These read as later additions and do not match the
surrounding voice:

- **line 1159–1166**: *"However, it is somewhat expected that the mandatory selection strategies would have a
  significant effect on the downstream routing phases, since it conditions them to be obligated to visit the
  selected bins."* Conversational, grammatically slipped ("it conditions them"/plural antecedent), and
  self-undermining in the paper's *concluding* argument — it tells the reader the headline finding was
  predictable, immediately after the section arguing it is important. The observation is worth keeping;
  reframe it as the mechanism ("the selection stage bounds the constructor's feasible set, which is why its
  effect propagates") rather than as a concession.
- **line 1207–1222**: the Future Work paragraph is a single ~16-line run-on sentence chaining six unrelated
  items through "as well as … and … Furthermore … Finally". Break into a short list.
- **line 1181–1188**: similar run-on in the Conclusion, ending on the awkward "making them lack a bin subset
  selection mechanism without a upstream phase" (also: "a upstream").

**Other:**
- The Conclusion's opening (line 1170) was recently rewritten and is now much clearer than the version it
  replaced, but it has become a 3-clause sentence carrying the paper's thesis. Split it.
- "SLSL2" appears in Table 6's Policy column against "SL2" everywhere else.
- Line 1128: "the experimental design cannot identify that mechanism" — elsewhere consistently "the design".

---

## 6. Bibliography hygiene

| Entry | Issue |
|---|---|
| `LYSGAARD2004` | **Objectively incomplete**: no volume, no pages, no DOI. Fix. |
| `WENTGES2006` | **Internally inconsistent**: `year = {2006}` but the DOI string encodes 1997 and the journal volume given is ITOR vol. 4. Verify against source and align key, year and DOI. |
| `GU1999` | **Internally inconsistent**: `year = {1999}` with INFORMS JOC vol. 10. Verify. |
| `BARNHART1970` | Key says 1970, entry correctly says 1998. Invisible in output — lowest priority, but rename while you're in there. |
| `inbook`, `inproceedings`, `f386d3b9a20d4a43a28a2d07aed60460` | Raw Mendeley/exporter default keys. Harmless in output; a maintenance hazard, and a signal the `.bib` has not been curated. |

33 of 71 entries are cited; the rest are shared-file overhead and fine.

---

## 7. Prioritised action list

### Blockers — must clear before submission
1. **Fix the 2.77× erratum** (§2.1). Recompute on a common slice or drop the multiplier. Highest priority:
   it is the paper's headline number and it is wrong.
2. **Reconcile Eq. (4) with the code** (§2.2). Fix the code and re-run SL2, or correct the equation and
   delete the conservatism clause. Do not ship the current mismatch.
3. **Redraw Fig. 6** — the axis labels are unreadable and it illustrates the headline result (§3.1).
4. **Fix Fig. 2** — remove the false noisy-sensing callout and reroute the occluding arrow (§3.3).
5. **Fix Fig. 4's Pareto front** — ALNS is non-dominated and is omitted (§3.2).
6. **Resolve the page budget** against the actual venue limit (§4.7).

### High — cheap and materially strengthening
7. Appendix table of all constants and hyperparameters, including *z* = 0.84 and the exact-method time
   budgets (§2.3, §4.2).
8. Resolve the *K* = vehicles-or-trips ambiguity and repair the CLS "between vehicles" description (§4.3).
9. Fix the §4.1 / §4.4 sensing contradiction (§4.4).
10. Fix Fig. 3 (scale bar, depot, Google/OSM caption) and Fig. 5 (log-axis ticks, ALNS/HGS colours).
11. Fix minor numeric errata 2.5(a)-(c), and extend `gen_paper_latex.py` to emit the derived prose numbers
    as macros so this class of error cannot recur.
12. Engage the inventory-routing and orienteering/prize-collecting literatures (§4.5).

### Medium
13. Compress the improver subsection to its design-flaw statement (§4.6).
14. Repair the three register-break passages (§5).
15. Bibliography fixes (§6).
16. State the Fig. 8 bin-selection criterion; recolour Fig. 7 for colour-blind safety.

### For the journal extension
17. Multi-seed replication — the single change that converts most of the paper's descriptive claims into
    inferential ones.
18. Controlled improver rerun from a fixed deterministic constructor (issue #49).
19. Full 90-day factorial, or an explicit selection-effect estimate.
20. A depot-inside-service-area network, to test the paper's own central geometric explanation.
21. Rewrite the abstract around the actual contribution (§4.1).

---

## 8. Closing assessment

The parts of this paper that were **generated** are trustworthy — I reproduced every table cell I checked
from the raw CSVs, and the exclusion and balancing logic is more careful than the norm in this literature.
The parts that were **typed** are not: the headline multiplier is off by 44%, one strategy's defining equation
does not match the code that produced its results, three smaller numbers are loose, and three figures carry
defects a reviewer will see in the first pass — including one that asserts in bold the opposite of what the
methods section says was done.

That is an unusually specific diagnosis, and it points at an unusually specific fix: **the boundary between
generated and hand-written content is exactly where the errors live.** Push the generator boundary outward
to cover the derived numbers in the prose, and most of this class of defect becomes structurally impossible.

The underlying science is sound and the intellectual honesty — particularly §5.6's "one methodological hazard
appears in three forms" — is a real asset that would survive peer review well. Fix the six blockers and this
is a solid, publishable contribution whose methodological section is worth more than its results section, and
should be positioned accordingly.
