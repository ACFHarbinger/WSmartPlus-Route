# Holistic Evaluation of the MPVRPP Simulation-Framework Paper

**Manuscript:** *A Simulation Framework for the Multi-Period Vehicle Routing Problem with Profits in Smart Waste Collection*  
**Authors:** Afonso Cruz Fernandes, Héctor Bonilla-Londoño, Diana Jorge, Manuel Lopes, Pedro A. Santos, Tânia Ramos  
**Format:** Springer LNCS (compiled PDF: 34 pages, US Letter, dated 28 August 2026)  
**Sources examined:** `paper.tex` / `paper.pdf`, generated tables and figures, `mybibliography.bib`, simulation logs under `assets/output/30days/`, summary CSVs, Hydra run configs, and the corresponding Python implementations of selection, construction, and the simulator.  
**Reviewer:** Grok 4.6 (xAI)  
**Stance:** Tough but fair. Credit is given where it is earned. Claims that do not survive contact with the logs, the code, or the published tables are stated as such.

---

## 0. Verdict

This is a **serious piece of applied OR engineering** wrapped in a **paper that is not yet ready to be believed at face value**.

The architectural idea — a three-stage policy (mandatory selection → profitable construction → geometric improvement) evaluated on real municipal networks under paired demand realizations, with an unusually honest integrity protocol — is the real contribution. The empirical headline that *who you force onto the truck matters more than which CVRP solver builds the tour* is interesting, plausible, and supported by the balanced 30-day tables.

Around that core, the manuscript currently asks the reader to accept a number of things that are false, overstated, or simply not in the document:

| Claim in the paper | What is actually true |
|---|---|
| The study evaluates NCO (abstract) | Zero learned-solver rows in any table |
| Selection efficiency range is “nearly four times” the constructor range | Published tables give **≈ 2.8×** |
| 90-day runs are the 30-day Pareto subset | **All 60 BPC configs** were rerun; **ALNS was not rerun at all**, despite sitting on a 30-day front |
| Operation is single-vehicle with a capacitated VRPP | One route per collection day, yes; several constructors load **2–3×** the physical truck payload under Gamma-3, and shift duration is not enforced |
| Service-Level uses a \(\sqrt{n_d}\) safety term | The implementation uses a **linear** \(n_d \cdot z \cdot \hat\sigma_i\) term |
| Look-Ahead “simulates” a non-parametric demand process | It is a **deterministic mean-rate projection** plus a bundling pass |
| The daily problem maximises profit \(\mathcal{P}\) | Tables rank **kg/km** and overflow counts; profit is in the logs and is never reported |

A reviewer who checks the logs will not give the authors the benefit of the doubt on the rest. The right outcome for a journal (EJOR, COR, Transportation Science, Networks) is **major revision**. For a short LNCS conference slot the paper is also **too long, too mismatched to its abstract, and too under-specified to reproduce**.

**Overall: 6.0 / 10** as currently written. The work underneath is closer to 7.5 if the claims are aligned with the experiment and the missing numbers are put on the page.

---

## 1. What the paper is, and what it pretends to be

There are two papers fighting inside this document.

**Paper A (the one that was actually run).** A classical-OR benchmark of 8 constructors × 5 selection variants × 2 improvers × 2 demand processes × 3 network sizes, 480 stored 30-day runs, 174 stored 90-day runs, on plastic bins in Rio Maior and Figueira da Foz. The interesting scientific object is the *selection* stage, not the zoo of constructors.

**Paper B (the one the title, abstract, and related work advertise).** A general simulation *framework* for stochastic MPVRPP that hosts Neural Combinatorial Optimization, noisy IoT sensing, multi-vehicle / multi-depot dispatch, and a methodology for adapting arbitrary CVRP algorithms.

Paper B is a software claim. It may even be true of the repository. It is **not** what the experiment measures. The body of the paper is more honest about this than the abstract — the introduction says the architectural scope is “broader than the experiment reported below,” and the NCO paragraph eventually admits that “every stored 30- and 90-day benchmark row … uses one of the eight classical constructors.” That admission is buried after a page of Pointer Networks, AM, POMO and DIFUSCO. A reader of the abstract never hears it.

A framework paper without a **software-availability statement** is also incomplete. The compiled manuscript never points to a repository, a DOI, a license, or a version pin. For a paper whose first contribution is “a reproducible simulator,” that is not a minor omission. It is the difference between a benchmark and a screenshot of a benchmark.

---

## 2. Scores by dimension

| Dimension | Score | One-line reading |
|---|---:|---|
| Title and abstract | 4.5 | Accurate as a software slogan; false as a description of the study |
| Introduction and motivation | 7.5 | Clean, well paced; the contribution list is better than the abstract |
| Problem formulation | 5.5 | Compact and mostly right; coefficients, loss, capacity, and \(K\) are missing or inconsistent |
| Related work | 6.0 | Right local citations; wrong neighbourhood (IRP / TOP / PVRP almost absent); NCO overweighted |
| Method / policy interface | 8.0 | The three-stage split is the best idea in the paper |
| Algorithmic descriptions | 6.5 | Broad and readable; not reproducible from the text; PG-CLNS is underspecified for an “original” method |
| Experimental design | 5.0 | Factorial 30-day idea is sound; \(R=1\), no time windows, unpublished solver budgets, confounded improvers, misdescribed 90-day sample |
| Data integrity and honesty | 8.5 | Genuinely good. This is the part other OR benchmark papers should copy |
| Analysis and claims vs. tables | 5.0 | Several numerical and design claims do not match the tables or the CSVs |
| Discussion and limitations | 8.0 | Unusually careful about causal overreach; then the prose frays |
| Figures and tables | 6.5 | The generated pipeline is a strength; several figures are broken or dishonest about what they contain |
| Writing | 7.0 | Scholarly for most of the body; frozen abstract; sloppy close |
| Reproducibility | 4.0 | No hyperparameters, no \(r_w, c_{km}, Q\), no time limits, no code URL, one seed |
| **Overall** | **6.0** | Major revision. Do not send this to a top OR journal as-is |

---

## 3. Strongest parts (and how to make them stronger)

### 3.1 The three-stage policy split is the right model of the problem

Waste collection with smart bins is not “CVRP with a funny objective.” It is a *when* decision (which bins are unignorable today), a *what* decision (which extra bins repay the detour), and a *how* decision (the tour geometry). Folding those into a single solver is exactly why the literature is incomparable. Separating them, and putting each behind a registry, is the paper’s genuine intellectual contribution.

**Make it stronger.**

- Draw the daily information flow as a *contract*, not a marketing diagram: inputs, outputs, invariants (“constructors may not drop a mandatory bin”; “improvers may not change the visit set”). The current Fig. 1 is a labelled pipeline, not a specification.
- State, formally, that construction is a *myopic* single-period VRPP and that the only multi-period intelligence in the reported policies sits in selection. That sentence is currently implied. It should be theorem-like, because it is the reason selection dominates the tables.
- Add a “null selection” cell (empty mandatory set; constructors free) and a “must-collect-all” cell. Without those, the claim that “selection is the large knob” is a comparison among five *flavours of forcing*, not a comparison of forcing against not forcing. The discussion already *wants* this ablation. Do it.

### 3.2 Integrity-aware analysis

This is the best methodological paragraph in the manuscript, and it is not close.

Four SWC-TCF runs are degenerate (including a 90-day run that dies on day 13 with 23,886 overflow events). The authors (i) detect them by tonnage shortfall against a *paired* scenario median, not by overflow; (ii) drop the whole scenario cell for every constructor, not just the failing one; (iii) further restrict selection and improver marginals to slices where every level of the compared factor actually ran; (iv) generate the tables from code rather than by hand. The arithmetic checks out: 3 thirty-day cells × 8 constructors = 24 rows removed, 21 of them innocent, 456 remaining, 57 per constructor.

Most benchmark papers silently average the survivors. These authors did not.

**Make it stronger.**

- Put the detection rule in a numbered definition, with the 0.071 vs {0.30, 0.47, 0.47, 0.86} gap as a figure, not only as prose.
- Report **kg collected** and **kg lost** in the constructor table. Tonnage is the quantity you used to throw runs out; it is perverse not to show it as a result. The logs already contain both.
- Stop putting the degenerate points back into the appendix figures “with a caveat.” A caveat on a 2,168-overflow outlier that sets the x-axis is not a caveat, it is an axis hijack. Filter the appendix the same way you filter the tables, or move the raw points to a clearly marked “failure gallery.”

### 3.3 Paired demand realizations

Re-seeding so that every algorithm in a scenario sees the same bin-by-bin, day-by-day fill path is the correct way to run this experiment. Combined with the remote-depot geometry (median depot–bin distance ≈ 5× median bin–bin distance), it gives the selection result a *mechanism*: deferral amortises a large fixed leg over a larger load. The single-bin fill trajectories (Fig. 4) make that mechanism visible. That figure is the best graphic in the paper.

**Make it stronger.**

- Name the seed (`seed: 42` in the Hydra configs) and the pairing procedure in enough detail that a reader could re-implement it without the repo.
- Add a *nearby-depot* counterfactual, as the discussion already requests. Until that exists, “selection dominates construction” is a statement about **two Portuguese networks with a remote depot**, not about MPVRPP.
- Show the same three bins under Look-Ahead / SL as well as CF70 / CF90, so the figure argues for the whole frontier, not only for a threshold knob.

### 3.4 Honest discussion of what the design cannot support

The improver section refuses to call the CLS vs Fast-TSP delta a treatment effect, because constructor output was not held fixed. The horizon section refuses a cross-constructor ranking at 90 days. The discussion says, in plain language, that a filter correlated with the factor under comparison can *determine* the comparison. This is more scientifically adult than the average COR paper.

**Make it stronger.**

- Do not then undermine that honesty with a 90-day sampling story that the CSVs contradict (Section 5.2 below). Honesty about a design is only valuable if the design description is accurate.
- The last two paragraphs of the discussion and the whole conclusion currently read like a lab notebook. Cut them by a third. Keep the valuation-of-overflows point; it is the operational takeaway.

### 3.5 Breadth of the constructor panel, used as a *panel*

Eight constructors spanning compact MILP, branch-price-and-cut, ALNS, HGS, a domain SANS, a swarm memetic, an ACO hyper-heuristic, and a home-grown pheromone/LNS hybrid is a lot of work. The useful move is that they are treated as a *panel over which to average selection*, not as a bake-off the authors need to win. PG-CLNS as “never leads a column, on 5/6 fronts” is the right way to talk about a compromise method.

**Make it stronger.**

- Publish the per-day time limit (60 s constructor + 30 s improver in the stored configs), `exact_mode: false` for BPC, ACO-HH’s 10 ants / 50 iterations, and the physical \(Q\) and \((r_w, c_{km})\). Without those, “ACO-HH is the fastest” is a statement about a lightly budgeted hyper-heuristic, not about ACO-HH.
- Add one dumb baseline: nearest-neighbour with profit screening, or “collect the mandatory set only, in distance order.” Right now the weakest named constructor still looks like a serious algorithm. The floor is missing.

---

## 4. Weakest parts (and how to fix them)

### 4.1 The abstract is a frozen, slightly false advertisement

A comment in `paper.tex` (lines 73–77) forbids editing the abstract because it was the conference abstract of record. That may be a bureaucratic constraint. It is not a scientific one. The abstract:

1. Lists NCO among the algorithms the methodology adapts, in the same breath as HGS and ALNS.
2. Promises “an extensive benchmark evaluates the solvers” without saying the solvers that were evaluated are classical.
3. Does not mention the actual headline (selection ≫ construction on the observed efficiency–service range).
4. Does not mention real Portuguese networks, 30 vs 90 days, or the integrity protocol.

**Fix.** If the venue forbids changing the conference abstract, put a *structured* “Extended abstract / highlights” box in the introduction that states, in one sentence each: what was run, what was not run, and what the main empirical result is. If this is going to a journal, rewrite the abstract from scratch. A catchy abstract for *this* paper is:

> Smart bins turn waste collection from a must-visit CVRP into a multi-period profitable-routing problem: today’s tour resets tomorrow’s fill. We separate that problem into mandatory selection, route construction, and geometric improvement, and we replay identical demand on two Portuguese networks for 480 thirty-day runs. Across five selection rules and eight classical constructors, selection moves efficiency by about 2.8× more than construction does. Constructors differ mainly in runtime and overflow tails. No learned solver is evaluated.

That is not poetry. It is true.

### 4.2 The NCO material is not a contribution. It is a residue

Related-work paragraphs on Pointer Networks, AM, POMO, DIFUSCO, and “learning to search” would be justified in a paper that trains one of them, or even in a paper that *runs a frozen checkpoint* through the new interface. This paper does neither. The Images tree still contains AM / DDAM / TransGCN architecture PDFs and training-loss screenshots that are not used. That is what leftover paper B looks like on disk.

**Fix.** Cut the NCO review to four sentences and one survey citation. Move “the interface can host learned constructors” to future work, which is where it already lives. Do not list NCO in the keywords.

### 4.3 The 90-day experiment is misdescribed

The paper says: only 30-day Pareto-front configurations were carried forward (174 runs); therefore no cross-constructor comparison is reported; 165 paired configs are compared to themselves.

The summary CSVs say something else.

| Constructor | 30-day runs | 90-day runs |
|---|---:|---:|
| BPC | 60 | **60** |
| PG-CLNS | 60 | 42 |
| ACO-HH | 60 | 30 |
| HGS | 60 | 12 |
| SANS | 60 | 12 |
| PSOMA | 60 | 12 |
| SWC-TCF | 60 | 6 |
| ALNS | 60 | **0** |

BPC was rerun in full. ALNS, which the paper itself places on 1 of 6 constructor-level Pareto fronts, was not rerun at all. The 90-day overflow heatmap in the appendix is visually a BPC / PG-CLNS / ACO-HH / SL-heavy slab, not a Pareto catalogue.

This is not a small wording issue. The entire epistemic status of Section 4.3.4 (“Effect of the Planning Horizon”) depends on *what* was selected. “Paired against itself, on a performance-selected subset” is a different claim from “paired against itself, on an undocumented convenience subset that happens to contain every BPC run.”

**Fix.** Describe the 90-day sample as it is. If it was “everything that finished in time plus a hand-picked extra,” say that. If it was supposed to be Pareto and the pipeline drifted, say that. Recompute Table 5 only on configs that were *actually* non-dominated at 30 days, and put ALNS back or explain its absence. Do not use the word Pareto for a set that contains 60/60 BPC and 0/60 ALNS.

### 4.4 “Nearly four times” is not what the tables say

Constructor efficiency range, balanced 30-day table: \(6.65 - 5.36 = 1.29\) kg/km.  
Selection efficiency range, balanced 30-day table: \(7.38 - 3.81 = 3.57\) kg/km.  
Ratio: **2.77×**.

Unbalanced raw means from the 480-row CSV give 2.75×. Medians give ~2.6×. There is no reasonable reading of the published numbers that produces “nearly four.”

This is the paper’s headline empirical sentence. Getting the factor wrong, while hedging in the next sentence that it is “not a general variance decomposition,” is the worst of both worlds: a strong number, incorrectly computed, with a disclaimer that tries to make the strong number un-attackable.

**Fix.** Compute the ratio from the same balanced slices you already built, report it as ~2.8×, and, if you want a variance decomposition, do one (e.g. a simple ANOVA or a range-normalised main-effect plot) instead of comparing two ranges taken on *different* slices (57-run constructor cells vs 80-run selection cells).

### 4.5 The stated objective is not the reported objective

Equation (2) maximises

\[
\mathcal{P} = r_w \sum w_{i,d} - c_{km} \sum \mathrm{dist}.
\]

The stored logs contain `profit` and `reward`. The tables report kg/km, overflows, km, and runtime. Overflow and lost mass **do not appear in \(\mathcal{P}\)**. CLS, by the paper’s own admission, scores moves on distance only. Fast-TSP is a TSP solver.

So the pipeline is: constructors (mostly) maximise a linear profit; improvers minimise distance; the paper ranks a ratio and a service count that nobody optimised.

That is a legitimate *evaluation* choice — operators care about kg/km and overflows — but then:

- \(\mathcal{P}\) is not the object of the study, and should not be written as if the policies solve it over the horizon. The multi-period “objective” (equation after (2)) is not what any reported policy maximises. Selection is a heuristic; construction is myopic; improvement is geometric.
- Profit *rankings are not the same as kg/km rankings*. From the 30-day CSV, mean profit orders SANS above PSOMA above SWC-TCF; mean kg/km orders SWC-TCF above PSOMA above SANS (and after the integrity filter the published table is BPC ≻ PG-CLNS ≻ ACO-HH ≻ HGS ≻ ALNS ≻ SWC-TCF ≻ PSOMA ≻ SANS). If you refuse to show profit, you cannot call this a VRPP paper without wincing.
- \(r_w\) and \(c_{km}\) are never given. In the run they are **\(r_w = 0.65 \times 898/1000 = 0.5837\) €/kg** (plastic) and **\(c_{km} = 1.0\) €/km**. I verified this against day-level `profit` in the BPC logs: \(0.5837 \cdot 190 - 96.167 = 14.736\). A reader cannot reconstruct the trade-off without those numbers.

**Fix.** Add a parameter table: \(r_w\), \(c_{km}\), \(Q\) (by city), bin volume, density, shift length (or “shift unenforced”), sensing noise (\(\sigma=0\)), constructor time limit, improver time limit, seed, \(R=1\). Add a profit column. Add a kg-lost column. Say explicitly that kg/km is an *ex post* operational KPI, not the training / search objective.

### 4.6 Capacity, fleet size, and shift: the experiment is not the formulation

The formulation has a fleet of \(K\) vehicles, capacities \(Q_k\), and a complete graph. The protocol paragraph says the experiment is single-vehicle, single-depot. The Hydra configs for the stored runs set `n_vehicles: 0`, which the BPC adapter treats as **no vehicle limit**. In practice every collection day produces **one** depot-to-depot tour (no intermediate returns).

That one tour is not a plausible shift:

- Figueira da Foz, Gamma-3, HGS, Look-Ahead: a single route with **318 stops** and **4,702 kg**.
- Figueira da Foz, Gamma-3, Last-Minute CF90, PSOMA / ALNS / PG-CLNS: peak daily loads **7,061–7,095 kg**.
- Physical truck payload in `get_area_params` is **2,500 kg** (Figueira plastic) and **3,500 kg** (Rio Maior plastic).
- BPC on Figueira *empirical* sits on 2,500 kg exactly on several days (capacity binding, good). BPC on Figueira *Gamma-3* sits on **4,999.6 kg** (the percent-converted \(Q\) used as if it were kilograms). ALNS / PG-CLNS / PSOMA go through even that inflated cap.

Jorge et al. (2022) — co-authored by two of the present authors, implemented here as SANS — is a paper *about workload and shift duration*. The present experiment turns that method loose with no \(T_{\max}\). Future work then proposes “working-shift duration constraints (e.g., the Capacitated Team Orienteering Problem…)” as if this were a new idea rather than a constraint the authors already published and then dropped.

Collection time in the constants file is 3 minutes/bin. 318 stops × 3 min = 15.9 hours of service *before driving*. Calling this “single-vehicle operation” is true only in the degenerate sense that the decoder emitted one sequence.

**Fix, in order of honesty:**

1. Publish \(Q\) per city and the units the solvers actually see.
2. Audit daily load against \(Q\) and flag infeasible tours. Several named constructors are currently being scored on infeasible solutions.
3. Either enforce shift duration or stop citing workload-aware SANS as if it were the same algorithm.
4. If `n_vehicles: 0` was a config accident, say so and rerun. If it was deliberate “unlimited fleet, but the search happened to return one route because \(Q\) was large,” say *that*. Do not write “restricts every scenario to one vehicle.”

### 4.7 The Service-Level and Look-Ahead write-ups do not match the code

**Service-Level, paper:**

\[
\hat w_{i,d} + n_d\hat\mu_i + z\hat\sigma_i\sqrt{n_d} \ge 100\%.
\]

**Service-Level, `selection_service_level.py`:**

```text
predicted = current_fill + mu * horizon_days + threshold * sigma * horizon_days
mandatory if predicted >= 100
```

The safety term is \(z\,\hat\sigma_i\,n_d\), not \(z\,\hat\sigma_i\sqrt{n_d}\). The \(\sqrt{n}\) form is the right one for i.i.d. daily increments. The linear form is a different, more conservative rule. \(z\) is never given; the YAML uses `confidence_factor: 0.84`. The threshold in the implementation is whatever was stuffed into `context.threshold`.

**Look-Ahead, paper:** “the most computationally expensive of the three … because it simulates forward,” “assumes no parametric demand distribution.”

**Look-Ahead, `selection_lookahead.py`:** if `current + rate >= 100` today, mark mandatory; zero those bins; find the next day any remaining bin would hit 100 under a **constant** estimated rate; bundle anything that would overflow before that day. There is no sampling from \(P_i\), no sensor noise, no Monte Carlo. It is a deterministic bundling heuristic, in the spirit of Jorge et al. (2022), and it is cheap.

These are not documentation nits. SL1 vs SL2 is one of five points on the paper’s main frontier. If the equation is wrong, the frontier is mislabelled.

**Fix.** Typeset the rule that ran. Give \(z\). Describe Look-Ahead as a deterministic rate projection with a bundling pass, and drop “computationally expensive.” If you want the \(\sqrt{n}\) rule, implement it and rerun.

### 4.8 Formulation gaps (even before the experiment)

- **Lost mass is not a state variable.** Equation (1) uses \(\min(\cdot, C_i)\), so overflowed waste vanishes. The simulator tracks `kg_lost`; the math does not.
- **Sensing is not in the math.** Strategies are said to see \(\hat w\); the study sets \(\varepsilon=0\); the simulation-loop figure still draws \(\varepsilon\sim\mathcal{N}(0,\sigma^2)\) and “ground truth is hidden.” Pick one world and draw that world.
- **Notation drift.** Waste is \(\vec w_d\) in §2 and \(\mathbf{f}_t\) in Fig. 2; routes are \(\mathcal{A}_d\), \(\mathcal{A}_{d,k}\), \(\mathcal{A}_{d,\cdot}\) depending on the line; \(V\) is used both as a set and as \(|\mathcal{V}|-1\).
- **Complete graph vs road distances.** Google Maps and OSM matrices are not symmetric complete-graph distances. If the solvers see an asymmetric matrix, the formulation should.
- **No \(T_{\max}\), no service time, no number of vehicles in the instance data.** The daily VRPP as written is under-constrained relative to the operational story in the introduction.
- **Multi-period objective vs myopic solvers.** Writing \(\max_\pi \mathbb{E}[\sum_d \mathcal{P}_d]\) and then running independent daily VRPP solves with a myopic selector is fine, but then \(\pi\) is *not* an optimizer of that expectation. It is a composed heuristic. Say that.

### 4.9 Related work is pointed at the wrong map

What is present: Hess et al. 2024 (waste routing survey), Ramos et al. 2018 (two-commodity flow, implemented), Jorge et al. 2022 (SANS + Look-Ahead, implemented), Lopes et al. 2023, de Morais et al. 2024, Zhang et al. 2013 (multi-period VRP with profit), Archetti–Speranza–Vigo VRPP chapter, Vidal HGS, Røpke–Pisinger ALNS, Barnhart column generation.

What is almost absent, and should not be:

- **Inventory Routing** (Bell, Federgruen, Coelho–Cordeau–Laporte). Multi-period, inventory at nodes, routing, optional visits: that *is* this problem with a waste-flavoured inventory. Calling it MPVRPP without locating it next to IRP will read, to an EJOR referee, as not knowing the neighbourhood.
- **Team Orienteering / Profitable Tour / PCTSP** beyond one handbook chapter. The daily problem with a shift would be a capacitated TOP. The authors know this — future work names CTOP — and still do not review it.
- **Periodic VRP.** Different coupling (visit patterns rather than inventory), but it is the other standard multi-period routing family.
- **Operational waste-collection constraints** that Jorge 2022 treated as first-class (shift, balance) and this paper deleted.

The exact-methods subsection is a competent mini-lecture on ng-routes, subset-row cuts, and Farkas pricing. It is longer than the waste-collection related work. For a benchmark paper that never reports a dual bound or an optimality gap, that is the wrong centre of mass.

**Citation hygiene, specifically wrong or embarrassing:**

- `Lin2017` is cited for Farkas pricing. It is a 4OR paper on column generation for *multi-objective LP non-dominated sets*. That is not the reference. Use Lübbecke–Desrosiers, or the standard CG infeasibility / Farkas dual discussion in Barnhart et al. 1998 (which you already cite, under the key `BARNHART1970`).
- `BARNHART1970` is Barnhart, Johnson, Nemhauser, Savelsbergh, Vance, *Operations Research* **1998**. The key is a lie.
- `WENTGES2006` is Wentges 1997 (ITOR vol. 4).
- `LYSGAARD2004` title in the `.bib` is “capacitated vehicle problem” — the word *routing* fell out.
- Keys `inbook`, `inproceedings`, and `f386d3b9a20d4a43a28a2d07aed60460` (Røpke–Pisinger) should never leave a private Zotero dump.
- Duplicate Gu et al. 1999 (computation vs complexity) is fine if both are used; the keys should not look like two random Semantic Scholar scrapes.

### 4.10 PG-CLNS as “an original design”

A framework paper may include a new constructor. An original constructor in a framework paper must still be specified. What the reader gets is one paragraph of metaphor (pheromone on edges, LNS on individuals, HVPL as inspiration) and no:

- pseudocode,
- complexity,
- parameter table (population, rounds, destroy size, evaporation),
- ablation (pheromone only / LNS only / both),
- comparison to plain ALNS that would tell us whether the volleyball is doing any work.

HVPL (Sun et al. 2023) is a location-routing with simultaneous pickup-delivery algorithm. “Inspired by” is doing a lot of lifting. Either write a short algorithmic subsection that a competent student could reimplement, or demote PG-CLNS to “a registered hybrid LNS, details in the supplementary code.”

### 4.11 Experimental design, beyond the 90-day story

- **\(R = 1\).** One demand realisation per cell. The pairing across algorithms is a strength; the absence of pairing *across realisations* is a hole. Every mean in Table 1 is a mean over scenarios, not over noise. The paper admits this. It should also stop using the word “stochastic” as if the policies were evaluated against a distribution they were shown more than once.
- **No statistical tests, no intervals.** With \(n=57\) constructor cells you can at least show a bootstrap over scenarios. Medians next to means is good and not a substitute.
- **Improver confounding**, already admitted: then why is Table 3 still formatted as a treatment comparison (CLS, Fast-TSP, \(\Delta\), wins)? The honest display is a scatter of (constructor output features) vs \(\Delta\) kg/km, or a rerun with a frozen tour.
- **Time budgets unpublished**, and not equal in effect: ACO-HH’s search is tiny; SANS and HGS run into the wall; BPC is `exact_mode: false` with a 60 s cap, greedy fallback, and is still called an “exact method” in the section heading. The paper does warn “best solution found within budget.” The heading and the related-work sermon on certificates of optimality do not.
- **Gamma-3 vs Empirical are different regimes and, in the logs, appear to stress capacity differently.** The paper is right not to pool them. It is not right to leave the capacity-unit question unexamined after BPC’s empirical cap is 2,500 kg and its Gamma-3 cap is 5,000 kg on the same network.
- **No industrial baseline.** Jorge et al. (2022) compared to the company’s plan and reported a ≥ 45 % profit lift. This paper, with two of the same authors and the same country, compares eight solvers to each other and never to current practice.

---

## 5. Section-by-section notes

### 5.1 Title

*A Simulation Framework for the MPVRPP in Smart Waste Collection* is a software title. The result that will be cited, if anything is, is the selection-versus-construction split. A better title would put that in front, e.g. *Mandatory selection, not tour construction, dominates efficiency–service trade-offs in multi-period smart waste collection* — with the framework as the vehicle, not the destination.

“MPVRPP” in the running head is ugly and not a standard acronym. “MPVRP with profits” is already a mouthful; adding a second P does not help.

### 5.2 Abstract and keywords

Catchiness: low-medium. It reads like a grant paragraph (critical municipal service, profound implications, lack of standardized environments, we propose a Python framework). The sentence that should land — *selection, not construction, is the first-order decision* — is not in it.

Keywords include Neural Combinatorial Optimization. They should not.

### 5.3 Introduction

This is the best-written section. Short sentences, a real operational picture (imperfect sensors, capacity, tomorrow’s state), a contribution list that already contains the integrity protocol. Two problems:

1. Contribution 3 still treats “learned constructors can use the same interface in future comparisons” as a present contribution. It is a property of the software, not a result.
2. The motivation is morally right and quantitatively empty. One number — fuel share of municipal waste cost, overflow fines, or the 45 % in Jorge et al. 2022 — would do more than “sanitation and public health.”

### 5.4 Problem definition

Keep equation (1); it is the right picture of a reset. Then add: a loss variable; a sensing equation; the actual \((r_w, c_{km}, Q, K, T_{\max})\) of the study; and a sentence that the *reported* policies do not optimise the horizon expectation.

The daily model is a capacitated profitable tour / orienteering problem with mandatory nodes. Saying that, and citing the TOP literature, would cost one sentence and save a referee report.

### 5.5 Related work

See 4.9. Compress exact methods. Add IRP. Cut NCO. Fix the bibliography keys before a copy-editor has to.

### 5.6 Methodology

The stage descriptions of ALNS, HGS, SANS, PSOMA, ACO-HH are good *intuition pumps* and bad *specifications*. For a framework/benchmark paper, move full operator lists and parameter tables to an appendix, but *have* the appendix. CLS’s “distance only, not profit” is an important and well-flagged property; it also means CLS can, in principle, harm \(\mathcal{P}\). Check whether it ever does, on the paired logs, and say so.

The simulation-loop figure is the most misleading graphic in the paper. It sells observation asymmetry and Gaussian sensor noise as part of “this study.” The caption quietly sets \(\varepsilon=0\). Draw the experiment you ran.

### 5.7 Experiments and results

The 30-day factorial display is the heart of the paper and is mostly well handled: balanced marginals, means and medians, constructor Pareto with the “spread is tails” reading, selection frontier with km-per-avoided-overflow, scenario split. Specific issues:

- Constructor Pareto (Fig. 3) would be more honest with median overflow on the x-axis, or with error bars across scenarios. As drawn, HGS looks like a disaster and ALNS/PSOMA look identical; the table’s medians say seven of eight constructors are tied at 4.0.
- Runtime-vs-\(N\) (Fig. 5) is a good figure. It is also the only place construction clearly separates. Lean on it harder: “if you already picked a selection rule, pick the constructor on the time–tail plane, not on mean kg/km.”
- Strategy bar chart (Fig. 6) has **colliding x-tick labels** (“Last-Minute (CF90)Look-AheadLast-Minute (C…”). This is below standard. Rotate the ticks.
- Improver waterfalls (Fig. 7) are fine as a *warning* plot. They are not fine as the only improver result.
- Scenario table: Empirical 4.71 kg/km vs Gamma-3 7.35, overflows 6.1 vs 15.0. Good. The paper should also show **kg lost**, because Gamma-3’s extra overflows may or may not be a lot of mass.
- Granular appendix Pareto is actually informative (selection colouring is the right encoding). Then the authors reintroduce the 2,168-overflow SWC-TCF point “as raw,” which flattens the Gamma-3 panel. That is a visual own-goal.

### 5.8 Discussion, limitations, future work, conclusion

The geometric explanation of the selection ordering (remote depot) is the smartest paragraph in the discussion. The “one methodological hazard, three forms” paragraph is the second.

Then the prose falls apart:

- “simulations to be run on difference temporal horizons”
- “the simulator is being built to accommodate” (present continuous: is the contribution finished?)
- a 12-line future-work sentence that concatenates ablations, multi-vehicle, sensor noise, adaptive selection, CTOP, learning-based models, matheuristics, timeout telemetry, and nearby-depot topologies with one “as well as”

Limitations, as a list of facts, are good. As a list of *consequences for the claims*, they could be sharper: “therefore we do not claim (i) a ranking of constructors at 90 days, (ii) a causal improver effect, (iii) robustness to sensor noise, (iv) validity for in-town depots, (v) validity for multi-vehicle shifts.”

There is no data-availability or code-availability statement after the acknowledgements. For this paper, that is a defect, not a style choice.

---

## 6. Writing, figures, and LNCS surface quality

**Prose.** The body, up through the integrity section, is better than typical LNCS engineering papers: specific, un-hyped, willing to use “descriptive” and “does not identify.” The frozen abstract, the NCO padding, and the last two pages drag the average down. Avoid “profound implications.” Avoid “unified baseline” until the missing parameters are on the page.

**Typography.** The LNCS class sets a 12.2 × 19.3 cm text block. The PDF is US Letter (612 × 792 pt), so the printed page is a small LNCS island in a large white frame. Compile with the class’s intended paper size (or `a4paper` consistently) before a camera-ready. `todonotes` is still in the preamble. Hyperlinks are all blue; print LNCS often wants `hidelinks`.

**Figures.**

| Figure | Verdict |
|---|---|
| Policy configuration space | Clear, slightly marketing |
| Simulation loop | Visually good, scientifically oversold (\(\varepsilon=0\) vs drawn noise) |
| Network maps | Good; depot omitted as advertised; \(N=100\) not shown |
| Constructor Pareto | Fine; annotate that x-spread is tails |
| Runtime scaling | One of the best; log y is justified |
| Fill trajectories | Best figure; keep and extend |
| Strategy bars | **Broken labels** |
| Improver delta | Fine as a diagnostic |
| Appendix Pareto | Dense but content-rich; do not include the failure outlier |
| Appendix strategy scatter | Overlapping labels (two RM-170 markers on top of each other) |
| Appendix 90-day heatmap | Useful once the sample is honestly described; log colour bar must be in the caption more loudly |
| Appendix CLS “table” | **A PNG of a table.** Unselectable, unsearchable, un-LNCS. This is not acceptable in a results appendix. Rebuild as `longtable` |

**Tables.** Booktabs, generated, footnoted. Good. Missing columns: profit, kg, kg lost. Constructor table’s “best in bold” on overflow *means* while noting that medians are tied is slightly at war with itself; bold the medians or stop bolding the tail-driven means.

**Length.** 34 LNCS pages is a journal article. If the target is a conference volume, this will be desk-reshaped. The related-work lecture and the NCO pages are the obvious cuts. The appendix PNG table is not a substitute for a real table and costs a landscape page.

---

## 7. What I checked in the codebase and logs that the paper does not tell you

These are not optional extras. They are the experimental conditions.

| Quantity | Value in the stored 30-day runs | In the paper? |
|---|---|---|
| Seed | 42 | No |
| Realisations per cell | 1 | Yes (limitations) |
| Sensing noise | \(\mu=0,\sigma^2=0\) | Partially (\(\varepsilon=0\) in a caption) |
| Constructor time limit | 60 s / day | No |
| Improver time limit | 30 s | No |
| BPC `exact_mode` | false | Implied, not named |
| ACO-HH search | 10 ants, 50 iterations | No |
| \(r_w\) (plastic) | 0.5837 €/kg | No |
| \(c_{km}\) | 1.0 €/km | No |
| \(Q\) physical | 2500 kg (Figueira), 3500 kg (Rio Maior) | No |
| `n_vehicles` | 0 (unlimited in several adapters) | Contradicted (“one vehicle”) |
| Shift duration | Loaded, not binding | No (future work pretends it is future) |
| SL \(z\) | 0.84 in YAML; linear \(n_d\) term in code | “held constant,” \(\sqrt{n}\) in the equation |
| 90-day BPC coverage | 60/60 | “Pareto subset” |
| 90-day ALNS coverage | 0/60 | Silent |
| Profit in logs | yes | not tabulated |
| kg lost in logs | yes | not tabulated |

A “reproducible simulator” paper that withholds this table is not reproducible from the PDF. It is reproducible only if the reader already has the Hydra outputs.

---

## 8. Credit, without the sugar

Credit is due.

- Building eight heterogeneous constructors, including a full BPC with ng-routes and Farkas pricing, plus SANS from a prior paper by co-authors, and running them on two real cities, is a large engineering effort. That effort is real even when the write-up overreaches.
- The three-stage interface is the correct way to think about smart-bin routing, and it is cleaner than the monolithic solvers the related work criticises.
- The integrity protocol is better than standard practice.
- The remote-depot explanation of the selection ordering is a real piece of analysis, not a restatement of a table.
- The authors repeatedly refuse causal language their design cannot support. That is rarer than it should be, and it should not be unlearned in revision — it should be applied to the 90-day paragraph and the abstract as well.
- Automated generation of tables from logs (`logic/gen/gen_paper_latex.py`) is the right instinct. Extend it until the appendix “table” is also generated TeX.

Credit is **not** due for: evaluating NCO; for a fourfold selection effect; for a Pareto-selected 90-day study; for a fully stated capacitated single-vehicle experiment; or for a formulation that matches the code of the two selection rules that define the paper’s main frontier.

---

## 9. Priority patch list (what I would demand as a referee)

**Must fix before resubmission — factual.**

1. Rewrite the abstract (or add highlights) so NCO is not an evaluated method.
2. Replace “nearly four times” with the number the tables actually imply (~2.8×), computed on one balanced slice.
3. Describe the 90-day sample as it exists in `simulation_summary_90d.csv`. Recompute the paired table on a truly Pareto-selected set, or drop the word Pareto.
4. Typeset the Service-Level rule that ran; drop \(\sqrt{n_d}\) or implement it.
5. Describe Look-Ahead as a deterministic projection, not a stochastic simulation.
6. Publish \((r_w, c_{km}, Q, K, T_{\max}, \varepsilon, \) time limits, seed, \(R)\).
7. Audit capacity: report the fraction of days each constructor exceeded physical \(Q\); do not rank infeasible tours as if they were feasible VRPP solutions.
8. Remove or rebuild the rasterised appendix table; filter degenerate runs out of appendix plots the same way they are filtered from Table 1.
9. Add a code- and data-availability statement.
10. Fix bibliography keys and the Lin 2017 Farkas citation.

**Should fix — design and framing.**

11. Add profit and kg-lost columns; say kg/km is an ex-post KPI.
12. Add a null-selection / must-collect-all ablation, even on one network.
13. Enforce or drop shift duration; do not advertise SANS as the Jorge 2022 method if \(T_{\max}\) is off.
14. Cut NCO related work to a paragraph; add IRP / TOP.
15. Repair Fig. 6 tick labels; show \(N=100\) or say why the map is 170 only.
16. Freeze constructor hyperparameters in a table.

**Would make the strong parts actually strong.**

17. Nearby-depot counterfactual (the discussion already asks for it).
18. \(R \ge 5\) replications on at least the \(N=100\) slice.
19. Frozen-tour improver rerun.
20. One industrial / current-practice baseline, in the spirit of Jorge et al. 2022.

---

## 10. Bottom line

The authors built a real testbed and ran a real factorial experiment on real cities. They noticed the right phenomenon (selection dwarfs construction on these networks, constructors live in the tails and the wall-clock), and they handled failed MILP runs with more care than is common. That is enough for a good paper.

It is not enough for *this* paper, yet. The manuscript still sells a neural, noisy, multi-vehicle framework; reports a classical, noiseless, shift-free, single-sequence experiment; misstates its own headline factor; misstates its 90-day sample; and writes down a Service-Level equation that the repository does not run. Those are not stylistic problems. They are the difference between a benchmark another group can trust and a benchmark another group will have to reverse-engineer.

Be as strict with the abstract, the 90-day paragraph, and the selection equations as the authors already were with the SWC-TCF failures. If they do that, the rest of the work deserves to be published. If they do not, a competent referee will do it for them, less kindly.

---

*End of report.*
