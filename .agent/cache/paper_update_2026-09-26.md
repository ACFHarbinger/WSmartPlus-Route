# Shared report: paper update of 2026-09-26 (notation from the beamer, algorithms from the code)

**Paper:** `assets/papers/Simulation-Framework-for-the-MPVRP-with-Profits-in-Smart-Waste-Collection/`.
The submodule is checked out at upstream `origin/main` = `399d22c` (2026-09-17). The superproject
still pins `0cd8eeb`. Upstream is 5 commits ahead: Overleaf updates, plus Hector's
`paper_versaoHector1.tex` and `rascunho.tex`.
**Target file:** `paper.tex` (owner question Q1).
**Inputs:**
- `temp/mpvrpp_beamer.pdf`: Bonilla-Londoño, Fernandes, Jorge, Lopes, Santos, Ramos, "Do SWCRP ao
  MPVRPP", Sept 2026, 35 slides, in Portuguese. A text dump is at `~/.cache/wsr-review/beamer.txt`.
  Formulas are garbled in the dump, so read the PDF pages for every equation.
- `paper_versaoHector1.tex` §3: a formulation draft in a *different* notation, to be reconciled.
- The codebase on `main` (`8ca4a4703` or later).
- The logic-review report `.agent/cache/logic_review_2026-09-26.md`: P rows for the algorithm
  descriptions, B rows for results that may change.

**Protocol:** `.agent/tasks/paper-update-2026-09-26.md`

**Phase 1 (now): proposals only.**
- Every change goes into this report as a row with the exact old text, the new text (LaTeX), and
  its source (beamer slide, code `file:line`, paper, log file).
- Nobody edits `paper.tex` in this phase.
- After the owner rules, each accepted group becomes a GitHub issue and is delegated.

## 0. Owner rules for this update

1. The notation follows the beamer (Ramos et al. 2018 + the MPVRPP extension), **except** that the
   beamer's `S_i` stays `w_i` in the paper (the waste at bin `i`).
2. The algorithm descriptions (selection, construction, improvement, the Neural Agent/AM) are
   brought in line with what the code on `main` actually does.
3. Other improvements are welcome. Each one is a row with its justification.

## 1. Notation: proposed canonical table (Claude draft; agents amend in §1.b, the owner rules in §6)

The index convention follows the beamer: the period goes in a **superscript** `t ∈ T = {1,…,τ}`.
It replaces the paper's subscript `,d` and `d = 1..D`.

| Beamer symbol | Proposed paper symbol | Meaning (unit) | Current paper symbol(s) | Notes / collisions |
|---|---|---|---|---|
| `I = {0,1,…,n+1}` | same | nodes: depot 0, bins 1..n, depot copy n+1 | `𝒱`, `v_0` (paper); `V`, `B`, `N` (Hector) | `I \ {0,n+1}` = the bins. Suggest a short name for it, e.g. `I_b` (owner Q4). |
| `T = {1,…,τ}`, `t` | same | periods (days) | `d = 1..D` | Frees `d` for the distance `d_ij`. |
| `C` | same | transport cost per km (€/km) | `c_km` | The paper's `C_i` (bin capacity) must become `E_i`. |
| `R` | same | revenue per kg (€/kg) | `r_w` | |
| `Ω` | same | cost per vehicle used (€) | none | Check whether the simulator or policies charge a per-vehicle cost. If not, say so (Ω = 0 in the experiments). |
| `Q` | same | vehicle capacity (kg) | `Q` | |
| `K`, `K^max`, `k^t` | same | fleet available / vehicles used in `t` | `K_d` (the route count) | The paper says the route count is unbounded (multi-trip). Reconcile with `k^t ≤ K^max` against the code (`sim.n_vehicles`). |
| `B` | same | waste density (kg/m³) | none | Check what the code uses (`constants`, the bin data). |
| `d_ij` | same | distance between i and j (km) | `dist(·,·)`, `L_ij` (Hector) | Hector's symmetrisation remark is worth keeping if the code symmetrises. Verify. |
| `S_i` | **`w_i`** (owner rule) | waste in bin i read by the sensor (kg) | `w_{i,d}` | Multi-period: see `s_i^t` in the variables below. |
| `a_i`, `a_i^t` | same | expected / actual accumulation in i (kg per period) | `δ_{i,d} ∼ P_i` | The paper's `δ` must be renamed, because the beamer's `δ` is the service level. The simulator's accumulation is stochastic: keep `a_i^t ∼ P_i` for the simulator, and the expected value in the MILP. |
| `E_i` | same | bin capacity (kg) | `C_i` | |
| `H`, `O` | same | the set of bins that will overflow, and its size | none | |
| `δ` | same | share of bins allowed to overflow (%) | none (the paper's `δ` is accumulation) | |
| `ψ` | same | maximum overflow threshold (% of `E_i`) | none | Connects to the `last_minute` threshold (70 %) and the service-level selectors. Lane F maps it. |
| `M` | same | minimum fill rule of the limited approach | none | Collides with the paper's mandatory set `𝓜_d`. Proposal: keep the calligraphic `𝓜^t` for the mandatory set (owner Q5). |
| `Ū_i` | same | big-M bound `ψE_i + max_t a_i^t` | `U_i` (Hector: `C_i + max a`) | Hector's bound uses `ψ = 1`. Use the beamer's form. |
| `ρ` | same | residual value of the end-of-horizon stock | `r_0` (Hector) | |
| `x_ij^t, y_ij^t, g_i^t, k^t` (slide 22) | same | edge used / flow / bin visited / vehicles used | `𝒜_{d,k}` (a route list) | |
| `s_i^t` | **`w_i^t`** (proposal, owner Q2) | waste in i at the start of t, before collection (kg) | `w_{i,d}` | Follows from the `S_i → w_i` rule: the state and its initial condition share a letter, `w_i^1 = w_i`. The alternative is to keep `s_i^t` and use `w_i` only for the sensor reading. |
| `q_i^t` | same | waste collected from i in t (kg) | none | |
| `w_i^t` (overflow binary) | **`o_i^t`** (proposal, owner Q2) | 1 if i overflows at the end of t | none (Hector: `o_{i,d}`) | Must change, because `w` is taken. `o` matches Hector's draft. |
| none | `ℓ_i^t` (proposal, owner Q3) | waste lost to overflow (kg) | none (Hector: `ℓ_{i,d}`) | The beamer says "no lost waste" (a limitation), but the simulator reports kg lost, so the paper needs a symbol if it reports it. |
| none | route notation | the ordered visits of route k in t | `𝒜_{d,k}`, `a_{t,k}`, `T_{d,k}` | All three collide (`a` = accumulation, `t` = period, `T` = the period set). Lane D proposes a replacement. |

### 1.b Amendments to the notation table (agents append here, one row each, with justification)

| ID | Symbol | Proposal | Why | Agent |
|---|---|---|---|---|

## 2. Formulation and problem definition (§2 "Problem Definition" of the paper)

What the beamer adds, and where it goes:

- the context and the Valorsul evidence (slides 3–4);
- the assumptions (slide 5);
- the notation (slides 6, 8, 9 and 22);
- the two-commodity flow (slides 6 and 7);
- model 1 (slides 11–12), model 2 (slides 15–16), the smarter trigger (slide 19), and model 4
  = MPVRPP (slides 21–25), eqs. (1)–(44);
- the consistency notes on (16) (`≥`) and on (18) (`2g_j`) (slide 17);
- δ vs ψ (slide 18);
- the terminal effect and `ρ` (slide 26);
- the decision flow (slide 27);
- the worked 4-bin example (slides 28–30);
- the model hierarchy (slide 31);
- "from the model to the policy: the three stages" (slide 32);
- the traceability table (slide 33);
- the limitations (slide 34);
- the references (slide 35).

The last item maps (40) to mandatory selection, `g_i^t` to construction, and (29)–(37) to
improvement. It is the bridge to §4 Methodology.

| ID | Paper location (line in `paper.tex` @399d22c) | Old text (short quote) | Proposed new text / structure | Source | Agent | Status |
|---|---|---|---|---|---|---|

## 3. Algorithm descriptions (Methodology: selection, constructors, improvers, NA/AM, protocol)

| ID | Paper location | What the paper says | What the code does (`file:line`) | Proposed text | Agent | Status |
|---|---|---|---|---|---|---|

## 4. Results that the code changes put at risk

This covers bugs found or fixed since the reported runs. Examples:
- the cf90 → cf70 selector mirroring (B-claude-01 on 2026-09-25) means the psoma/sans cf90 rows
  may duplicate cf70;
- the DS-15 time definition;
- the BPC exactness fixes;
- the HGS split fix;
- the open B rows of the 2026-09-26 logic review.

| ID | Paper table/figure/claim | Affected by (bug ID / commit) | Evidence (log file, rerun) | Proposed action (rerun / footnote / drop) | Agent | Status |
|---|---|---|---|---|---|---|

## 5. Other improvements

This covers structure, clarity, related work, bibliography, figures, consistency, LaTeX hygiene and
the conclusions.

| ID | Location | Proposal | Why | Agent | Status |
|---|---|---|---|---|---|

## 6. Owner questions and rulings

- **Q1.** Target file: `paper.tex` (with Hector's §3 draft as input to reconcile and re-notate)? Or
  should `paper_versaoHector1.tex` become the base?
- **Q2.** Should the state variable be `w_i^t` (which requires the overflow binary to become
  `o_i^t`), or should it stay `s_i^t`, with `w_i` used only for the sensor reading?
- **Q3.** Should the paper introduce lost waste `ℓ_i^t`, given that the simulator reports kg lost
  and the beamer model omits it?
- **Q4.** Should the bin set get a short name (`I_b`, or `I^*`)?
- **Q5.** Should the mandatory set be `𝓜^t` (calligraphic) next to the beamer's fill rule `M`?
- **Q6.** Where should model 4 (the full MPVRPP MILP, eqs. 22–44) go: in the body, or in an
  appendix with only the state transition and service-level constraints in the body? The LNCS page
  limit applies.
- **Q7.** Should the worked 4-bin example (slides 28–30) go in the paper, or stay in the beamer?

## 7. Per-agent sections

Each agent adds a section headed `## 7.<lane>. <Agent> — lane <X> — 2026-09-26`. It lists:

- which pages of the beamer the agent read;
- which paper lines it read;
- which code it checked;
- the IDs it filed;
- its disagreements.

## 8. Consolidation → GitHub issues (Claude + Codex)
