# Issue 89 — H6 deliverables and final-round review

2026-09-28, Codex. Paper base `9898e2f`. No shared paper edits, simulations,
commits to the main repository, or pushes. Scratch clone:
`~/.cache/wsr-review/codex-paper-h6`.

## Apply order

1. **Paper:** `issue-89-h1-h5-integrated.patch` on `9898e2f`. This is an
   alternative to applying the five agents' individual patches, NOT an additional
   patch after them. It records the reviewed integration: H1 first, H2's five
   overlapping prose blocks chosen, then H3/H4/H5. H6 subsequently repairs the
   density-symbol and overflow-timing regressions in those overlapping blocks.
2. **Paper:** `issue-89-h6-paper.patch` on that integration. Includes six generated
   tables, two changed graphics, and the H6 prose/appendix changes. The PDF is
   intentionally excluded; rebuild it.
3. **Root:** Grok's `issue-85-h2-figure-generator.patch`, then
   `issue-89-h6-generator.patch`. The latter changes the generator, JSON config,
   and Jinja template. Application of this stack was checked in an isolated tree.
4. **Review proposal, not included in the tested H6 text:**
   `issue-89-front-review-proposal.patch` applies after H6. It corrects two false
   front-half statements without changing Hector's formulation or proofs.
   Claude/Hector should incorporate these corrections before declaring the
   whole paper ready; see findings below.

All patch paths above are under `.agent/cache/patches/codex/`, except Grok's.
The integrated patch has three inherited trailing-whitespace warnings; H6's
working diff passes `git diff --check`.

## H6 changes

- Added cumulative lost kilograms to all six results tables: per-run means on
  exactly the existing balanced slices, paired means at 30/90 days, the paired
  improver difference, and raw totals in the excluded-run table. Kept overflow
  events distinct from discarded mass. Config/template changes keep tables
  within the text width and use LM-CF70/LM-CF90 consistently.
- Recorded each constructor's archived 60-second configured budget, PG-CLNS's
  30-second inner limits, CPU/wall-clock/nested/retry caveats, one algorithm seed
  setting per configuration, and missing BPC fallback-frequency telemetry.
- Qualified the 4.5-fold runtime comparison and overflow interpretation. The
  objective has no direct overflow incentive; construction still matters
  indirectly (ALNS 5.9 versus HGS 19.9 mean overflow events).
- Accounted for all nine missing horizon pairs and named the six constructors
  with approximately threefold overflow growth; SWC-TCF differs, ALNS absent.
- Moved three raw granular graphics to the appendix, with raw-data caveats;
  retained their labels. Removed the appendix PNG table from the document,
  preserving its source asset. The existing raw 90-day heatmap remains caveated.
- Softened Conclusion language and kept three future priorities: independent
  seed repetitions, fleet/shift limits, and a depot within the service area.
  H4 supplies the removal of driver readings.
- Repaired H1/H2 leftovers: density no longer uses the bins-set symbol B,
  constructor pheromone is consistently tau, remaining instance n labels become
  N, the model's day-d overflow test reads closing w_(i,d+1), and American
  spelling is used in the remaining prose occurrences.

## Numeric evidence

Inputs: `docs/private/global/simulation/simulation_summary.csv` (480 rows) and
`simulation_summary_90d.csv` (174). Existing centralized exclusions/balancing
are unchanged. `kg_lost` is nonmissing on all retained rows.

- Constructor table: 456 runs, 57 each. Lost kg means: ACO-HH 60.5, ALNS 30.3,
  BPC 44.6, HGS 100.4, PG-CLNS 30.5, PSOMA 31.2, SANS 46.3, SWC-TCF 79.1.
- Selection: 80 runs each. LA 27.2, LM-CF70 18.9, LM-CF90 81.1, SL1 9.8,
  SL2 1.2 lost kg.
- Improvers: 224 pairs; CLS 52.9 versus FTSP 54.4 lost kg, difference -1.5;
  23 CLS wins among 50 non-ties, 174 ties. No nonzero differences fall inside
  the zero-tie tolerance for any reported improver metric.
- All 174 raw 90-day configurations have raw 30-day counterparts. Three are
  dropped with the 90-day anomalous cell (ACO-HH/BPC/SWC-TCF, FF350 Gamma LA
  FTSP). Six others lose their 30-day counterpart: BPC/PG-CLNS in FF350 Gamma
  LA CLS, and ACO-HH/BPC/PG-CLNS/PSOMA in FF350 Gamma SL2 FTSP. Remaining:165.
- Horizon pairs: ACO-HH28, BPC57, HGS12, PG-CLNS40, PSOMA11, SANS12, SWC-TCF5.
- All 36 `assets/output/30days/**/hydra/pruned_config.yaml` files have simulation
  seed42 and constructor time_limit60, including the SWC gurobi list and SANS
  `new` mapping. PG inner ACO/LNS values are30. These prove configured settings,
  not effective execution limits or historical corrected-code behavior.

## Final review findings (not a blanket approval)

1. **MEDIUM — abstract:** “no constructor has a reason to serve a full but
   remote bin” does not follow from absence of an overflow penalty. A profitable
   visit can be chosen; mandatory selection and SWC's own threshold can force
   it. The separate front-review proposal supplies narrower wording.
2. **MEDIUM — Model and simulator paragraph:** “same state evolution” and
   agreement whenever no bin overflows ignore collect-then-deposit versus
   deposit-then-collect. A nonoverflowing bin can yield different collected
   mass and closing stock. The separate proposal states both ordering and the
   strict/inclusive overflow difference.
3. **MEDIUM — H2/D6 evidence reconciliation:** the new text contradicts the
   owner's exact closest100/farthest170 account with 54/100 and167/170 matches.
   Grok owns that matrix/index analysis. Its exact ranking and sampled-triangle
   claims need an attached reproducible calculation (matrix version, coordinate
   mapping, sample seed/tolerance), and reconciliation with the owner before
   final acceptance. H6 has not independently certified those new measurements.
4. **Scope limitation:** this pass integrates/reviews Round H against the
   preceding review history from399d22c. It does not establish that every
   literature theorem or historical execution provenance is independently
   verified. In particular, do not interpret a successful LaTeX build as proof
   of the new Spinelli reduction or of the archived runtime semantics.

The old visit-set S^t appendix example is absent under D1(a); no S^t remains
in paper.tex. Reintroducing that old example would require adapting its state
order as well as its symbols. Hector's model, delta, and proofs are preserved.

## Validation

- All six tables regenerated from archived CSVs; three graphics regenerated
  (runtime figure byte-identical, policy-space/strategy figures changed).
- Paper patch stack applies from9898e2f; front-review proposal passes apply-check
  after H6. Root generator patch applies after Grok H2 generator patch.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex`: exit0,
  53pages, no undefined references/citations. Only two inherited overfull boxes
  (21.47pt assumption A4 and2.72pt Properties appendix), no new ones.
- Visually inspected rendered table pages31,33,36,38: all six tables readable
  and within margins. No simulator or full Python test suite was run: changes
  are report generation and manuscript text, not policy/state implementation.
