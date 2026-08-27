# Brief — Codex / Chat (co-team-lead, reviewer, and final editor of the paper)

**Branch:** `feat/paper-results-and-website` · **Bus:** `.agent/bus/2026-08-25.md`

You are the reviewer, and on the paper specifically you are the **final editor
with rewrite authority**. You go last. If the Methodology or Results sections
need a full rewrite rather than corrections, do the rewrite — you do not need to
route it back through me, and you should not treat my draft as a baseline to be
preserved. Everyone else's edits land before yours; yours are the ones that
ship.

Post every finding to the bus under `### Codex — 2026-08-25 (topic)`.

## R1 — Adversarially re-derive the numbers (highest priority, do first)

I posted a set of quantitative claims in today's bus entry. Re-derive each one
yourself from `docs/private/global/simulation/simulation_summary.csv` (480 rows,
30d) and `simulation_summary_90d.csv` (174 rows, 90d). **Do not read my code
to do it — write your own.** Two independent derivations that agree are worth
something; one derivation checked twice is not.

Specifically confirm or refute:

1. The 30-day grid is balanced at 60 runs per constructor with every
   constructor x strategy cell filled; the 90-day grid is not (ALNS = 0 runs).
2. `SWC-TCF / LA / Gamma-3 / N=350` has `days=15` and `kg≈37,065` where the
   other seven constructors in that cell have `days∈[20,27]` and `kg≈70,300`
   — i.e. it is a truncated run, not a policy result.
3. The per-constructor and per-strategy tables in my **third** bus entry
   ("post-exclusion numbers") — not the kickoff entry, whose tables are stale
   and superseded. Check the digits.
4. CLS beats Fast-TSP on 212 of 238 paired configurations that are not ties,
   mean delta +0.68 kg/km — and, more interestingly, that **all 26 losses are
   at N=350 and belong only to HGS, PSOMA and SWC-TCF**. That conditional is
   the result I intend to publish; attack it.
5. Efficiency falls monotonically across LM-CF90 → LA → LM-CF70 → SL-SL1 →
   SL-SL2, but overflow risk does *not* fall with it. Confirm both halves; I
   originally claimed a joint monotone ordering and it is wrong.
6. That dropping whole scenario cells (rather than only the degenerate rows)
   is the right call, and that `n` really is uniform at 57 per constructor
   afterwards.

**Also look for outliers I did not find.** I found two by inspecting the cells
I happened to aggregate. Sweep systematically: any run whose `days` is far
below its horizon, any `km` more than ~2x its cell median, any `kg` far below
its cell median. Truncated runs are the failure mode — find all of them.

## R2 — Review the paper prose against the data

Once I have pushed the rewritten Methodology and Results in
`assets/papers/Simulation_Framework_for_the_MPVRP_with_Profits_in_Smart_Waste_Collection/paper.tex`:

- Every numeric claim must trace to a CSV row or an aggregate over rows.
  Flag anything that reads as plausible but is not derivable.
- Every algorithm named in Methodology must exist in `logic/src/policies/`.
  The mapping is in `.agent/reports/claude/policy_name_map.md` once I write
  it — verify it rather than trusting it.
- Check that no claim generalises across the Gamma-3 / Empirical boundary
  without saying so. They are very different load regimes.
- Check the 90-day claims are scoped to the constructors actually run at 90
  days.

## R3 — LaTeX correctness

Pre-existing bugs to confirm fixed, plus anything new:

- Two `\label`s inside one float (`fig:g31_logtime`/`fig:g31_ncols` and the
  e93 pair) — both resolve to the same number. Must become separate floats or
  a single label.
- Bare `\ref{}` with no "Fig."/"Table" prefix throughout Results.
- The Methodology sentence that dies mid-clause: *"which has the number of
  days "*.
- `paper.log` for undefined references and overfull boxes after each rebuild.

## R4 — Review the other agents' diffs

Agy (design) and Opencode (interactive/3D) are both editing `docs/website/`.
Watch specifically for: the two of them redefining the same CSS custom
properties, `App.tsx`/routing conflicts, any dependency added that is not in
`docs/website/package.json`, and bundle-size regressions from 3D libraries.

## Ground rules

- Update `docs/moon/CHANGELOG.md` in the same commit as any fix you apply.
- Never push to `main` (`AGENTS.md` §5.3).
- If you disagree with me, say so on the bus with the evidence. Being the
  co-lead means overruling the lead when the data says so.


## R5 — Final editorial pass (do this last, after Agy and Opencode have had theirs)

Everyone is now editing the paper, not only me. Agy and Opencode have been asked
for their opinions and edits on
`assets/papers/Simulation_Framework_for_the_MPVRP_with_Profits_in_Smart_Waste_Collection/paper.tex`
in addition to the website. Their passes land before yours; yours closes the
file.

Your remit on that final pass:

- **Rewrite freely.** Correcting my prose sentence by sentence is not the
  assignment. If a section reads better restructured, restructure it. The one
  thing that must survive unchanged is the constraint that every number traces
  to the CSVs through `logic/gen/gen_paper_latex.py` — never hand-edit a
  generated table under `Tables/`; change the generator and re-run it.
- **Reconcile the voices.** Three or four agents writing into one manuscript
  will not sound like one author. Make it sound like one.
- **Adjudicate conflicts.** Where Agy or Opencode disagree with my reading of a
  result, you decide, and record the decision on the bus with the reason.
- **Rebuild before you call it done**: `latexmk -C && latexmk -pdf` from the
  paper directory, run to convergence. It currently builds at 24 pages with no
  undefined references and two sub-9pt overfull boxes. Do not leave it worse.

Sections I did not touch and which are still unwritten, in case you want them:
the Conclusion is still the placeholder sentence "Should write some text here",
and Related Work's Multi-Period VRPP subsection is still commented out.
