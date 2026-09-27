# Issue 75 / P10 handoff

## Deliverables

- `issue-75-p10-results-provenance.patch`: 338 lines, three paper-repo files:
  `paper.tex`, `Tables/results_horizon.tex` (footnote only), and new
  `Evidence/P10-provenance.md`.
- `issue-75-p10-generator-note.patch`: 11 lines, main-repo file
  `logic/gen/json/paper_latex_config.json`, horizon note string only.

Paper base: `5ae3dd7`. Apply paper patch on `paper-update/beamer-notation`;
apply the companion JSON patch in WSmart-Route so regeneration preserves the
corrected horizon footnote. No numeric cells or image assets change.
Neither shared checkout was edited; nothing committed or pushed.

Work/build copy: `~/.cache/wsr-review/codex-paper-p10`.
Baseline build copy: `~/.cache/wsr-review/codex-paper-p10-baseline`.
Build logs: `~/.cache/wsr-review/codex-paper-p10-build.log` and
`~/.cache/wsr-review/codex-paper-p10-baseline-build.log`.

## Scope

R-codex-01..06, R-kimi-01, R-gemini-01 and I-codex-02 are all represented
in the paper's new Implementation Provenance subsection and its source-file
evidence ledger. Corrected repeated Pareto-only claims in Design, Horizon,
Limitations, the appendix caption and the generated horizon-table footnote.
Qualified runtime, BPC certificate claims, HGS path exposure, CF mirroring,
ACO provenance and neural-only insulation. PSOMA's simulator global reseed
is explicit; it is not treated as an unseeded experiment. Future-only
capacity splitting does not retroactively validate archived routes.

Q8: original records pending recovery; retain/rerun decisions deferred.
Q9: 224 integrity-filtered pairs retained; 240 raw pairs remain distinguished.
No simulator reruns or newly substituted experimental outcomes.

Adjacent limitations text was reconciled with landed P9: no asserted
per-trip feasibility certificate and no unsupported claim that low-tonnage
SWC runs crashed or terminated early. This is prose only, not a changed
exclusion rule. P11 should preserve these qualifications when editing the
conclusions.

## Verification

- Both patches pass `git apply --check`, the paper patch against a pristine
  `5ae3dd7` checkout. `git diff --check` passes.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error` exits 0; no undefined
  references or citations. Output: 49 pages (base 48).
- All six regenerated tabular bodies match the base exactly, including the
  horizon table after generating its amended footnote from the copied JSON.
- Read-only archive audit rerun: 480 log/summary matches; selection,
  CF70/CF90, timing and paired-count checks retain previously reviewed values.
- Inspected edited pages 24,25,27,30,31,32,33,36,37,39,45; no new clipping or
  layout defects in these edits.
- Two pre-existing overfull boxes reproduce in both builds with identical
  widths: 64.85919pt in the §3 route notation (source line635), 2.50024pt in
  the four-bin appendix table. They are outside P10 and remain P12 work;
  this is not a claim that the whole document has zero layout warnings.

The evidence ledger records input/script SHA-256 hashes, reproducible commands,
code revisions and the limits of local `.agent/cache` artifacts. Those local
artifacts are not misrepresented as a publicly available run manifest.
