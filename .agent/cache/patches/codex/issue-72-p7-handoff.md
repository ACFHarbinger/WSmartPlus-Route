# Issue 72 / P7 handoff

Patch: `issue-72-p7-improvers.patch` (107 lines, 3 hunks, paper.tex only).
Base: `2fc97d9ed9abbb1897ddc25fa7bb92aca215b359`.
Work/build copy: `/home/pkhunter/.cache/wsr-review/codex-paper-p7`.
Build log: `/home/pkhunter/.cache/wsr-review/codex-paper-p7-build.log`.

Implemented A-codex-01..03, I-codex-01, Muse's exception fallback and Q9.
- CLS ordered neighborhoods, 1,000-iteration cap, sampled higher-order moves,
  no exhaustive local-optimum guarantee, exception returns original tour.
- Fixed-service distance reduction increases reported profit when transport
  cost is positive and no per-vehicle charge is counted.
- Fast-TSP per-route assignment, integer-distance exactness qualification,
  archived 30-second per-route budget, stochastic search/unused seed.
- Incorporated current wrapper safeguards (00d7301ea): range-aware scaling
  and return-input-order on library-call failure, explicitly future runs only.
- kg/km is the reported efficiency metric; daily profit is the objective.
- Names the tonnage-shortfall filter, all-constructor cell exclusion and
  matched-improver restriction yielding 224 pairs rather than 240 raw pairs.

Validation: cached-index apply check against pristine base passed; diff check
passed; latexmk exited 0; zero undefined references/citations and zero overfull
hboxes. PDF has 38 pages (base 37). Visually inspected pages 13,14,18,23.
Recomputed raw/filtered pair counts with gen_paper_latex.improver_pairs.
No tables changed, source edits, simulator runs, commits or pushes.

Claude applies this patch to the paper branch and rebuilds with other lanes.
Do not include the scratch paper.pdf in the patch. P10 remains after P7/P9.
