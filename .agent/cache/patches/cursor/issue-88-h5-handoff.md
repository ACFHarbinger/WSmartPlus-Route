# Issue #88 / H5 handoff (Cursor)

- **Patch:** `.agent/cache/patches/cursor/issue-88-h5-la-sl.patch`
  (25 lines, 2 hunks, `paper.tex` only)
- **Base:** `9898e2f`; stacks on Mistral's H1 (`issue-84-h1-hector-notation.patch`)
- **Work copy:** `~/.cache/wsr-review/cursor-paper-h5`
- **Range:** SL "Two properties" paragraph (~1007) and the Look-Ahead grid
  sentence (ends before `\label{sec:constructors}`)
- **Nothing committed or pushed.** Shared submodule not edited.

Hector Point 3, both sentences:

1. **SL linear-in-ν.** Dropped the "positively correlated increments"
   justification. The linear buffer is now named as configured conservatism
   relative to an i.i.d.\ $\sqrt{\nu}$ aggregation; Empirical and Gamma-3
   are independent day to day, so it is not a serial-correlation correction.
2. **Look-Ahead grid.** Replaced "no threshold variant, so fewer
   configurations" with: Look-Ahead is one variant (no threshold or depth
   split); every variant is crossed equally.

**Verification:** `git apply --check` passes on pristine `9898e2f` and on
`9898e2f`+H1. `latexmk -pdf` on the H1+H5 tree exits **0**, **0 undefined
references/citations**, **53 pages** (same as the H1-only baseline). The two
pre-existing overfull boxes are in Hector's front half / appendix, not in
this diff. Both corrected sentences are in the rendered PDF.

D5 (kg-lost column) and D6 (N=100/170) are out of scope for #88.
