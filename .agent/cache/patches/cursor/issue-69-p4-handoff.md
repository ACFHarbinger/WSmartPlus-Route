# Issue #69 / P4 handoff (Cursor)

- **Patch:** `.agent/cache/patches/cursor/issue-69-p4-mandatory-selection.patch`
  (191 lines, 1 hunk, `paper.tex` only)
- **Base:** `5ae3dd7`
- **Work copy:** `~/.cache/wsr-review/cursor-paper-p4`
- **Range:** `\subsection{Mandatory Selection Strategies}` through the
  Look-Ahead paragraph (ends before `\label{sec:constructors}`)
- **Nothing committed or pushed.** Shared submodule left at `5ae3dd7`.

Implements A-cursor-01..04 against the landed §3 labels:

- `\eqref{eq:force}` (not a hardcoded (40); P3's force-visit)
- `\mathcal{M}^t`, `\psi` kept distinct from LM's `\mathrm{CF}`
- no `\delta` / no `H\le n\delta` as something the selectors do
- percent vs ratio units; SL equation in percentage points
- Look-Ahead seed + bundle; empty seed ⇒ empty `\mathcal{M}^t`

**Verification:** `git apply --check` on a pristine `5ae3dd7` clone passes.
`latexmk -pdf` exits **0**, **0 undefined references/citations**, **48 pages**
(unchanged from wave 2).
