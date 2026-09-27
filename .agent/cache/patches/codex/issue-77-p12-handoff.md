# Issue 77 / P12 handoff

Paper base: `3ea8cc02f9c8cf79bf6477f08f36325f72cdfd1b`.
Full reviewed range: `399d22c..3ea8cc0`, all 13 changed paths.
Findings: `.agent/cache/paper_update_2026-09-26.md`, section 8.6 (14 grouped findings).

Apply `issue-77-p12-full-review.patch` in the paper repository. It includes
text, bibliography, four generated-table text changes, the regenerated
simulation-loop image, and three new evidence/reproducer files. The binary
image patch is included; `paper.pdf` is deliberately excluded, to be rebuilt.
Apply `issue-77-p12-generator.patch` in WSmart-Route so table headers,
exclusion footnote and diagram wording survive regeneration. Both patches
pass `git apply --check` against their corresponding current checkouts.

Major corrections: directed residual flow, forbidden depot arcs, zero-demand
connectivity, first-period mass bound; appendix terminal-stock calculation
and rho=R optimum; S^t renamed to I(R^t); n/nu, T/temperature and count-header
collisions; post-arrival state convention; algorithm and causal-claim fidelity;
remaining provenance overclaims; malformed Mes2014 entry and Mes2012 metadata.

Validation completed in `/home/pkhunter/.cache/wsr-review/codex-paper-p12`:
- latexmk exits 0, 51 pages (reviewed base50), no undefined citations/references,
  no duplicate labels, no overfull boxes or oversized floats, zero BibTeX warnings.
- Full document rendered for layout inspection; corrected appendix pages,
  routing display and regenerated diagram inspected at readable resolution.
- Independent exact integer enumeration of2,688 visit-set trajectories;
  rho=0 optimum60.532, rho=R optimum79.18. Algebraic model boundary checks pass.
- All480 available30-day raw summaries reproduce; all six numerical table
  bodies unchanged, including224 improver pairs and165 horizon pairs.
- Patch whitespace and application checks pass. No simulator runs.

Remaining distinctions are explicit rather than hidden: the linear reference
force threshold is strict whereas SWC's implemented test is inclusive; owner
Q14's overflow definition itself is inclusive everywhere. New connectivity
cuts repair the reference formulation, not the historical solver. Example
matrix retained but disclosed as a modification of slide28. Historical
manifests/datasets/90-day logs remain pending recovery; reruns deferred by Q8.
Page budget remains unknown. Owner reviews before Hector; nothing merged.

No shared paper edits, commits or pushes by Codex.
