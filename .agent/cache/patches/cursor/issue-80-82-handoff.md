# Cursor — code cleanup (#80 B-cursor-03, #82 M-cursor-02/03)

**Date:** 2026-09-28
**Worktree:** `~/.cache/wsr-review/cursor-code` (shared checkout not edited; `assets/papers/**` untouched)
**Python:** `~/.cache/wsr-main-venv/bin/python`

## Revision (#82, Codex §10.2 #7)

- **Base:** `f1975cdf9` (clean-up batch 1; Cursor #80 already on `main`)
- **Patch:** `.agent/cache/patches/cursor/issue-82-reward-selector-parity.patch`
- **SHA-256:** `2e148ec6f340401c02a36ee4472d22197692146637ec53d0bd100e20235a3edb`
- `git apply --check` passes on `f1975cdf9` and on current `main` (`a323312b3`).

**What changed vs the first #82 patch:** `_sim_profit` and `_policy_scaled_profit`
are gone. Profit tests call production `Bins.collect` (Rio Maior plastic,
`area="riomaior"`, `waste_type="plastic"`) and
`BaseRoutingPolicy._load_area_params` (via a stub that only implements
`_run_solver`). Four full 47.5 kg bins + 96.167 km still yield 14.736 from
`collect`, and that equals `fill_percent_sum * revenue_scaled - km * C`.
Training still goes through `VRPP.get_costs`. A regression in either
production formula now fails these tests.

Selector parity (M-cursor-03) is unchanged: it already called the real scalar
and vectorized selectors.

**Verification:** 10 passed in 0.12s (5 profit + 5 selector); ruff clean. No
production-code edit; no `test_sim`; no behaviour change.

## First round (historical)

1. **`.agent/cache/patches/cursor/issue-80-bmc-oi-docstrings.patch`** — landed as
   `5c07acf08`.
2. First `#82` patch used test-local copies of the profit formulas. Codex
   rejected that (§10.2 #7); this file's revision section supersedes it.
