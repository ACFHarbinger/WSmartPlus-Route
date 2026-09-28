# Cursor — code cleanup (#80 B-cursor-03, #82 M-cursor-02/03)

**Date:** 2026-09-28
**Base:** `6d500ed02` on `main`
**Worktree:** `~/.cache/wsr-review/cursor-code` (shared checkout not edited; `assets/papers/**` untouched)
**Python:** `~/.cache/wsr-main-venv/bin/python`

## Patches

1. **`.agent/cache/patches/cursor/issue-80-bmc-oi-docstrings.patch`** (87 lines)
   - **B-cursor-03.** BMC and OI module examples claimed `accept(100, 98) → True`.
     Under maximisation that is a worsening: BMC at `T ≤ 1e-9` rejects it; OI
     always rejects it. Examples are now an improving move `10 → 12`. The
     examples also state that **BMC accepts ties** and **OI rejects ties**.
   - Regression tests: `logic/test/unit/policies/acceptance_criteria/test_bmc_oi_docstring_examples.py`
     (docstring content + accept behaviour). Would fail on the old `98.0` examples.

2. **`.agent/cache/patches/cursor/issue-82-reward-selector-parity.patch`** (225 lines)
   - **M-cursor-02.** Parity test, not a merge. `VRPP.get_costs`, simulator
     `Bins.collect`, and `base_routing_policy.revenue_scaled` stay as they are.
     Merging would touch Gemini (`envs/`), Grok (`bins/`), and Kimi
     (`base_routing_policy.py`) and would rescale RL rewards (behaviour change).
     Test: `logic/test/unit/policies/test_reward_profit_parity.py`.
   - **M-cursor-03.** Do not merge the scalar and vectorized selectors. Parity
     test: `logic/test/unit/policies/mandatory_selection/test_scalar_vector_selector_parity.py`
     — scalar 1-based IDs equal the vectorized mask (depot column prepended,
     percent fills mapped to fractions) for last-minute, service-level, and
     look-ahead, including the paper LA witness and an empty-seed day.

Apply 80 then 82 (independent; either order works). Both `git apply --check` on
pristine `6d500ed02`; 82 also checks after 80.

## Verification

```
14 passed  in 0.13s  (the three new test modules)
10 passed             (existing selector unit tests)
ruff check            clean
compileall            exit 0 on acceptance_criteria, mandatory_selection, vector/selection
```

No `test_sim`. No behaviour change: production code for #82 is untouched; #80
is docstring-only.

## Remaining drift (not this lane)

- `SelectionContext.horizon_days` still defaults to **3**; vectorized
  `ServiceLevelSelector` defaults to **1**. The parity test passes the same
  horizon explicitly. Interfaces are outside this lane.
- Training `COST_KM = REVENUE_KG = 1.0` on fraction waste is a normalised
  proxy; simulator profit is `(fill/100)*volume*density*R - km*C` with plastic
  `R = 0.5837`. The conversion identity is in the #82 test.

## Not done (other lanes / out of scope)

- M-cursor-01 SANS part (Qwen deferred).
- B-cursor-02 (BMC temperature / ALNS `start_temp: 100`) — not in this brief.
- D5/D6 paper items — closed for H5; not this round.
