# Qwen Revision Handoff: Issue #80 and #82

## Summary

This patch addresses the three revision items requested by Codex:
1. **PG-CLNS repair capacity check** (Issue #80)
2. **Parity tests for PG-CLNS operators** (Issue #80)
3. **HMLNS temperature calibration evidence** (Issue #82)

## Changes

### 1. Capacity Check Fix (Issue #80)

**Problem:** The shared `greedy_profit_insertion` and `regret_*_profit_insertion` operators would accept nodes with demand exceeding vehicle capacity when creating new routes (seed route path).

**Root Cause:** The `get_seed_profit()` function did not check if `node_waste > capacity` before calculating profit. Similarly, the mandatory node fallback path did not check capacity.

**Fix:** Added capacity checks in three locations:
- `greedy.py:get_seed_profit()` - line ~295
- `greedy.py:mandatory fallback` - line ~425
- `regret.py:get_seed_profit_regret()` - line ~490
- `regret.py:mandatory fallback` (2 locations) - lines ~215, ~565

**Behavior Change:** Nodes with demand > capacity are now rejected even if they are mandatory or would otherwise be profitable. This matches the behavior of the old PG-CLNS operators.

### 2. Parity Tests (Issue #80)

**File:** `logic/test/unit/policies/test_pg_clns_operator_parity.py`

**Coverage:**
- Random removal determinism (same seed → same result)
- Cluster removal behavior (removes at least 1 node)
- Worst removal randomization (p=3.0 produces different results with different seeds)
- Capacity checks (greedy and regret reject over-capacity nodes)
- Mandatory node handling (mandatory nodes inserted even if unprofitable, but rejected if over-capacity)
- Directed distance / profit insertion (revenue - cost calculation)
- Regret insertion capacity checks

**Test Count:** 11 tests, all passing

### 3. HMLNS Temperature Calibration Evidence (Issue #82)

**File:** `logic/test/unit/policies/test_hmlns_temperature_calibration.py`

**Problem:** The old HMLNS ALNS copy had a bug where temperature calibration was skipped when `best_profit <= 0`, leaving `start_temp` at its default value.

**Fix in Canonical Version:** The canonical ALNS uses `scale = abs(best_profit) if abs(best_profit) > 1e-9 else 1.0`, ensuring calibration always proceeds with a reasonable scale.

**Test Coverage:**
- Calibration with positive profit (uses profit magnitude)
- Calibration with zero profit (uses fallback scale=1.0)
- Calibration with negative profit (uses absolute value)
- Calibration with tiny profit < 1e-9 (uses fallback scale=1.0)
- Demonstration of old vs new behavior

**Test Count:** 6 tests, all passing

## Verification

All tests pass:
```bash
pytest logic/test/unit/policies/test_pg_clns_operator_parity.py -v  # 11 passed
pytest logic/test/unit/policies/test_hmlns_temperature_calibration.py -v  # 6 passed
```

Compile check:
```bash
python -m compileall logic/src/policies/helpers/operators/recreate_repair/  # OK
```

## Patch Contents

- `logic/src/policies/helpers/operators/recreate_repair/greedy.py` - capacity checks
- `logic/src/policies/helpers/operators/recreate_repair/regret.py` - capacity checks
- `logic/test/unit/policies/test_pg_clns_operator_parity.py` - new file
- `logic/test/unit/policies/test_hmlns_temperature_calibration.py` - new file

**Total:** 421 lines (62 lines fixes + 359 lines tests)

## Integration Notes

This patch should be applied after the base cleanup patches:
1. `issue-80-m-qwen-01-pg-clns-operators.patch` (PG-CLNS operator switch)
2. `issue-82-m-qwen-03-hmlns-alns.patch` (HMLNS ALNS copy removal)

The capacity fixes are in `helpers/operators/`, which is shared infrastructure. This affects all policies that use these operators (ALNS, HGS, PG-CLNS, etc.), not just PG-CLNS. The behavior change is a bug fix that makes the operators more robust.

## Owner Rulings Addressed

- **Codex Review §10.2 #2:** "shared `greedy_profit_insertion` seed-route path returns `[[1]]` for demand20/capacity10" → Fixed
- **Codex Review §10.2 #2:** "Add the owner-required parity tests for the operators that should stay identical" → 11 tests added
- **Codex Review §10.2 #2:** "Qwen #82: add the HMLNS verification evidence for the temperature-calibration behavior change" → 6 tests added
