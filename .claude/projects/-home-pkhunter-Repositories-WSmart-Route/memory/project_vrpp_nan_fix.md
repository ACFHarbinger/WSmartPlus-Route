---
name: vrpp-action-mask-nan-fix
description: NaN crash in VRPPEnv decoder when mandatory bin has zero waste — deadlock fix in _get_action_mask
metadata:
  type: project
---

Mandatory-bin + zero-waste deadlock in `VRPPEnv._get_action_mask` (`logic/src/envs/routing/vrpp.py`).

**Why:** `LookaheadSelection._add_bins_to_collect` can mark a bin as mandatory even if its current fill is 0 (recently collected, high accumulation rate → overflows before next cycle). The policy_na input sets `waste = bins.c / 100`, so these bins have `waste == 0.0`. The action mask's `waste > 0` filter blocks them while the mandatory constraint simultaneously blocks the depot, leaving all logits -inf → `softmax` NaN → crash.

**Fix:** In `_get_action_mask`, after the waste filter, re-allow pending mandatory bins regardless of waste level:
```python
mask[:, 1:] = mask[:, 1:] | pending_mandatory[:, 1:]
```
Mandatory is a hard operational constraint; the waste filter is only a profitability heuristic.

**How to apply:** If decoder NaN crashes recur on other envs/policies, check the same pattern: mandatory + waste-filter interaction can deadlock action mask.
