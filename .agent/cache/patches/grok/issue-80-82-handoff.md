# Grok — simulator fixes, revision (#80, #82)

Base `f1975cdf9` (`main` after clean-up batch 1). The simulator files in that commit match `6d500ed02`, which is the tree the worktree still has checked out. Shared checkout logic was not edited. Nothing was committed.

Apply **issue 80, then issue 82**. `git apply --check` passes for each patch on a clean `f1975cdf9`, and issue 82 checks again after issue 80.

## Patches

| Patch | SHA-256 |
|---|---|
| `.agent/cache/patches/grok/issue-80-simulator-robustness.patch` | `fefa28eb96a15866b8013a1c3d04c8a5a058e764ad4e6c010f19b0504f37b046` |
| `.agent/cache/patches/grok/issue-82-simulator-refactors.patch` | `e5caf732dcd88c2705a0beca64829ef6e669abccc8d917606ca1a205f26fcae8` |

`actions/route_construction.py` is outside this lane. The only edit there is the M-grok-01 gate. No other lane lists that build.

## Revision for §10.2 #3

The first #80 patch deposited row 0 twice on the stats-file path: `set_sample_waste` copies it into the opening level, and `load_filling` then added `waste_fills[day - 1]`, so day 1 added that same row again. Rows 10, 20, 30 became level 20 after day 1.

The corrected ledger keeps the two roles apart when `start_with_fill` is set (the stats file turns that flag on):

- Row 0 is the opening level only.
- Day `d` deposits row `d`. With rows 10, 20, 30 and a 2-day horizon the levels are 10, then 30, then 60.
- The deposited rows are exactly `rows[1:]`, and the final level is their sum with the opening row when nothing is collected and every value stays under capacity.
- The day after the last row raises `IndexError`. It does not wrap onto row 0.

Generated samples are `n_days` increment rows, so that layout used to index off the end on day `n_days`. When `stats_filepath` is set and no waste file is loaded, `Bins` draws `n_days + 1` rows: row 0 is the opening draw, and days 1..`n_days` deposit rows 1..`n_days`. The final day reads the last row. A file-backed sample is not extended; it still has to contain the opening row itself.

The default path (`start_with_fill` false, `stats_filepath` null) still generates `n_days` rows and reads `waste_fills[day - 1]`. That is the path the paper runs use. No 10-day `test_sim`: it would not enter the stats-file branch.

## #80 behavior otherwise

- Resume clock: `ctx.tic = perf_counter() - ctx.run_time`. The next checkpoint elapsed time is the stored elapsed time plus new work. The DS-15 sample `time` field is still the sum of daily policy times.
- Failures are recorded with `policy`, `sample`, `sample_id`, and `error` on the sequential `CheckpointError` path, on an unsuccessful day-loop result, on any other sequential exception, and in the parallel callback.
- `simulator_testing` raises `SystemExit` when that list is non-empty. The process status is 1.
- A policy with no successful sample is left out of the aggregate. Successful samples are averaged on their own, and the process still exits non-zero if any sample failed.
- Resume sample lists use `policy_result_key` (the display slug), in policy order.
- Expanded display names take the longest registered constructor key. `ms_regular_alns_ri_none` displays `ALNS`. `get_canonical_policy_name` is unchanged.
- `model_ls` is padded to at least three slots.

## #82 behavior

- Scenario tree: still built unless `uses_scenario_tree` is `False`. Nothing sets that flag, so current runs are unchanged. `bins/prediction.py` stays.
- Mean and std of `SIM_METRICS` are written once, from `display_log_metrics`. `FinishingState` writes `samples` and `daily` only. The sequential loop still computes the in-memory mean and std. Resume still calls `output_stats` on logs that already exist. `tracking/**/analysis.py` was not edited.
- A `SimulationContext.run()` that never reaches the orchestrator no longer writes a one-sample `mean` section. The orchestrator path still writes it through `display_log_metrics`.

## Not changed

DS-15, DS-16, overflow accounting, profit, directed kilometres, and D-grok-01. Checkpoints stay.

## Tests

`~/.cache/wsr-main-venv`, from the worktree. `test_stats_file_horizon_keeps_opening_row_out_of_the_deposits` checks the 10/20/30 ledger (final level 60). `test_generated_stats_sample_includes_the_final_day` checks that a generated stats sample has `n_days + 1` rows and that day `n_days` deposits the last row.
