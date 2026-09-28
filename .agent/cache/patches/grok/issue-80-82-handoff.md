# Grok — simulator fixes (#80, #82)

Base `6d500ed02`. Worktree `~/.cache/wsr-review/grok-code` (detached, not committed). Shared checkout logic was not edited. `assets/papers/**` was not touched. D-grok-01 stays.

Apply **issue 80, then issue 82**. Both `git apply --check` on a clean `6d500ed02` tree, and issue 82 also checks after issue 80 is applied. They share `simulator.py`: 80 records failures, 82 removes that file's extra mean/std write.

## Patches

- `.agent/cache/patches/grok/issue-80-simulator-robustness.patch`
- `.agent/cache/patches/grok/issue-82-simulator-refactors.patch`

`actions/route_construction.py` is outside this lane. The only edit there is the M-grok-01 gate. No other lane lists that build.

## #80 behavior

- Resume clock: `ctx.tic = perf_counter() - ctx.run_time`. The next checkpoint elapsed time is the stored elapsed time plus new work. The DS-15 sample `time` field is still the sum of daily policy times.
- Failures are recorded on every path, with `policy`, `sample`, `sample_id`, and `error`:
  - sequential `CheckpointError` (previously swallowed)
  - a day-loop result with `success` false (the running state stores the error and returns it)
  - any other sequential exception
  - the parallel callback, which previously popped `sample_id` and then printed `unknown`
- `simulator_testing` raises `SystemExit` when that list is non-empty. `SystemExit` is not an `Exception`, so `run_simulator_test` does not turn it into a successful return. The process status is 1.
- A policy with no successful sample is left out of the aggregate. It is not written as an all-zero mean. If some samples succeeded, the mean and std use those samples only, and the process still exits non-zero.
- Resume sample lists are keyed by `policy_result_key` (the display slug, same string as `ctx.pol_name` and `log_<slug>_*.json`), in policy order. A raw id such as `alns` no longer KeyErrors against its slug on resume.
- `start_with_fill` still copies row 0 into the opening level. The daily increment, true and noisy, is `waste_fills[day - 1]`, including when `start_with_fill` is set. A sample with one row per day is not indexed one past the end. Day 1 therefore uses row 0 both as the opening level and as that day's increment.
- Expanded display names (`ms` or `ri` in the id) take the longest registered constructor key. `alns` and `hgs` are not dropped. `ms_regular_alns_ri_none` displays `ALNS`. Ids without those tokens are unchanged, and `get_canonical_policy_name` (the seed key) is unchanged.
- The day context receives `model_ls` padded to at least three slots. A missing tuple is `(None, None, None)`, which stops the neural-agent 3-tuple unpack and the branch-and-price `[2]` index from raising before the policy body. A neural policy with no weights still fails later on a null model; that body is Gemini's tree.

## #82 behavior

- Scenario tree: still built before `adapter.execute` unless `uses_scenario_tree` is `False`. The default is true. No policy class was edited, so nothing opts out and this is behavior-neutral for current runs. `bins/prediction.py` stays.
- Mean and std of `SIM_METRICS` are written once, from `display_log_metrics`, after aggregation. `FinishingState` writes `samples` and `daily` only. The sequential loop still computes the in-memory mean and std and does not write them. Resume still calls `output_stats`, which reads logs that already exist. `tracking/**/analysis.py` also writes mean/std when it reads logs; that file is Mistral's and was not edited.
- A `SimulationContext.run()` that stops in `FinishingState` and never reaches the orchestrator no longer writes a one-sample `mean` section. The orchestrator path still writes it through `display_log_metrics`.

## Not changed

DS-15, DS-16, overflow accounting, profit, directed kilometres, and the paper patches. No `test_sim`. Checkpoints stay.

## Tests

`~/.cache/wsr-main-venv`, under `flock ~/.cache/wsr-review/heavy.lock`, from the worktree. 50 passed:

`test_simulator_robustness.py`, `test_simulator_refactors.py`, `test_states.py`, `test_bins_core.py`, `test_features_test.py`, `test_simulation_core.py`, `test_simulator_integration.py`, `test_policy_seed_key.py`, `logic/test/unit/pipeline/features/test_test_pipeline.py`.

New tests cover the resume clock sign, the ALNS display token, the stats-file last day, empty and partial aggregation, sequential checkpoint and unsuccessful results, the parallel sample id, the resume slug, non-zero exit, the 3-tuple, the scenario-tree opt-out and the default build, and a single mean/std writer.
