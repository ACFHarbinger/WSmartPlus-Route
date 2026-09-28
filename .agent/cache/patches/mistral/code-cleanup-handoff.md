# Handoff: Mistral — code cleanup round (#80 C3, #81 C4, #82 C5)

**Base:** `main` at `6d500ed02`. Worktree `~/.cache/wsr-review/mistral-code`
(local commits only; nothing pushed). **Apply order: 81 → 80 → 82** — each
patch is the diff of the next commit in my worktree chain
(`3211d41b3` → `d2d95c924` → `e7b65553f`), so they must be applied in that
sequence. All three `git apply --check` pass in sequence on a pristine
`6d500ed02` worktree, and the assembled result is byte-identical to my
verified tree.

## issue-81-c4-dead-code.patch (18,251 lines, mostly deletions)

- **D-mistral-01–04, 06–08:** RL contextual-bandit helper cluster
  (`agents/contextual/`, `agents/contextual_bandits.py`, `evolution_cmab.py`,
  `features/context.py`), `policies/context/`, the dead
  `pattern_and_itinerary.py` twin, the four orphaned exact-solver
  `params.py` (CP-SAT, LBBD, PH, ST-EF — re-verified unread at this base),
  `monkey_patch.py`, `logic/examples/`, `export_onnx.py`, `logic/store/`,
  `logic/migrations/`.
- **D-mistral-09 (test-only, per ruling):** the whole dr_alns cluster with
  its test (`envs/dr_alns.py`, `models/core/dr_alns/`, `pipeline/rl/core/
  dr_alns.py`, `test_dr_alns.py`), the `train.yaml` `rl.dr_alns` section,
  `DRALNSConfig` (`configs/rl/core/dr_alns.py` + the `RLConfig.dr_alns`
  field and import), and the `gymnasium` dependency (only importer).
- **D-mistral-10 (test-only, per ruling):** `drift_detection`, `stepwise_ppo`,
  `time_tracking` (rl/core), `task_utils`, `training_utils`, `boolmask`
  (+ its property test), `distance_graph_convolution`,
  `normalized_activation_function`, `gradient_tracker`, `ks_aco_qlearning`,
  `validation/debug_utils` — each deleted **with its test(s)**;
  `test_modules.py` trimmed (the two classes + imports; the EGC
  `skipif` decorator preserved — my first regex ate it, caught by the test
  run and restored).
- **D-mistral-11 (unused config fields, grep-verified each):** 7 TrainConfig
  fields (+ fixture keys), `HPOConfig.hop_range` (+ fixture), 27 MetaRLConfig
  fields (+ 9 `meta_train.yaml` keys + dangling section comments),
  `BCConfig.use_exact_separation`, `AdaptiveKernelSearchConfig.time_limit_stage_1`,
  `OptimConfig.lr_min_decay`, `MandatoryManagerSelectionConfig.manager_critical_threshold`
  (+ train.yaml), `bernoulli` (+ the `BernoulliSelectionConfig` class, zero
  readers), `RewardShapingConfig.rewards_size`,
  `ContextFeatureExtractorConfig.selection_threshold` (+ policy_rl_ahvpl.yaml),
  `RLConfig.{contextual,gp_cmab,evolution_cmab}` (+ the
  `LinUCBConfig`/`GPCMABConfig`/`EvolutionaryCMABConfig` classes and their
  `other/__init__` re-exports; `agent_type`'s docstring trimmed to the live
  kinds; the yaml `evolution_cmab` block removed),
  `DemonAlgorithmConfig.max_demon_credit`, `StepCountingConfig.step_limit`
  (+ ac_schc.yaml), `HGSConfig.{min_diversity,diversity_change_rate}`,
  `GRPOConfig.group_size`. **Not deleted:** `RLConfig.context_features`
  (live: `pipeline/rl/meta/contextual_bandits.py`), `bandit`/`td_learning`
  (live: RL-ALNS), `features`/`reward`.
- **D-mistral-12 (per ruling):** the six `logic/gen` scripts removed
  (`export_for_studio`, `export_loss_landscape`, `export_website_data`,
  `gen_dataset_analysis`, `gen_presentation`, `gen_simulation_analysis`);
  `gen_paper_latex.py`, `report_utils.py`, `gen_dist_matrix.py` stay.
- **D-cursor-01:** `MAX_LENGTHS` removed from `constants/simulation.py`
  (`envs/generators/op.py`'s independent local copy untouched).
- **Stale doc references:** AGENTS.md §6.1 no longer names `boolmask.py`
  (the masked_fill example stays); the AGENTS.md constants example drops
  `MAX_LENGTHS`; the PyTorch badge/table say 2.13.0 (pyproject pins
  `torch==2.13.0`); `docs/modules/UTILS_MODULE.md` drops the boolmask /
  training_utils / task_utils sections; the RL features `__init__`,
  `utils/tasks/__init__` and `validation/__init__` docstrings no longer
  mention deleted modules. `docs/moon/roadmaps/*` historical mentions left
  (historical entries are never rewritten).
- **Dependencies dropped from `logic/pyproject.toml`:** `einops`, `pydantic`,
  `ml-dtypes`, `latex2mathml`, `python-pptx`, `docxtpl` (runtime, zero
  importers), `gymnasium`, `onnx`, `onnxsim` (gpu extras; only importer was
  deleted dr_alns/export_onnx). **Kept:** `pyarrow`
  (`tracking/logging/structured_logging.py:145` uses it as the parquet
  engine — my Phase-1 "only export_for_studio" claim was stale at this
  base), `openpyxl` (5 live files).

## issue-80-c3-config-knobs.patch (470 lines)

- **B-mistral-03 (`iterations_per_temp`) — deleted, not wired.** The SA
  solver's schedule is acceptance/frozen-streak driven (`solver.py:357-373`)
  and has no per-temperature iteration loop to wire the knob into; wiring
  would have changed the live SA policy's behaviour for a knob that has
  never worked. Field + docstring + yaml key + the "mapped directly"
  comment (now "(initial_temperature, cooling_rate) mapped directly")
  removed.
- **B-mistral-04 (tracking fields) — deleted:** `wst_tracking_uri`,
  `real_time_log`, `profiler_buffer_size` from `TrackingConfig`, the keys
  from all 7 `logic/configs/tracking/*.yaml`, and the two dead
  `tracking.wst_tracking_uri=...` e2e overrides (plus their now-unused
  `tracking_uri` temp dirs). **Two more unread fields found by the new
  regression test and deleted with them:** `mlflow_run_name`,
  `zenml_store_url` (+ their yaml keys). `tracking_uri` and
  `mlflow_tracking_uri` are live and untouched.
- **Regression test** `logic/test/unit/configs/test_config_fields_are_read.py`:
  every `TrackingConfig`/`SAConfig` field must be referenced by a non-config
  source file (yaml does not count — it only sets keys). **Fails on the
  pre-fix tree** (5 tracking fields + `iterations_per_temp`), **passes on
  the patched tree** (verified by stashing/reverting).

## issue-82-c5-config-refactors.patch (1,288 lines)

- **M-mistral-01 (yaml-link updaters merged):** `ms_updater.py` + `ri_updater.py`
  (177 lines each, byte-similar) replaced by one parameterised
  `utils/target/policy_link_updater.py` (233 lines) exposing the six
  historical names; `controllers/cli/tgt_parser.py` and the package
  `__init__` rewired. **~120 net LOC saved.** New round-trip test
  `logic/test/unit/utils/target/test_policy_link_updater.py` (5 tests:
  ms round-trip, ri round-trip, listings, unknown-key ValueError, dry-run
  no-write) — the precondition my refactor row set, since nothing tested
  the updaters before.
- **M-mistral-02 (config↔params dedup) for the two pairs I own:** the EGH
  and LASM `params.py` re-declared every config field with duplicated
  docstrings (22 and 18 dup windows). Now
  `ExactGuidedHeuristicParams(ExactGuidedHeuristicConfig)` (0 own fields)
  and `LASMPipelineParams(LASMPipelineConfig)` (2 own fields:
  `lbbd_cut_families`, `rl_state_features` keep the historical `None`
  defaults where the config ships list factories — verified by
  field-by-field default comparison). `from_config`/`to_dict` kept.
  270→65 and 391→65 lines (**~530 LOC saved**). All call sites are
  keyword-based (verified). **BPC/MS-BPC-SP pairs NOT touched — Kimi's
  `exact_and_decomposition_solvers/**` this round; asked on the bus.**

## Verification

- `compileall` green after each phase.
- `import_sweep.py` (under the heavy lock): **2041 modules, 1 failed** —
  only `logic.gen.gen_paper_latex`'s pre-existing script-style
  `from report_utils import` (the file is script-runnable; `report_utils.py`
  stays per the ruling). Same failure exists on pristine `6d500ed02`.
- Unit tests (flock): `logic/test/unit/{configs,utils,policies}` —
  **616 passed, 1 failed**: the pre-existing Gurobi-license test pollution
  in `test_ils_rvnd_sp_paper.py` (passes standalone on the pristine base;
  identical behaviour to the 2026-09-26 review). `test_modules.py`:
  **16 passed, 1 skipped** (the EGC torch_sparse skip, preserved).
- Patch chain apply-checked and applied in order on a pristine
  `6d500ed02` worktree; the assembled tree is identical to my verified tree.
- No simulator runs (no runtime-path behaviour change: every deleted module
  had zero importers; the two refactors are behaviour-preserving by
  construction and verified by the parity checks above).

## Behaviour changes (for the export README / #83)

- **None on any live path.** The SA and tracking knob deletions remove
  knobs that no code read; the yaml keys were inert. The EGH/LASM params
  refactor is default-identical (field-by-field verified). New regression
  tests will fail if anyone re-adds an unread config field to those two
  classes.
- `EGH`/`LASM` now inherit their fields from the config dataclasses: a
  field added to the config appears in the params automatically (the
  drift that produced B-kimi-34/D-kimi-06 cannot recur for these pairs).

-- Mistral
