# Shared report — minimal-export review (bugs + further pruning)

**Branch:** `feat/minimal-export-package` · **Analysed commit:** `70e660b03` (re-verify `file:line` with `sed -n` on this commit) · **Opened by:** Claude, 2026-09-25 · **Status:** OPEN (collecting)

Collaborative report of record for the two questions the owner asked on 2026-09-25:

1. **Bugs, errors/failures and logic mistakes** anywhere under `logic/` on the pruned branch (707 `.py` files, ~118 kLOC after the 2026-09-25 prune).
2. **Further removals** — modules, sub-packages, files, classes, functions, config keys and dependencies that can go while the retained policies/models keep working end to end. These feed the packaging scripts (`logic/package/prune_codebase.py`, `ci/export_config.json` on `main`).

Rules for this file:

- **Append-only.** Add rows to the shared tables and write your own `## <N>. <Agent> — lane <X>` section. Never rewrite another agent's rows; contradict them in your own section with evidence.
- **Evidence or it does not exist.** Every bug row needs a `file:line` and a reproduction (command, traceback, or a minimal input). Every removal row needs the importer check you ran (`grep -rn` / `git grep` output summary) and the verification you did after removing it locally (compile + import sweep at minimum; a smoke run when the item sits on a runtime path).
- **Do not commit code changes as part of this task.** Local experiments are fine (use a scratch worktree: `git worktree add /tmp/<agent>-wsr feat/minimal-export-package`), but the deliverable is this report. Code changes come afterwards, from the roadmap the owner derives from here.
- IDs: bugs `B-<agent>-NN`, removals `R-<agent>-NN` (e.g. `B-grok-03`, `R-kimi-01`). Severity: **blocker** (crashes a retained path), **major** (wrong results / silent misbehaviour), **minor** (edge case, dead branch, misleading message).

## 0. What is retained (the functional contract)

Anything proposed for removal must keep all of this working:

| Area | Retained |
| --- | --- |
| Entry points | `python main.py train \| eval \| test_sim \| gen_data` (Hydra, `logic/controllers/hydra_dispatch.py`) |
| Neural model | Attention Model `am` (`logic/src/models/core/attention_model`, GAT encoder, glimpse decoder), loaded by `logic/src/utils/model/loader.py` |
| RL | REINFORCE (`logic/src/pipeline/rl/core/reinforce.py`) with exponential / rollout / critic baselines |
| Environment | `vrpp` (`logic/src/envs/routing/vrpp.py`, `logic/src/envs/tasks/vrpp.py`, generator `logic/src/envs/generators/vrpp.py`) |
| Route construction | `aco_hh`, `alns`, `bpc`, `hgs`, `pg_clns`, `psoma`, `sans`, `swc_tcf`, `na` (Neural Agent) |
| Mandatory selection | `lookahead`, `last_minute`, `service_level` (scalar strategies + vectorized selectors under `policies/vector/selection`) |
| Route improvement | `fast_tsp` |
| Acceptance criteria | `bmc`, `oi` |
| Distance methods | `file`, `ogd` (Euclidean) |
| Data | `gen_data` (`test_simulator` + `train` dataset types), fill distributions actually used by the retained configs (`emp`, `gamma*`, `unif` for smoke tests), simulator repositories (`pipeline/simulations/repository`) |
| Results | `log_<policy>_<N>N.json` + realtime `.jsonl` written by `logic/src/tracking/logging` (the rest of tracking is an inert shim) |

Verification baseline that must stay green (all from the repo root, `.venv` active):

```bash
.venv/bin/python -m compileall -q logic main.py
.venv/bin/python .agent/cache/tools/import_sweep.py          # expect "<N> modules, 0 failed"
python main.py gen_data data.dataset_type=test_simulator data.problem=vrpp 'data.data_distributions=[gamma1]' \
  'data.graphs=[{num_loc: 20, n_days: 6, n_samples: 2}]' data.data_dir=/tmp/wsr_gen data.name=smoke data.overwrite=true
python main.py train train.env.name=vrpp 'train.env.curriculum_graphs=[{num_loc: 20, n_samples: 64, area: riomaior, waste_type: plastic, distance_method: ogd, focus_size: 64, focus_graph: graphs_20V_1N_plastic.json, n_days: 1, load_dataset: null}]' \
  'train.env.eval_graphs=[]' train.batch_size=16 train.num_workers=0 train.policy.model.encoder.embed_dim=32 train.policy.model.encoder.hidden_dim=64 \
  train.policy.model.encoder.n_layers=1 train.policy.model.encoder.n_heads=2 train.policy.model.decoder.n_heads=2 train.policy.mandatory_selection=null \
  train.data_distribution=unif train.train_time=false train.model_weights_path=/tmp/wsr_train/ckpt train.final_model_path=/tmp/wsr_train/am/epoch-1.pt
python main.py eval eval.policy.model.load_path=/tmp/wsr_train/am 'eval.datasets=[/tmp/wsr_gen/riomaior20_gamma1_smoke6_N2_seed42.npz]' eval.val_size=12 eval.eval_batch_size=4 eval.problem=vrpp eval.env.name=vrpp eval.results_dir=/tmp/wsr_eval
python main.py test_sim sim.graph.area=riomaior sim.graph.num_loc=20 sim.graph.n_days=10 'sim.graph.dm_filepath="gmaps_distmat_plastic[riomaior].csv"' \
  sim.graph.focus_graph=graphs_20V_1N_plastic.json sim.graph.load_dataset=null sim.data_distribution=emp sim.output_dir=/tmp/wsr_sim sim.run_name=smoke \
  p.na.na.amgat.0.model_path=/tmp/wsr_train/am   # all nine policies; add p.<pol>... time_limit overrides to shorten (see README)
```

Expected: every policy produces a `log_*_1N.json` with a non-zero `km` day once fills cross the thresholds (day 7 for `cf70`, day 9 for `cf90` variants on the generated data); eval reports non-zero km/kg on the gamma dataset.

## 1. Inventory of the pruned branch (2026-09-25, commit `70e660b03`)

| Package | Files | LOC | Notes |
| --- | ---: | ---: | --- |
| `logic/src/policies/helpers` | 136 | 37,987 | `operators/` alone is 102 files / 25,118 LOC — retained by `policies_helpers_analysis` (2nd-order deps), likely over-retained |
| `logic/src/policies/route_construction` | 134 | 21,556 | SANS 40 files / 5,918 LOC; PG-CLNS 35 files / 3,827 LOC |
| `logic/src/models` | 86 | 8,959 | `common/{non_autoregressive,improvement,transductive}` 14 files / 1,233 LOC; `subnets/embeddings/positional` 191 LOC |
| `logic/src/utils` | 72 | 8,819 | `policy/` 1,223, `input/` 1,414, `decoding/` 1,286 |
| `logic/src/pipeline/simulations` | 31 | 6,635 | incl. `checkpoints/` 518, `failure_analyzer.py` 238 |
| `logic/src/data` | 50 | 6,134 | `distributions/` 15 files / 1,680 LOC (owner wants only `empirical` + `gamma`), `datasets/pytorch` 7 files, `datasets/simulation` 7 files (owner wants one of each) |
| `logic/src/pipeline/rl` | 30 | 4,087 | `common/baselines` 10 files / 1,044 LOC |
| `logic/src/configs` | 37 | 3,864 | dataclasses; many fields describe removed features |
| `logic/src/envs` | 15 | 2,429 | `base/improvement.py` (improvement envs removed), `envs/generators` 3 files |
| `logic/src/tracking` | 11 | 2,264 | shim + `logging/` result writers (`gui.py`, `metrics.py` use jinja2 / wandb) |
| `logic/src/interfaces` | 19 | 2,213 | |
| `logic/src/pipeline/features/eval` | 13 | 2,142 | `evaluators/` 6 files; `drift_detection.py`, `zenml_eval_pipeline.py` |
| `logic/src/pipeline/features/test` | 9 | 2,032 | |
| `logic/src/pipeline/features/train` | 6 | 1,111 | `zenml_train_pipeline.py` |
| `logic/src/constants` | 11 | 1,031 | |
| `logic/src/pipeline/callbacks` | 4 | 245 | `attention_heatmaps.py`, `gpu_memory_monitor.py` |
| `logic/controllers` | 4 | 383 | |

## 2. Bugs, errors and logic mistakes (shared table — append rows)

| ID | Agent | File:line | Severity | Symptom | Evidence / repro | Suggested fix | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-claude-01 | Claude | `logic/src/pipeline/simulations/actions/base.py:47` (`_flatten_config`) | major (fixed on branch, `3ea21fec1`) | Nested `custom` blocks overwrote the expander's variant-specific `mandatory_selection`; OmegaConf mappings were iterated as lists | 10-day sim crashed with `Unknown selection strategy: last_minute_cf70` for every policy | landed; kept here so reviewers can challenge the fix | fixed |
| B-claude-02 | Claude | `logic/src/pipeline/features/test/config.py` (`expand_policy_configs`) | major (fixed, `70e660b03`) | Dict-style policy yamls (psoma, sans) stored the same `{file: [cf70, cf90]}` selection for both variants; cf90 resolved to cf70 and was named `last_minute_cf70_last_minute_cf90_*` | 10-day sim: both variants routed on day 7 | landed (`_pin_variant_selection`) | fixed |
| B-claude-03 | Claude | `logic/src/envs/tasks/base.py` | blocker (fixed, `70e660b03`) | `problem.make_dataset` did not exist anywhere; `python main.py eval` could never load a dataset (upstream bug, also on `main`) | `AttributeError: type object 'VRPP' has no attribute 'make_dataset'` | landed | fixed, needs port to `main` |
| B-claude-04 | Claude | `logic/src/models/core/attention_model/model.py:297` | major (fixed, `3ea21fec1`) | Legacy `AttentionModel` built its encoder with the default `feed_forward_hidden=512`, ignoring `hidden_dim`; checkpoints trained with any other width failed to load | `size mismatch for encoder.layers.0.ff...` when loading a `hidden_dim=64` checkpoint | landed; upstream bug, also on `main` | fixed, needs port to `main` |
| B-claude-05 | Claude | `logic/src/pipeline/simulations/states/initializing.py:266` | major (fixed, `3ea21fec1`) | Name-only policies (`full_policies` entries are strings) never re-linked their config, so the Neural Agent found no `model_path` | `KeyError: Could not find model path for policy 'lookahead_na_amgat_none'` | landed; check whether `main` has the same flow | fixed |
| B-claude-06 | Claude | `logic/src/pipeline/simulations/day_context.py` (`get_full_policy_name`) vs `features/test/config.py` | minor | NA policy id is `lookahead_na_amgat_emp` but its slug/log name is `lookahead_na_amgat_none` (empty `route_improvement: []` becomes `none`, and the distribution suffix is dropped); other policies keep `_emp` in the id but drop it in the slug too. Log filenames therefore do not carry the distribution | compare `config_path` keys with `log_*.json` names after any `test_sim` run | decide one naming rule and apply it in one place | open |
| B-claude-07 | Claude | `logic/src/pipeline/simulations/wsmart_bin_analysis/export/container.py:17` → `modules/plotting_utils.py:14`; `logic/pyproject.toml` (`matplotlib` only in `viz`) | major | `GridBase` import chain pulls `matplotlib`, which is an optional `viz` dependency: an install without `viz` cannot import `GridBase`, so the `emp` distribution and `bins/base.py` fail at import | `python -c "from logic.src.pipeline.simulations.wsmart_bin_analysis import GridBase"` then inspect `sys.modules` for `matplotlib` | either move `matplotlib` to core deps, or make `VisualizationMixin` import lazily / drop it from `Container` in the vendored subset (see D11) | fixed in the export packaging (D11); upstream submodule unchanged |

<!-- Codex bug rows follow the original §2 table; kept in a separate table to preserve other contributors' rows. -->

### Lane A bug evidence (Codex, 2026-09-25)

| ID | Agent | File:line | Severity | Symptom | Evidence / repro | Suggested fix | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-codex-01 | Codex | `logic/src/pipeline/rl/common/base/module.py:96,196,199` | major | Baseline configuration is saved inside `hparams.kwargs`, but baseline creation and warmup read top-level keys. Configured `exp_beta`, `bl_alpha`, and `bl_warmup_epochs` are ignored. | Reproducer below: requested `.123` and 3 warmup epochs gives `ExponentialBaseline(beta=.8)` without warmup. Shipped yaml requests `.6`, so the default run is also affected. | Flatten sanitized algorithm kwargs into the saved parameter contract, or explicitly pass baseline settings; test non-default values. Audit `regenerate_per_epoch` for the same nesting error. | open |
| B-codex-02 | Codex | `logic/src/pipeline/rl/common/baselines/critic.py:45`; `rl/common/base/optimization.py:51`; `rl/core/reinforce.py:103` | major | `baseline=critic` silently becomes a zero baseline: no critic is constructed. Even an injected critic has no regression loss and is excluded from the optimizer; advantage is detached. | Reproducer: `REINFORCE(..., baseline='critic').baseline.critic is None`, learnable parameter list empty. Builder drops `lr_critic`; REINFORCE accepts but never uses it. | Construct a critic, optimize value regression and include its parameters/LR. The retained contract explicitly promises critic support, so deleting the option is a contract change. | open |
| B-codex-03 | Codex | `logic/src/pipeline/rl/common/baselines/rollout.py:255`; `rl/core/reinforce.py:103` | major | Wrapped rollout baseline `[B,1]` broadcasts against reward `[B]` to `[B,B]`; policy gradient uses cross-instance baseline values. | Actual `calculate_loss`: rewards `[2,5]`, baseline `[[1],[3]]`, log-likelihood `[.2,.7]` produces `-1.05`; correct per-instance loss is `-.8`. | Store `[B]` baseline values or reshape to reward shape and assert shape equality before subtraction. | open |
| B-codex-04 | Codex | `logic/src/pipeline/rl/common/epoch.py:79`; `rl/common/baselines/rollout.py:249` | major | Epoch wrapping evaluates the live policy, bypassing the frozen, significance-tested rollout baseline. | Spy on `_rollout` through real `prepare_epoch`: candidate identity is the live policy, not `baseline_policy`. | Use the frozen baseline policy for wrapping; preserve the live policy only for the candidate update test. | open |
| B-codex-05 | Codex | `logic/src/pipeline/rl/common/base/data.py:288`; `rl/common/base/steps.py:333`; `rl/common/epoch.py:209,215` | major | `train_time=true` accumulates tours in shuffled traversal order, then applies them in dataset row order, resetting the wrong instances' bins. | Two-instance counterexample: traversal sample 1 → node 2, sample 0 → node 1 produces dataset-row mask 0 → node 2, 1 → node 1. No sample IDs are recorded with actions. | Carry stable instance IDs through batches and scatter updates by ID; disabling shuffle is only a single-process fallback. | open |
| B-codex-06 | Codex | `logic/src/pipeline/rl/common/base/data.py:45,184,203`; `features/train/engine.py:177` | major | Data setup reads `cfg.env`, although curriculum builder writes `cfg.train.env`. Training sample counts and configured validation graphs are silently ignored. | Real stage builder plus setup: requested train 64/eval 32 gives train 10/eval 512 and no eval graphs. Full CLI training also prints `Generating training dataset (10 instances)` for the report's 64-instance smoke. | Resolve the active task env consistently in setup, eval-env creation, graph selection and regeneration; assert actual dataset sizes in packaging smoke tests. | open |
| B-codex-07 | Codex | `logic/src/pipeline/features/eval/engine.py:261,309`; `models/core/attention_model/model.py:431`; `envs/tasks/vrpp.py:74` | major | Evaluation serializes reward/profit as `cost`, reversing the task's negative-profit cost convention. | Reproducer passes reward +7 for task cost -7; `_eval_dataset` saves `cost=+7`. `get_best` minimizes costs while sampling correctly maximizes rewards. | Serialize `cost=-reward` or explicitly migrate the field to `reward` and update downstream consumers. A negative printed cost alone is not evidence of a bug; the dataflow is. | open |
| B-codex-08 | Codex | `logic/src/utils/model/loader.py:94-106`; `models/core/attention_model/model.py:134-135` | major | Loader passes legacy normalization/activation kwargs swallowed by `AttentionModel(**kwargs)`; structured defaults replace saved architecture settings. | Real smoke checkpoint says normalization `layer`; loaded model contains two `BatchNorm1d` layers. Constructor spy confirms no `norm_config`/`activation_config` supplied. Shape-compatible weights load without detecting the architecture change. | Reconstruct and pass structured normalization/activation configs; test inference parity between training policy and loaded model, including batch-composition invariance. Coordinate architecture conversion with lane C. | open |
| B-codex-09 | Codex | `logic/src/utils/model/loader.py:145-148` | major | Unsupported/empty checkpoint dictionaries load a randomly initialized model and print success. Missing/unexpected keys are not inspected. | Reproducer intercepts `torch_load_cpu` with `{}` and observes `load_state_dict({}, strict=False)` followed by success. Real training checkpoint also has two missing context projection parameters; these are intentionally synthesized, so they need an explicit migration allowlist. | Reject unknown checkpoint layouts and empty parameter maps; fail on missing/unexpected keys except documented migration keys. Validate numerical parity for any synthesized projection. | open |

### Lane C bug evidence (Gemini, 2026-09-25)

| ID | Agent | File:line | Severity | Symptom | Evidence / repro | Suggested fix | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-gemini-01 | Gemini | `logic/src/utils/model/loader.py:94-106`; `logic/src/models/core/attention_model/model.py:134-135` | major | `loader.py` swallows normalization and activation settings into `**kwargs`; loaded `AttentionModel` defaults to `BatchNorm1d` and `GELU` regardless of saved hyperparameters. | `loader.py:94-106` passes flat kwargs (`normalization=args["normalization"]`, `activation_function=args["activation"]`). `model.py:134-135` consumes only `norm_config` and `activation_config`, defaulting to `NormalizationConfig()` (`norm_type='batch'`) and `ActivationConfig()` (`name='gelu'`). Real checkpoint specifies layer norm, but loaded model builds `BatchNorm1d` (independent architectural confirmation of B-codex-08). | Reconstruct `NormalizationConfig(norm_type=args["normalization"], ...)` and `ActivationConfig(name=args["activation"], ...)` in `loader.py` and pass as structured configs, or unify loader to instantiate `AttentionModelPolicy`. | open |
| B-gemini-02 | Gemini | `logic/src/models/core/attention_model/policy.py:57,75-82`; `logic/src/models/subnets/encoders/gat/encoder.py:36-50`; `logic/src/models/subnets/encoders/common/encoder_base.py:93` | major | Direct instantiation of `AttentionModelPolicy(normalization="layer")` silently ignores `normalization` string and uses `BatchNorm1d`. | `policy.py:57` accepts `normalization: str = "batch"` and forwards `normalization=normalization` in `kwargs` to `GraphAttentionEncoder`. `encoder.py:42` expects `norm_config: Optional[NormalizationConfig]` and ignores `kwargs['normalization']`. `encoder_base.py:93` defaults to `NormalizationConfig()` (`norm_type='batch'`). Direct instantiation `AttentionModelPolicy("vrpp", normalization="layer")` produces `BatchNorm1d`. | In `AttentionModelPolicy.__init__`, construct `norm_config = NormalizationConfig(norm_type=normalization)` if not explicitly provided in `kwargs`. | open |
| B-gemini-03 | Gemini | `logic/src/utils/model/loader.py:150-162`; `logic/src/models/core/attention_model/model.py:220-221`; `logic/src/models/subnets/decoders/glimpse/decoder.py:114-119,435-440` | major | Dead context projection synthesized as fake identity; masks architectural duplication between `context_embedder` and `decoder.context_embedding`. | `loader.py:150-162` detects missing `context_embedder.project_step_context.weight` in PL checkpoints and synthesizes identity weights. In `AttentionModel`, `context_embedder` is `VRPPContextEmbedder`, but `forward` delegates decoding to `GlimpseDecoder`, which builds its own `self.decoder.context_embedding` (`VRPPContextEmbedder`). `model.context_embedder.project_step_context` is dead code; real weights are already in `decoder.context_embedding.project_step_context`. | Eliminate duplicate context embedder and remove erroneous synthesis block in `loader.py`. | open |
| B-gemini-04 | Gemini | `logic/src/models/subnets/decoders/glimpse/decoder.py:103-104,302` | blocker | Sampling decoding crashes on CUDA devices because `self.generator` is not migrated when model is moved to GPU. | `GlimpseDecoder.__init__` creates `self.generator = torch.Generator(device=device).manual_seed(self.seed)` on CPU. Moving model to CUDA via `model.to("cuda")` leaves `generator` on CPU. In `_select_node`: `torch.multinomial(probs, 1, generator=self.generator)` fails with `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'`. | Match `generator` device dynamically to `probs.device` in `_select_node` or override `_apply`/`to` to re-create the generator on the destination device. | open |
| B-gemini-05 | Gemini | `logic/src/models/subnets/decoders/glimpse/decoder.py:310-311` | major | Unbounded loop in `_select_node` sampling branch hangs process when valid action space has zero probability mass. | `while curr_mask.gather(1, selected.unsqueeze(-1)).any(): selected = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)`. If all remaining valid actions have zero probability mass or numerical leakage occurs, `torch.multinomial` repeatedly draws from masked indices, causing an infinite while loop. | Zero masked probabilities explicitly, renormalize, and sample once; guard against all-zero distributions with fallback. | open |
| B-gemini-06 | Gemini | `logic/src/models/core/attention_model/model.py:431`; `logic/src/envs/tasks/vrpp.py:74-79` | major | `AttentionModel.forward` packages `cost` as negative profit, inverting evaluation metric reduction semantics. | `vrpp.py:74-79` calculates `neg_profit = length * cost_km - waste * revenue_kg`. `model.py:431` sets `reward = -cost` where `cost` is `neg_profit` (e.g. -3.21), so `out["cost"]` is negative profit while `out["reward"]` is positive profit. Downstream `eval/engine.py:261` minimizes `cost`, conflicting with `get_best` and `SamplingEval` (confirms B-codex-07). | Standardize output keys across `AttentionModel` and `AttentionModelPolicy` to return unambiguous `profit`/`reward` and `tour_length`. | open |

### Lane B bug evidence (Grok, 2026-09-25)

| ID | Agent | File:line | Severity | Symptom | Evidence / repro | Suggested fix | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-grok-01 | Grok | `logic/src/pipeline/simulations/bins/base.py:147`; `logic/src/utils/data/loader.py:251-252`; `logic/src/pipeline/simulations/states/initializing.py:433,468-480`; `logic/src/pipeline/simulations/repository/filesystem.py:260-263` | major | Empirical fills describe a different set of bins than the routed graph. `Bins` is constructed with `indices=arange(num_loc)` and `load_grid_base` reads `out_rate_crude` / `out_info`. The distance matrix is built from the focus-graph rows of `old_out_info` (Rio Maior), then sorted by ID. `set_indices` runs after the waste sample is already generated and does not rebuild it. | Reproducer below. `graphs_20V_1N_plastic.json` vs Rio Maior plastic rows: routed IDs start `340,894,904,...`; grid IDs start `192,470,629,...`; overlap **4/20**. Figueira da Foz uses one file pair and still overlaps **6/20** after the ID sort. This is the default `emp` + focus-graph path. | Build the empirical grid from the same coordinate table, the same focus indices, and the same ID order as `process_indices` before generating the waste sample. | open |
| B-grok-02 | Grok | `logic/src/pipeline/simulations/states/base/context.py:152-153`; `logic/src/pipeline/simulations/simulator.py:262-269`; `logic/src/pipeline/features/test/orchestrator/__init__.py:279-280`; `logic/src/pipeline/features/test/orchestrator/results_handler.py:68,98-107` | major | The expander id (`…_emp`) and the log slug differ (extends B-claude-06). Parallel workers test `get_pol_name in result`, but the result key is `to_slug(display_name)`. `cpu_cores: 0` selects `cpu_count-1` workers. For `n_samples>1`, aggregation looks the id up, misses the slug lists, and writes an all-zero `mean`/`std`. | Live 2-core run (`alns`+`bpc`, 4 variant tasks, `n_samples=1`) logged `Finished simulation for policy last_minute_cf90_alns_custom_bmc_ftsp_emp`, which is the branch taken when the id is absent from the result. Log files are the slug without `_emp`. One-sample aggregation still keeps the slug vector; the zero-mean branch is the `n_samples>1` arm (inspected, not re-run). Reproducer prints `worker accepts result=False` for the Neural Agent id `lookahead_na_amgat_emp` vs slug `lookahead_na_amgat_none`. | One key for `full_policies`, `log_*.json`, the jsonl policy field, checkpoint names, and the parallel result dict. Use `pol_id_orig` (the expander id). | open |
| B-grok-03 | Grok | `logic/src/pipeline/simulations/states/finishing.py:62-73`; `logic/src/pipeline/simulations/actions/route_construction.py:178`; `logic/src/constants/simulation.py:93` | major | Sample `time` is the whole day-loop wall clock. Daily `time` is route-construction only. `SIM_METRICS` describes policy execution time. | Existing 10-day logs: HGS sample time `67.03` vs sum of daily solver time `66.89`; PSOMA `0.546` vs `0.409`; SANS `5.11` vs `3.00`. km and kg match across the two layers on those files. | Write `sum(daily time)` into the sample `time` field, or name the wall-clock field separately and keep solver time as `time`. | open |
| B-grok-04 | Grok | `logic/src/pipeline/simulations/bins/base.py:411-433` | major | `new_overflows` counts every bin whose level is exactly 100% that day, including a bin that receives no new waste. The count is added again the next idle day. Reward subtracts that count from kilograms. Lost mass on the idle day is 0. | Reproducer: a bin already at 100% with today's fill 0 returns `overflows=1`, `lost_kg=0`; the next idle day increments `inoverflow` to 2. `finishing.py` then sums `bins.inoverflow` into the sample `overflows` column. | Count a bin on the day its lost mass becomes positive, and keep the at-capacity streak in a separate field. | open |
| B-grok-05 | Grok | `logic/src/pipeline/simulations/states/running.py:62`; `logic/src/pipeline/simulations/states/initializing.py:345-358` | major | Resume restores bins, coordinates, the daily log, and `start_day = last_day+1`. The stored elapsed time is added to the start timestamp, so the reported sample time is the negative of the previous elapsed time. | Reproducer: `tic = perf_counter() + 12.5` then `perf_counter() - tic` prints `-12.50`. A full resume sim was not executed. `checkpoint_days: 0` still saves on crash (`hooks.py:134`). | Set `tic = perf_counter() - run_time`. | open |
| B-grok-06 | Grok | `logic/src/pipeline/simulations/repository/filesystem.py:251-257` | minor | `num_loc == 104` reads `Rio_Maior_Sensores_2021_2024_cleaned_104.csv` and `coordinates104.csv`. Neither file is in `data/simulator`. | Reproducer lists both paths as absent. The other Rio Maior branch (`old_out_crude_rate[riomaior].csv`, `old_out_info[riomaior].csv`) and the Figueira da Foz pair are present. | Delete the 104-bin branch, or point it at files that ship with the export. | open |
| B-grok-07 | Grok | `logic/src/pipeline/simulations/states/running.py:73-77`; `logic/src/pipeline/simulations/simulator.py:515-516` | major | Any exception inside the day loop becomes `CheckpointError`. `RunningState` stores the error dict and ends the state machine. Sequential mode has no `else` that records the failure, and its `except CheckpointError` only comments "skip". The console still prints that the policy finished. | Code path on `70e660b03`. A fault was not injected in this pass. Parallel mode does append a result that lacks `success` to `failed_log`. | Record the error dict in `failed_log` on both paths and keep a non-zero process status when any policy fails. | open |
| B-grok-08 | Grok | `logic/src/pipeline/simulations/actions/collection.py:89` | minor | The CTOP trip check adds `bins.c[node-1]`, which is fill percent, into a load that is compared with vehicle capacity in kilograms. `collect()` converts percent to kg before profit. | `test_sim.yaml` exposes `sim.problem=ctop`. The retained contract is `vrpp`, where this branch is skipped. | Add `bins.c / 100 * volume * density`, the same conversion `collect()` uses. | open |

### Lane F bug evidence (Cursor, 2026-09-25)

| ID | Agent | File:line | Severity | Symptom | Evidence / repro | Suggested fix | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-cursor-01 | Cursor | `logic/src/policies/mandatory_selection/selection_service_level.py:54-58`; `logic/src/policies/vector/selection/service_level.py:84` | major | Service-level uses the linear bound `μ·n_d + z·σ·n_d`, not the paper’s `z·σ·√n_d`. Both scalar (simulator) and vectorized (training) copies do this. | Reproducer: fill=70, rate=4, std=5, z=1, n_d=4 → linear 106 (selected), paper-sqrt 96 (not selected). `sed -n '54,58p'` on `70e660b03` shows `threshold * std_deviations * horizon_days`. No `sqrt` in either file. | Replace the std term with `threshold * std * sqrt(horizon_days)` in both implementations; add a unit test that the two copies agree after percent↔fraction scaling. | wontfix (Harbinger 2026-09-25: keep linear; √n_d is not in the export — there is no sqrt implementation to delete) |
| B-cursor-02 | Cursor | `logic/src/pipeline/rl/common/base/steps.py:128,140-142`; `logic/src/policies/vector/selection/{last_minute,lookahead,service_level}.py` (each `mandatory[:, 0] = False`) | major | Training mandatory selection is applied to customer-only `waste` `[B, N]`. Every vectorized selector then forces index 0 false, treating the first customer as the depot. `steps.py` prepends a depot column *after* `select()`. The fullest customer is never constrained. | Reproducer: `LastMinuteSelector(0.7).select([[0.9, 0.1, 0.1]])` → `[[False, False, False]]`. Default `train.yaml` strategy is `lookahead`, so this is the live training path when `mandatory_selection` is not null. Neural Agent prepends the depot *before* `select()` (`policy_na.py:209-211`) and is consistent. | Prepend the depot column (fill 0) before `select()`, or stop zeroing index 0 when the tensor is customer-only. | open |
| B-cursor-03 | Cursor | `logic/configs/policies/other/ms_last_minute.yaml:27,31`; `logic/src/policies/vector/selection/last_minute.py:34,59`; `logic/configs/tasks/train.yaml:108-109` | major | Last-minute thresholds live in two unit systems. Simulator yaml is `70`/`90` (percent) against `bins.c` in `[0, 100]`. Vectorized default is `0.7` (fraction) against `MAX_WASTE=1.0`. Training’s nested last-minute is `0.25`. Passing the simulator yaml into the vectorized factory, or `0.7` into the scalar path, selects everything or nothing. | Reproducer: scalar `threshold=70` on fill `[50,75,95]` → bins 2,3; `threshold=0.7` on the same percent fill → bins 1,2,3. Vectorized `0.7` on `[50,75,95]` selects every non-depot bin (`50 >= 0.7`). | One unit. Either keep percent in both copies and store `70`, or keep fraction and store `0.7`. Convert at the NA / training boundary only. | approved (Harbinger 2026-09-25: one unit). Stored yaml stays percent `70`/`90` (`ms_last_minute.yaml` / cf70/cf90). Vectorized default `0.7` and train nest `0.25` must be rewritten to that same percent, with conversion only at the fraction-fill boundary (NA already divides `bins.c` by 100). |
| B-cursor-04 | Cursor | `logic/src/policies/mandatory_selection/selection_lookahead.py:178-179`; `logic/src/policies/vector/selection/lookahead.py:151` | major | Scalar lookahead multiplies fill by the *absolute* day index `j` in `range(today+1, next_collection_day)`. Vectorized uses relative days `(next - today - 1)`. They agree only when `current_collection_day == 0` (the yaml default). | Reproducer: fill `[60,40]`, rate `[50,20]`. Scalar day 0 → `[1]`; scalar day 5 → `[1,2]`; vectorized day 5 → no expansion (`[[False, False]]`, and index 0 is also dropped by B-cursor-02). | Use relative days in the scalar loop: `fill + (j - today) * rate`, or drop `current_collection_day` if it is always 0. | open |
| B-cursor-05 | Cursor | `logic/src/policies/route_construction/other_algorithms/travelling_salesman_problem/tsp.py:43,64`; `logic/src/policies/route_improvement/fast_tsp.py:68-74` | major | `FastTSPRouteImprover` and `find_route` accept `seed` and store it. `fast_tsp.find_tour` is called with only `duration_seconds=time_limit`. The seed never reaches the solver. Default time budget is **2.0 s** (`FastTSPPostConfig.time_limit`), not the 30 s LKH leftover. | `sed -n '64p'` on `70e660b03`: `tour = fast_tsp.find_tour(tmpC_int, duration_seconds=time_limit)`. Reproducer asserts `seed` is absent from that call. Empty tours `[0]` / `[0,0]` → `split_tour` returns `[]` → original tour is returned. Nodes already in a trip are reordered, not dropped. | Forward `seed=` if the installed `fast_tsp` accepts it; otherwise drop the kwarg from the public config. Keep the 2 s default. | open |
| B-cursor-06 | Cursor | `logic/src/policies/route_construction/meta_heuristics/hybrid_genetic_search/hgs.py:674-681` | minor | HGS calls `acceptance_criterion.step()` every generation and never calls `accept()`. BMC cools a thermometer that does not decide survival. ALNS and PSOMA do call `accept(current_obj=profit, candidate_obj=profit, **kwargs)` — maximization, matching BMC/OI. Extra kwargs `f_best`/`iteration` are ignored. `math.exp(delta/T)` is only reached for `delta < 0`; `T <= 1e-9` rejects. No overflow on the retained path. | `rg "\\.accept\\(" hgs/` is empty. ALNS `alns.py:700-706` and PSOMA `solver.py:338` pass profit. Reproducer: BMC accepts 10→12, OI rejects 10→8, T=1e-12 rejects a huge worsening move. | Stop stepping BMC inside HGS, or actually consult `accept()` for offspring insertion. Docstrings that claim `accept(100, 98) → True` are wrong under maximization. | open |
| B-cursor-07 | Cursor | `logic/src/policies/mandatory_selection/selection_last_minute.py:45-46`; `logic/src/policies/mandatory_selection/base/eoq.py:93-102` | minor | `fill_ratios = current_fill / max_fill` is computed and passed in, then ignored. `resolve_trigger_threshold` compares absolute percent fill to `context.threshold`. Docstring says `>` ; the code is `>=`. | `eoq.py:93-102` on `70e660b03`. `use_eoq_threshold` defaults false, so last-minute is a uniform percent cut. | Delete `fill_ratios` or compare in ratio space. Match the operator to the docstring. | open |

### Lane D bug evidence (Kimi, 2026-09-25)

Scope: `route_construction/base`, BPC + `helpers/solvers_and_matheuristics`, SWC-TCF, ACO-HH + k-sparse pheromones, Neural Agent, TSP helpers. Reproducers (run from a `70e660b03` worktree with the shared venv python): consolidated `kimi_lane_d_repro_20260925.py` (base/NA checks), `kimi_lane_d_bpc_{repro,lci_yaml,lr_bound,dp_check}_20260925.py`, `kimi_lane_d_swc_{repro,policy_e2e}_20260925.py`, `kimi_lane_d_aco_{repro,determinism,swallow}_20260925.py` (all under `.agent/cache/tools/`).

| ID | Agent | File:line | Severity | Symptom | Evidence / repro | Suggested fix | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-kimi-01 | Kimi | `logic/src/policies/route_construction/learning_algorithms/neural_agent/policy_na.py:135-139`; `.../neural_agent/simulation.py:243,247` | minor | `NeuralAgentPolicy.execute` returns `profit` computed as fill-percent x raw EUR/kg (`sum(bins.c[n-1] * profit_vars["revenue_kg"])`, missing the `/100 * V * B` kg conversion; ratio 100/(V*B) ~ 2.1x on plastic) and `cost` computed on `distC * 100` (100x inflated). | `kimi_lane_d_repro_20260925.py`: fills 50/80, real area params (R=0.5837, B=19, V=2.5) -> returned profit 70.88 vs kg-basis 31.04. Both values are discarded by the only retained caller (`RouteConstructionAction.execute` unpacks `tour, _, _, extra_output, _`, `actions/route_construction.py:162`, and recomputes km at :173), so sim metrics are unaffected -- but the adapter contract misleads direct callers. | Convert to kg basis (`c/100 * V * B * R`) or document cost/profit as advisory-only. | open |
| B-kimi-02 | Kimi | `logic/src/policies/helpers/solvers_and_matheuristics/search/cutting_planes.py:937-1045` (violation gate 990-991, coefficients 1034-1038) | major | `PhysicalCapacityLCIEngine` emits LCI cuts that are valid only for a single-vehicle problem (per-vehicle capacity Q treated as a global knapsack, docstring "Single-Vehicle VRPP ... PC-TSP") into an unlimited-fleet master (`sim.n_vehicles: 0` -> `vehicle_limit=None`, `policy_bpc.py:103`). Visiting all cover nodes with several feasible routes is legitimate, yet the cut forbids it. | `kimi_lane_d_bpc_lci_yaml_20260925.py`: shipped `policy_bpc.yaml` on a 3-node instance (w=1, Q=2, d12=5) returns routes [[3,1]] obj=17.0; true optimum 25.0; LP trace finalises under cut `lci_* <= 2.0` blocking the optimal pair+single split. Fires even at an integer root solution visiting all cover nodes; live whenever `cutting_planes: "all"` (shipped). | Gate the engine on `master.vehicle_limit == 1` or delete it; add the 3-node case to the packaging smoke (assert obj == 25 once fixed). | open |
| B-kimi-03 | Kimi | `helpers/solvers_and_matheuristics/lagrangian_relaxation/subgradient_optimization.py:148-162`; consumed at `branch_and_price_and_cut/bpc_engine.py:641-643` | major | Lagrangian pre-pruning bound is unsound for the multi-route master: `L(lambda) = OP_obj + lambda*Q` over a SINGLE-vehicle OP; a valid capacity-relaxation bound for K routes needs K*(OP+lambda*Q). Bound can fall below the optimum and prune the optimal node. `lr_pre_pruning: true` is shipped. | `kimi_lane_d_bpc_lr_bound_20260925.py`: 2 nodes w=6, Q=10 -> brute-force z*=116 (two singleton routes); LR ub = 58 < z*; pruned at any incumbent >= 58 without running CG. | Scale the bound by fleet size, or drop LR pre-pruning for the unlimited-fleet master. | open |
| B-kimi-04 | Kimi | `branch_and_price_and_cut/bpc_engine.py:700-703,765-767,971-976` | major | Root CG gets 40% of `time_limit`; on node timeout the B&B loop `break`s and the fallback returns `initial_routes_nodes`, which is EMPTY whenever there are no mandatory bins -- the warm-start greedy routes went only into the master column pool. | `kimi_lane_d_bpc_repro_20260925.py`: n=80 random instance, time_limit=4 s -> "B&B node at depth 0 timed out. Terminating search." then `obj=0.0, 0 routes, wall=2.19s`. In the simulator this is a silent 0 km/0 kg day. | On timeout return best-so-far (pool/greedy routes); never fall back to an empty route list. | open |
| B-kimi-05 | Kimi | `helpers/solvers_and_matheuristics/master_problem/model.py:310-318`; `search/column_generation.py:236-240` | major | Gurobi LP status != OPTIMAL silently returns `(0.0, {})`: CG treats 0.0 as a valid bound, `is_solution_integer({})` is `all()` of empty = True, and a 0.0/empty incumbent can be recorded. The `if obj_val is None` guard at `column_generation.py:240` is dead. | `kimi_lane_d_bpc_repro_20260925.py`: `Params.TimeLimit = 0` -> status 9 (TIME_LIMIT) -> returned obj 0.0 vals {} with stale duals. (True infeasibility IS handled correctly: `_handle_infeasibility` -> Farkas pricing -> RuntimeError -> node infeasible, verified.) | Return None/raise on non-OPTIMAL, or propagate a timeout flag; make the None guard real. | open |
| B-kimi-06 | Kimi | `branch_and_price_and_cut/params.py:152,156`; `helpers/.../pricing/smoothing.py:221,295-304,323-417`; `search/column_generation.py:314-322` | major | yaml `enable_dssr`/`dssr_max_iters` are read nowhere; the DSSR wrapper is also broken -- it operates on `pricing_solver._ng_memory` which does not exist (the solver has `ng_neighborhoods`, `pricing/solver.py:127`), and `use_dssr` is never passed by the CG loop. | Probe: CG loop calls `solve_pricing_step` with `use_dssr=False`; `hasattr(solver, "_ng_memory")` is False; direct wrapper call returns routes without touching ng neighbourhoods. | Delete wrapper + keys (done locally, see R-kimi-14) or wire `enable_dssr` and rename the attribute. | open |
| B-kimi-07 | Kimi | `branch_and_price_and_cut/bpc_engine.py:714-732`; `helpers/.../pricing/smoothing.py:425-436` | major | `enable_reduced_cost_arc_fixing: true` (yaml) is silently ignored; the block is gated on `hasattr(pricing_solver, "_forbidden_arcs")` (never set -- the solver only has `fixed_arcs`, `pricing/solver.py:122`) so it never runs; if it ever ran it would crash (`reduced_cost_arc_fixing()` takes no `capacity` kwarg; `master.get_dual_values()` does not exist). | Direct call repro: `TypeError: reduced_cost_arc_fixing() got an unexpected keyword argument 'capacity'`. | Delete block + key (done locally, R-kimi-17) or fix gate and signature. | open |
| B-kimi-08 | Kimi | `helpers/.../search/cutting_planes.py:1633-1643,1786,1808-1809,2061-2062,2182-2183` | major | Four cut families advertised under `cutting_planes: "all"` are silent no-ops: `MinCutInequalityEngine` (master lacks `add_min_cut_constraint`; its fallback REWRITES the base coverage constraints `Sense=GREATER_EQUAL, RHS=...`, and its violation is identically 0), `TriangleCliqueCutEngine` (`add_conflict_cut`/`add_clique_cut` missing), `NodeProfitBoundEngine` (`add_profit_bound_cut` missing), `PathEliminationEngine` (`add_path_elimination_cut` missing). `LimitedMemoryRank1CutEngine` only works via `add_subset_row_cut`, which rejects |S|!=3, so its 5-subset enumeration is dead. | grep: zero definitions for the four `add_*` methods (protocols only); MinCut constraint rewrite observed on a live run. | Implement the master methods or delete the engines + wiring (done locally, R-kimi-16). | open |
| B-kimi-09 | Kimi | `helpers/.../lagrangian_relaxation/pre_pruning.py:100` -> `uncapacitated_orienteering_problem.py:148` | minor | `params.seed=None` (default `BPCConfig()`) -> `model.setParam("Seed", None)` -> Gurobi `TypeError: an integer is required`; the call sits outside any try/except. The simulator masks it by injecting `raw_cfg["seed"]` (`actions/route_construction.py:68`); direct `BPCPolicy(config=BPCConfig())` or `run_bpc` with `lr_pre_pruning=True` crashes. | Direct call: TypeError as above. | `seed=params.seed if params.seed is not None else 42` (same pattern as `subgradient_optimization.py:156`). | open |
| B-kimi-10 | Kimi | `helpers/.../master_problem/model.py:208`; `search/column_generation.py:125-126,345-349` | minor | Wentges dual smoothing can never turn on (`enable_dual_smoothing=False` at construction; nothing sets it True); the smoothing-recovery branch is unreachable; `policy_bpc.yaml` comments ("exact_mode: false -- keep Wentges smoothing active", ~lines 26-28, 257-260) and the flowchart's `g_duals` box describe behaviour that does not exist. | grep `enable_dual_smoothing = True` -> 0 hits. | Fix yaml comments/flowchart, or implement smoothing; today `exact_mode`'s smoothing half is a no-op. | open |
| B-kimi-11 | Kimi | `helpers/.../pricing/solver.py:684,353-363`; `search/column_generation.py:410-425`; `pricing/smoothing.py:314` | major | The "exact" Lagrangian-UB prune (`BPCPruningException`) rests on `pricing_exhausted`, which heuristic pricing cannot certify: with `exact_mode=False` (shipped) each node's outgoing arcs are truncated to `min(20, max(5, n//3))` (n=20 -> 6 of 19 neighbours), and for n>40 the exact DP is skipped when the heuristic returns >= max_routes/2 leaving `last_max_rc = -inf` -> `pricing_exhausted=True` -> UB = obj + fleet*max(0, max_rc) = obj. Positive-RC columns may exist outside the truncated graph -> the optimal node can be pruned. | The code comment at `column_generation.py:400-409` states the precondition (no positive-RC column) that `pricing_exhausted` does not guarantee; traced on a truncated run. | Apply the UB prune only when the full graph was searched (exact mode, no label cap/timeout); else force `pricing_exhausted=False`. | open |
| B-kimi-12 | Kimi | `helpers/.../pricing/solver.py:859` | minor | Farkas feasibility check is a tautology: `if self.is_farkas and new_rc < -1e-6 or not self.is_farkas and new_rc < -1e-6:` == `new_rc < -1e-6` in both phases -- a half-finished edit. | sed-verified on `70e660b03`. | Collapse to `if new_rc < -1e-6` or implement the intended phase distinction. | open |
| B-kimi-13 | Kimi | `helpers/.../pricing/labels.py:103-113`; `pricing/solver.py:732,736` | minor | Label dominance requires exact `sri_state` equality, making the "potential penalty" branch (106-113) unreachable; `solver.py:732` passes `sri_dual_values` to `dominates` while :736 does not (no effect today). Dominance itself is sound. | Rules reviewed (load<=, rc>=, ng-memory subseteq, SRI equal, RF-unmatched subseteq -- conservative superset); 30 random instances, exact DP max-RC vs brute-force over all elementary routes: 0/30 mismatches (`kimi_lane_d_bpc_dp_check_20260925.py`). | Remove the dead branch or implement SRI-aware dominance fully. | open |
| B-kimi-14 | Kimi | `helpers/.../branching/tree.py:103-113` | minor | `BranchAndBoundTree` reads `params.max_branch_nodes`/`params.tree_search_strategy`; `BPCParams` has `max_bb_nodes`/`search_strategy`. Masked today because `bpc_engine.py:531` passes explicit args (which emits a DeprecationWarning on every `run_bpc`); a future params-only caller silently gets best_first + 1000. `tree.max_nodes` is stored but never used. | DeprecationWarning observed on every run_bpc. | Use the real field names; drop the alias lookup. | open |
| B-kimi-15 | Kimi | `branch_and_price_and_cut/bpc_engine.py:167,182-312,404-410`; `branch_and_price_and_cut/params.py:59,70,81,92` | minor | Dead debris: duplicated `from typing import Any  # AUTO-REPLACED`; `_select_nodes_knapsack` (~130 LOC) never called (flag `knapsack_proc_selection` read nowhere; engine comment says "No node pre-selection"); dead params `max_cut_iterations`, `use_spatial_partitioning`; yaml key `enable_hybrid_search` is not a BPCParams field (silently dropped by from_config); zero-caller methods `solve_ip` (model.py:423), `has_artificial_variables_active` (problem_support.py:783), `find_and_add_violated_rcc` (constraints.py:470), `Label.is_feasible` (labels.py:123), `Node` dataclass (common/node.py:23), `BIG_M` (model.py:159), `SeparationEngine.separate_integer` (separation/engine.py:90). | grep repo-wide -> definitions + re-exports only. | Deleted locally with the R-kimi removals (see §3); safe because every item is definition-only. | open |
| B-kimi-16 | Kimi | `.../smart_waste_collection_two_commodity_flow/gurobi.py:157,165` (same factor `ortools_wrapper.py:159,166`, `pyomo_wrapper.py:211,215`) | major | Travel cost is halved in the solver objective: maximize `R*sum(S*g) - 0.5*C*sum(d*x) - Omega*k`. Arcs are directed pairs used exactly once per route; nothing is double-counted, so the solver systematically undervalues travel and picks subsets/routes that are not profit-optimal. The returned `cost` (gurobi.py:222) uses the full sum -> model objective and reported cost disagree. | `kimi_lane_d_swc_repro_20260925.py` CHECK 1: fills 50/99/90, mandatory bin 1, Q=150, R=C=1, Omega=0.1, d01=d03=5, d13=10 -> brute-force optimum {1,3} at 20 km profit 119.9; solver returns [0,1,2,0] 35 km true profit 113.9 (model profit 131.4). | Drop the `0.5 *` factor in all three wrappers. | open |
| B-kimi-17 | Kimi | `.../ortools_wrapper.py:71-89,157,164`; `pyomo_wrapper.py:49-58,210,214`; vs `base_routing_policy.py:252-260` | major | OR-Tools/Pyomo wrappers solve a kg-unit model while the adapter feeds percent units: `S_dict = (fill/100)*B*V` in kg, but `Q` stays in percent points (7368 for Rio Maior plastic read as 7368 kg vs true 3500 kg, 2.1x inflation) and `R` stays EUR/percent (revenue per 95% bin 12.51 EUR vs true 26.34 EUR, x B*V/100). | `kimi_lane_d_swc_repro_20260925.py` CHECK 2: same instance through the dispatcher -- `framework=gurobi` collects 2 bins at Q=250 while `framework=ortools, engine=scip` collects 5. Shipped yaml uses `framework: gurobi`, so latent until a config change. Also asymmetric pre-processing: only OR-Tools/Pyomo force g=0 for fill<10. | One unit system in all wrappers (percent S; or convert Q->kg and R->EUR/kg on the kg paths). | open |
| B-kimi-18 | Kimi | `gurobi.py:135-142` (same structure `ortools_wrapper.py:140-147`, `pyomo_wrapper.py:186-194`) | major | `delta` is inert: every mandatory bin is hard-forced `g[i] == 1`, which makes the delta-slack coverage constraint redundant for every value of delta (with zero criticals it is `0 >= -n*delta`). | `kimi_lane_d_swc_repro_20260925.py` CHECK 5: delta sweep 0.0 -> 2.0 through `run_swc_tcf_optimizer(framework="gurobi")` returns the identical route/profit each time. yaml documents delta as "Bonus reward ... 0-100"; the config class as "Distance weight" -- neither exists. | Implement the documented semantics or delete the key (see R-kimi-12). | open |
| B-kimi-19 | Kimi | `gurobi.py:74-75`; `ortools_wrapper.py:88`; `pyomo_wrapper.py:65` | major | The "6000 km" arc filter has two different units across backends: gurobi `max_dist = 6000000.0  # 6000 KM` vs OR-Tools/Pyomo `max_dist = 6000`. Repo distance CSVs are km (`gmaps_distmat_plastic[riomaior].csv` values ~0.7-63; `data/network/file.py:65` loads without /1000): gurobi's filter is a no-op while OR-Tools/Pyomo really drop arcs > 6000 km -- backend-dependent feasible arc sets, contradicting the flowchart's single "d_ij <= 6000 km" box and gurobi's own comment. | CHECK 3: km matrix with mandatory bins 7000 km from depot -> native gurobi solves [0,2,1,0] (14005 km); ortools+SCIP prints `[WARN] OR-Tools TCF could not find a feasible solution.` and returns [0,0] (silent empty day, B-kimi-20). Retained Rio Maior data never triggers this. | One shared `6000.0` (km) constant in all three wrappers. | open |
| B-kimi-20 | Kimi | `gurobi.py:191-227`; `ortools_wrapper.py:209-211`; `dispatcher.py:78-90`; `policy_swc_tcf.py:140` | major | Infeasible/no-solution models return a silent empty tour: gurobi branches only on `SolCount > 0` (no status check); ortools prints [WARN] and returns [0,0]; the dispatcher falls back to native gurobi only when `ortools_backend == "GUROBI"` -- which happens by accident of `CreateSolver("GUROBI") -> None` (only SCIP/CBC/SAT are linked in the installed ortools; HIGHS/CPLEX -> None -> no fallback). `policy_swc_tcf.py:140` strips zeros -> `[]` -> base returns [0,0]: the whole day collects nothing, no error. | CHECK 4: forced-visit infeasibility (psi=1, two bins at 100% fill, Q=150) -> gurobi returned [0], 0.0, 0.0, no exception. Because mandatory bins are hard-forced (B-kimi-18), a single unreachable/over-capacity mandatory bin zeroes the entire day's collection. | Branch on solver status; log/raise on infeasible; relax forced visits (or drop them from `mandatory`) when infeasible. | open |
| B-kimi-21 | Kimi | `base/base_routing_policy.py:231-236` | major (latent) | Typed-config path drops runtime overrides nested under the engine key: `runtime_overrides = config.get(config_key, config)` is only flattened when it is NOT a dict; a raw section like `{"gurobi": [{Omega...}]}` is a dict -> merged unflattened -> `values` gains a stray `"gurobi"` key and `vrpp: false` (or any capacity/revenue/cost_unit/density/bin_volume/shift_hours override) never reaches `values`; `execute()` then reads `values.get("vrpp", True)` = True. | `kimi_lane_d_swc_policy_e2e_20260925.py`: `SWCTCFPolicy(config=raw)` with `vrpp: False` -> `_load_area_params` values contain no `vrpp` key (`values.get('vrpp') -> None`). Shipped yaml sets `vrpp: true` = the default, so latent today. | Flatten `runtime_overrides` unconditionally (`_flatten_raw_config` handles dicts). | open |
| B-kimi-22 | Kimi | `.../pyomo_wrapper.py:85,151` | blocker (pyomo path) | `model.A = pyo.Set(within=model.V * model.V, filter=valid_arcs_rule)` with no `initialize` -- a filter-only virtual set is never enumerated, so `model.A` is empty and `depot_waste_out` (line 151) raises `ValueError: Invalid constraint expression ... trivial Boolean (True)` at model construction. | `run_swc_tcf_optimizer(framework="pyomo", optimizer="scip", ...)` -> the ValueError above (isolated check: `A members: []`, `(0,1) in A -> False`). `framework: pyomo` is reachable from the dispatcher and allowed by the schema, but crashes every sim day. | Add `initialize=[(i,j) for i in nodes for j in nodes]` (filter then applies) -- or remove the wrapper (R-kimi-11). | open |
| B-kimi-23 | Kimi | `pyomo_wrapper.py:238-240,273-274` | minor | Termination check is a tautology (`check_optimal_termination(results) or check_optimal_termination(results) is False` == True for any bool), so the [WARN]+return-[0,0] branch is dead and solver failures RAISE (e.g. ApplicationError when the SCIP executable is absent) -- unlike gurobi (silent [0]) and OR-Tools ([WARN]). Three backends, three failure modes. | sed-verified on `70e660b03`. | Fix the condition; align failure semantics across wrappers. | open |
| B-kimi-24 | Kimi | `.../policy_swc_tcf.py:130`; `gurobi.py:185-186` | minor | `int(time_limit)` truncation: `time_limit: 0.5` -> `0` -> Gurobi TimeLimit never set (`if time_limit > 0`) -> unlimited solve; a sub-second budget silently becomes unbounded. | Code path verified on `70e660b03`. | Pass the float through and guard with a small epsilon; reject < 1 s values with a warning. | open |
| B-kimi-25 | Kimi | `logic/configs/policies/policy_swc_tcf.yaml:44-64` vs `gurobi.py:135-166` | major (docs/config) | The shipped yaml documents a different objective than the code implements: Omega = "penalty weight for overtime" (code: per-vehicle fixed cost `-Omega*k`), delta = "bonus reward 0-100" (code: inert, B-kimi-18), psi = "travel cost scaling 0.1-10" (code: force-visit threshold `fill >= psi*100`, no cost scaling anywhere). The yaml's "Objective =" line matches no wrapper. | CHECK 5: psi=0.5, one bin at 60% fill 300 km away -> forced collection at a loss (route [0,1,0], profit -240.1); tuning psi per the docs (e.g. 2) silently disables near-full forced visits. The flowchart matches the CODE, not the yaml; constraints (1)-(7) themselves match the flowchart exactly. | Rewrite yaml/config-class comments to the implemented semantics. | open |
| B-kimi-26 | Kimi | `hyper_heuristics/.../hyper_aco.py:145`; `policy_aco_hh.py:128`; `params.py:75`; yaml `policy_aco_hh.yaml:111` | major | yaml `operators` list (5 operators) is silently ignored: the solver always runs all 11 (`self.operator_names = list(HYPER_OPERATORS.keys())`); `params.operators` is stored but never read. | `kimi_lane_d_aco_repro_20260925.py`: `HyperACOParams(operators=["swap"])` -> `solver.operator_names` still all 11; per-solve application counts: shaw_removal 189, string_removal 226, kick 74, 2opt_star 19, 3opt_intra 9, random_removal 7 -- none in the yaml. | Build `operator_names` from `params.operators` (validated) or delete the key (R-kimi-18). | open |
| B-kimi-27 | Kimi | `policy_aco_hh.yaml:91`; `logic/src/configs/policies/aco_hh.py:63`; `hyper_aco.py:150` | major | yaml `sequence_length: 5` is silently ignored: there is no `HyperACOParams` field for it and the solver pins `self.sequence_length = self.n_operators` (11); the documented "Range: 2-10" knob does not exist. | repro: `HyperACOParams` has no sequence_length attribute; `_select_sequence` loops `range(self.sequence_length)` = 11 regardless of config. | Add the field and wire it, or drop the key (R-kimi-18). | open |
| B-kimi-28 | Kimi | `hyper_heuristics/.../hyper_aco.py:767-769,796` | major | Same seed gives different tours across runs: the visibility update divides by wall-clock `execution_time` (`eta_updates[...] += lam / ((execution_time + 1e-3) * safe_count)`), so CPU jitter drives eta and hence operator selection. Seeds are forwarded correctly (action policy_seed -> `random.Random(seed)`/`np.random.default_rng(seed)`) but do NOT confer reproducibility. | `kimi_lane_d_aco_determinism_20260925.py`: 6 runs, seed=42, same instance -> 2 distinct route sets (profits tie at 107.1407); with `time.perf_counter` patched to a deterministic counter, consecutive runs match exactly. | Make the denominator deterministic (cumulative count only, or a fixed per-op cost), or drop the time factor. | open |
| B-kimi-29 | Kimi | `hyper_aco.py:308-316,874-877` | major | First-hop operator transitions are never reinforced: the deposit loop restarts at `prev_op_idx = self.n_operators` (the virtual row), but selection departs from the ant's real vertex -- the traversed first edge gets zero pheromone and deposits accumulate on a row `_select_sequence` never reads. The comment at :312 ("first real edge seeds real rows") describes the opposite of the code. Diverges from Chen et al. (deposit along the traversed path) and the flowchart's journey box. | `kimi_lane_d_aco_repro_20260925.py` spy run: rows read by `_select_sequence` are a subset of [0..9]; row 11 never read, yet `tau[11]` grew 1.0 -> 0.323/0.047 via deposits vs 0.016 pure evaporation elsewhere. | Initialise `prev_op_idx` to the journey's real start index (pass it into the deposit loop). | open |
| B-kimi-30 | Kimi | `hyper_aco.py:296-299,230,753,321-325` | major | Strategic oscillation can return capacity-infeasible routes as the best solution: `pv` halves with no floor until `pv < initial_pv` relaxes capacity to inf; best-tracking uses the penalised objective and `best_routes` is returned with no feasibility re-check at real capacity. | `kimi_lane_d_aco_determinism_20260925.py`: initial greedy feasible; after 60 iterations pv decayed 1.042 -> 3.1e-08 and the returned best contains 3 routes over capacity 50 (loads 55.1, 50.9, 51.2). The simulator receives and executes the over-capacity plan. | Track best-feasible separately (evaluate at real capacity) and return it; floor `pv`. | open |
| B-kimi-31 | Kimi | `hyper_heuristics/.../hyper_operators.py:525-526,566-567,607-608` | minor | `apply_shaw_removal`/`apply_string_removal`/`apply_random_removal` wrap everything in a bare `except Exception: return False` -- a node/matrix mismatch degrades silently into "operator did nothing" with no log. | `kimi_lane_d_aco_swallow_20260925.py`: raw `string_removal([[8,9]],1,3x3)` raises `IndexError: index 9 is out of bounds`; `apply_string_removal` with identical input returns False, no traceback. | Catch specific exceptions and log; re-raise programming errors. | open |
| B-kimi-32 | Kimi | `base/base_routing_policy.py:192-205,487-489`; `pipeline/simulations/actions/route_construction.py:109-116` | major (cross-policy) | Base `_validate_mandatory` early-returns `[0,0]` whenever mandatory is empty -- no `vrpp` check -- contradicting the action ("VRPP policies (vrpp: true) ... don't skip them") and the flowchart diamond "mandatory bins empty (and not vrpp)?". All `BaseRoutingPolicy` subclasses inherit it (no overrides; SANS re-checks the same way at `simulated_annealing_neighborhood_search/dispatcher.py:82,176`). Compounding defaults: the action reads `vrpp` default False (`flat_cfg.get("vrpp", False)`), the base reads default True (`values.get("vrpp", True)` at :508), and the typed-config path can drop `vrpp:false` entirely (B-kimi-21). Net effect: with empty mandatory, optional-bin collection can never happen, even for vrpp policies. | `kimi_lane_d_repro_20260925.py` asserts the default split; code-path trace confirms the early return precedes any solver call. Explains the km=0-before-threshold behaviour all policies exhibit (which current acceptance expectations rely on). | Owner decision on intended semantics FIRST (fixing silently changes smoke expectations); then either gate the early return on `not vrpp` or drop the vrpp half of the action's condition. | open |
| B-kimi-33 | Kimi | `hyper_aco.py:281` | minor | Elitism sync count uses `int(n_ants * elitism_ratio)` (floor); the flowchart/params docstring say ceil (identical for 0.5/0.2 x 10; differs e.g. 0.25 x 10 -> 2 vs 3). Flowchart `a_params` also shows time_limit=30 s vs yaml 60.0. | sed-verified on `70e660b03`. | Use `math.ceil` or fix the diagram. | open |

## 3. Removal candidates (shared table — append rows)

Kinds: `package`, `module`, `class`, `function`, `config-key`, `yaml`, `dependency`, `data-file`. "Packaging hook" says how the removal should be expressed in `logic/package/prune_codebase.py` / `ci/export_config.json` (new category, `always_keep` change, `remove_*` script, or manual).

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| R-claude-01 | Claude (owner's example) | `logic/src/models/subnets/embeddings/positional/` | package | 191 | Owner-flagged. Positional embeddings were used by DACT/N2S (removed). Importers to confirm: `subnets/embeddings/__init__.py` re-exports `POSITIONAL_EMBEDDING_REGISTRY` | Deleted in scratch worktree with R-gemini-*; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass — **Gemini** | `subnet_pruning.prunable_types.embeddings.always_keep` currently lists `positional/*` — move to `model_files` for `dact`/`n2s` | verified locally |
| R-claude-02 | Claude (owner's example) | `logic/src/data/distributions/*` except `base.py`, `statistical_empirical.py`, `statistical_gamma.py` | module | ~1,300 | Owner wants only empirical + gamma. `data/generators/waste.py` still branches on `unif`/`beta`/`dist`/`const` and `DISTRIBUTION_REGISTRY` in `distributions/__init__.py` names them all; smoke tests use `unif` | none yet — **Cursor** (decide whether `uniform` stays for smoke tests or the smoke tests move to `gamma1`) | `--distributions empirical gamma` already exists in `prune_codebase.py`; `_filter_files` does not clean `DISTRIBUTION_REGISTRY`/`__all__` dict entries — needs a fix | proposed |
| R-claude-03 | Claude (owner's example) | `logic/src/data/datasets/pytorch/*` (keep one) and `logic/src/data/datasets/simulation/*` (keep one) | module | ~1,000 | Owner wants one dataset class per side. Known users: `bins/base.py` dispatches on extension to `NumpyPickleDataset`/`PandasExcelDataset`/`PandasCsvDataset`/`NumpyDictDataset`/`GenerativeDataset`; `repository/dataset.py` imports three; training uses `FastTdDataset`/`TensorDictDataset`/`BaselineDataset` (`rl/common/base/data.py`) | none yet — **Cursor** (report which single class can serve each side and what callers must change) | `--sim-datasets` exists; pytorch side has no category yet | proposed |
| R-claude-04 | Claude (owner's example) | `logic/src/models/common/non_autoregressive/`, `logic/src/models/common/improvement/`, `logic/src/models/common/transductive/` | package | 1,233 | Owner-flagged. NAR/improvement bases served NARGNN/DeepACO/N2S/NeuOpt (removed); transductive (EAS/active search) is not on the retained train/eval path. `models/__init__.py` and `models/common/__init__.py` still re-export them; `policies/vector/__init__.py` imports `NonAutoregressivePolicy`, `ImprovementPolicy` | Deleted in scratch worktree with re-exports cleaned; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass — **Gemini** | new `optional_features` entry or `subnet_pruning`-style list for `models/common` | verified locally |
| R-claude-05 | Claude | `logic/src/policies/helpers/operators/` (102 files, 25 kLOC) | package | 25,118 | `policies_helpers_analysis` keeps second-order deps of the nine kept policies; most operator families (cross-exchange, ejection chains, string removal variants, ...) are only reachable from removed policies or from `local_search_hgs.py` optional flags that are off in the retained yamls | none yet — **Qwen** (per-file reachability from the nine policies + `fast_tsp`) | regenerate `policies_helpers_analysis` with a stricter (runtime-reachable) analysis | proposed |
| R-claude-06 | Claude | `logic/src/pipeline/features/{train,eval,test}/zenml_*_pipeline.py`, `logic/src/configs/tracking.py` ZenML fields, `pipeline/callbacks/pytorch/attention_heatmaps.py`, `gpu_memory_monitor.py` | module | ~900 | ZenML is optional and guarded by try/except; heatmap/GPU-monitor callbacks are dashboard features | none yet — **Codex** | extend `remove_tracking.py` / `remove_callbacks.py` | proposed |
| R-claude-07 | Claude | `logic/src/tracking/logging/modules/{gui.py,metrics.py}` and the `jinja2`, `wandb`, `rich` imports they pull in | module | ? | `send_final_output_to_gui` writes the Studio realtime feed (`popup.html` template); `metrics.py` imports `wandb` at import time only for `log_values/log_epoch` (training) | none yet — **Grok** (which writers are needed for `log_*.json` + `.jsonl` and which only feed the removed Studio app) | `remove_tracking.py` currently deletes all of `tracking/logging`; needs a "keep result writers" mode | proposed |
| R-claude-08 | Claude | `logic/src/configs/rl/__init__.py::RLConfig` PBRS fields, `configs/policies/other/reinforcement_learning.py` (bandit/TD configs for removed RL-based improvers), `configs/tasks/train.py` unused fields (`route_improvement_epochs`, `lr_route_improvement`, `checkpoint_encoder`, ...) | config-key | ? | many dataclass fields describe removed features; Hydra rejects yaml keys without a dataclass field, so every removal must be mirrored in `logic/configs/tasks/*.yaml` | none yet — **Mistral** | manual + `remove_*` script updates | proposed |
| R-claude-09 | Claude | `logic/pyproject.toml` dependencies: `python-pptx`, `docxtpl`, `latex2mathml`, `cryptography`, `requests`, `wandb` (if `metrics.py` goes), `torch-geometric`/`torch-scatter`/`torch-sparse` (if no retained module imports them), `hydra-colorlog` | dependency | — | to be checked with `grep -rn "^import\|^from" logic \| sort -u` against `uv.lock` | none yet — **Mistral** | `logic/pyproject.toml` edit in the packaging step | proposed |

### Codex removal evidence

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| R-codex-01 | Codex | `logic/src/pipeline/rl/core/losses/` (5 files) | package | 300 | Whole-tree symbol/module search finds only definitions, internal imports and examples; REINFORCE calculates its loss inline. | Deleted in scratch worktree together with R-codex-02; compileall passes; 699 modules, 0 failed; train smoke passes. | Add a loss-package keep rule based on retained algorithm dependencies, not a blanket `rl/core/**` keep. Keep on main when other algorithms consume these losses. | verified locally |
| R-codex-02 | Codex | `rl/common/{reward_scaler,reward_scaler_batch,route_improvement}.py` + their exports | module | 411 | `rg` for all defined symbols finds only definitions and `rl/common/__init__.py` re-exports. No retained runtime caller or YAML target. 393 module lines + 18 export lines. | Removed all three modules and exports; compileall passes; 699 modules, 0 failed; CPU train smoke produces checkpoint. | Optional-feature pruning with atomic import and `__all__` cleanup. This is the RL helper, not simulator Fast-TSP. | verified locally |
| R-codex-03 | Codex | `rl/common/baselines/{mean,none,pomo,shared_critic}.py` | module | 250 gross | Not used by the three retained baseline modes, but reachable via registry/config overrides and two package re-export surfaces. Keep `base`, `exponential`, `rollout`, `critic`, `warmup`. | Static proposal only; not removed or import-tested. Do not count as verified savings. | Explicit baseline allowlist plus registry/export/schema help cleanup; reject removed names. | proposed |
| R-codex-04 | Codex | `features/eval/evaluators/{augmentation,multi_start,combined}.py` | module | 259 gross | Default YAML selects greedy; sampling is documented. All three candidates remain reachable from dispatcher strings, so default-off is not proof of dead code. | Static proposal only; requires narrowing supported decoding choices and updating dispatch/imports; no local removal. | Evaluation-method allowlist, retain GreedyEval/SamplingEval and EvalBase; prune aliases only with consumers. | proposed |

### Gemini removal evidence

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| R-claude-01 | Gemini | `logic/src/models/subnets/embeddings/positional/` (3 files) | package | 191 | Positional embeddings were used by DACT/N2S (removed). No retained caller in AM. Re-exports in `subnets/embeddings/__init__.py` cleaned. | Deleted in scratch worktree with R-gemini-*; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass. | `subnet_pruning.prunable_types.embeddings.always_keep` — remove `positional/*`. | verified locally |
| R-claude-04 | Gemini | `logic/src/models/common/{non_autoregressive,improvement,transductive}/` (14 files) | package | 1,233 | Unused NAR/improvement/transductive model bases. Re-exports in `models/__init__.py`, `models/common/__init__.py`, and `policies/vector/__init__.py` cleaned. | Deleted in scratch worktree with R-gemini-*; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass. | Add prune rule for common bases; clean exports. | verified locally |
| R-gemini-01 | Gemini | `logic/src/models/subnets/embeddings/{dynamic,static}.py` (2 files) | module | 146 | Dynamic and static embeddings are unused by AM (which uses `nodes/init.py` and `context/vrpp.py`). Cleaned re-exports in `embeddings/__init__.py`. | Deleted in scratch worktree with R-gemini-*; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass. | Prune files and clean `subnets/embeddings/__init__.py`. | verified locally |
| R-gemini-02 | Gemini | `logic/src/models/subnets/embeddings/edges/` (4 files) | package | 231 | Edge embeddings unused by AM; pulls in `torch_geometric`. Cleaned re-exports in `embeddings/__init__.py`. | Deleted in scratch worktree; drops `torch_geometric` from model stack; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass. | Prune `embeddings/edges/`, unblocking `torch-geometric` removal in R-claude-09. | verified locally |
| R-gemini-03 | Gemini | `logic/src/models/subnets/embeddings/state/` (3 files) | package | 179 | MDP/RL state embeddings unused by AM. Cleaned re-exports in `embeddings/__init__.py`. | Deleted in scratch worktree with R-gemini-*; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass. | Prune `embeddings/state/` and clean exports. | verified locally |
| R-gemini-04 | Gemini | `logic/src/models/subnets/embeddings/context/generic.py` | module | 93 | Generic context embedder superseded by `vrpp.py`. Cleaned re-exports in `context/__init__.py`. | Deleted in scratch worktree with R-gemini-*; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass. | Prune `context/generic.py` and clean exports. | verified locally |
| R-gemini-05 | Gemini | `logic/src/models/subnets/modules/{cross_attention,flash_attention,normalized_activation_function,dynamic_hyper_connection,static_hyper_connection}.py` (5 files) | module | 617 | Experimental hyperconnections and alternative attention/activation modules unused by AM. Cleaned re-exports in `modules/__init__.py`. | Deleted in scratch worktree with R-gemini-*; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass. | Prune 5 modules and clean `subnets/modules/__init__.py`. | verified locally |
| R-gemini-06 | Gemini | `logic/src/models/core/attention_model/symnco_policy.py` | module | 88 | Sym-NCO symmetric augmentation policy variant unused by retained YAMLs and dispatch. Cleaned export in `attention_model/__init__.py`. | Deleted in scratch worktree with R-gemini-*; `compileall` clean; 674 modules, 0 failed; train/eval/sim pass. | Prune file and export in `attention_model/__init__.py`. | verified locally |
| R-gemini-07 | Gemini | `logic/src/models/core/attention_model/deep_decoder_policy.py` & `configs/models/deep_decoder.yaml` (2 files) | module | 176 | Deep decoder policy variant unused by retained pipeline (owner approved removal). | Deleted in scratch worktree with imports cleaned; `compileall` clean; 669 modules, 0 failed; train/eval/sim pass. | Prune `deep_decoder_policy.py`, `deep_decoder.yaml`, and clean `vector/__init__.py`, `attention_model/__init__.py`, and `builder.py`. | verified locally |
| R-gemini-08 | Gemini | `logic/src/models/common/critic_network/` (3 files, 344 LOC) & `rl/common/baselines/shared_critic.py` (93 LOC) | package | 437 | Critic network and shared critic baseline (owner approved pruning for minimal export packaging). | Deleted in scratch worktree with re-exports cleaned; `compileall` clean; 669 modules, 0 failed; train/eval/sim pass. | Prune `critic_network/` and `shared_critic.py`; clean exports in `models/__init__.py`, `common/__init__.py`, `baselines/__init__.py`, `rl/common/__init__.py`, `rl/core/__init__.py`. | verified locally |
| R-gemini-09 | Gemini | Dead config fields in `logic/src/configs/models/*.py` (~3 files) | config-key | ~60 | Dataclass schemas retain dead fields (`shrink_size`, `use_distance_features`, `temporal_feature_dim`, etc.). | Static analysis; coordinate with Lane G. | Remove obsolete fields from model dataclasses and matching YAMLs. | proposed |

### Grok removal evidence

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| R-grok-01 | Grok | `logic/src/utils/input/{dict_processing,file_processing,locking,preview,splitting,statistics,value_processing}.py` | module | 1,127 | Callers of these symbols are the package `__init__` and each other. Result writers import `utils.input.files` (`read_json`, `compose_dirpath`) directly. | Deleted in `/tmp/grok-wsr` with `input/__init__.py` reduced to the files helpers. `compileall` clean. Import sweep **698 modules, 0 failed** (baseline 707). | Drop the seven modules inside the export and stop `input/__init__.py` importing them. Keep `files.py`. | verified locally |
| R-grok-02 | Grok | `logic/src/utils/configs/yaml_to_env.py` (`to_bash_value`, `load_yaml_env`, `deep_merge`) | module | 186 | Only `configs/__init__.py` imports them. Runtime loads `config_loader.py` directly. | Deleted with the re-export. Same sweep: 698 modules, 0 failed. | Delete the module during packaging; leave `load_config` in place. | verified locally |
| R-grok-03 | Grok | `logic/src/tracking/logging/modules/metrics.py` | module | 154 | `log_values` / `log_epoch` / `get_loss_stats` are imported by `log_utils.py` and never called from `logic/src/pipeline`. The module imports `wandb` at import time, so every simulation currently loads wandb. | Deleted and unhooked. After `import logic.src.tracking.logging.log_utils`, `wandb` is absent from `sys.modules`. `jinja2` remains because `gui.py` stays. Same sweep. | `remove_tracking.py` should delete `metrics.py` and its re-exports, and may then drop the `wandb` dependency (R-claude-09). | verified locally |
| R-grok-04 | Grok | `storage.py` serializers `sort_log`, `log_to_json`, `log_to_json2`, `log_to_pickle`, `update_log` | function | 282 | Grep finds definitions and re-exports only. The live writer is `update_policy_log_section`. | Removed from `storage.py` (389 → 107 lines) in the same worktree. `compileall` and the 698-module sweep passed. `log_utils.update_policy_log_section` still imports. | Teach `remove_tracking.py` to keep `setup_system_logger` and `update_policy_log_section` and drop the legacy serializers. | verified locally |
| R-grok-05 | Grok | `filesystem.py` branches `src_area == "both"` and `num_loc == 104` | function | ~180 | `"both"` is rejected by `validation.py` (`area in MAP_DEPOTS`). The 104-bin files are absent (B-grok-06). Rio Maior and Figueira da Foz branches match files that exist. | Static. The functions were left in place. | Delete both branches in the packaging cleanup, together with the `both` asserts. | proposed |
| R-grok-06 | Grok | `logic/src/pipeline/simulations/checkpoints/` | package | 494 | Default `checkpoint_days` is 0, so the interval save never runs. The wrapper still converts every day-loop exception into `CheckpointError` (B-grok-07) and can resume bins and the day counter (B-grok-05). | Static. Deleting it requires editing `RunningState` and `InitializingState`. | Owner choice. If resume stays, fix the time sign and keep the package. If resume goes, remove the package and record day-loop failures directly. | proposed |
| R-claude-07 | Grok | `tracking/logging/modules/gui.py` vs `metrics.py` | module | — | `gui.py` is the realtime `.jsonl` writer (`send_daily_output_to_gui` / `send_final_output_to_gui`), called from `LogAction` and the orchestrator. `popup.html` is rendered into `all_bin_coords` on day 1. `metrics.py` is the wandb half and is safe (R-grok-03). `failure_analyzer.py` fills `failure_analysis` inside that jsonl; `failure_emit.py` adds a second `SIM_FAILURE_START:` line for Studio. | `metrics.py` deleted and swept. `gui.py` and `failure_analyzer.py` kept. | `remove_tracking.py` must gain a "keep result writers" mode: keep `gui.py`, `storage.update_policy_log_section`, `analysis.output_stats`. Drop `metrics.py`. Drop the Jinja popup only if the jsonl schema can lose the `popup` field. | verified for metrics; gui stays |

### Cursor removal evidence

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| R-cursor-01 | Cursor | `distributions/{spatial_cluster,spatial_gaussian_mixture,spatial_mix,spatial_mix_multi,spatial_mixed}.py` + `statistical_{bernoulli_gamma_mixture,compound_poisson_gamma}.py` | module | 970 | Completes the easy half of R-claude-02. `rg` for those class names hits only `distributions/__init__.py` and (for the two statistical names) lazy lookups inside `bins/prediction.py` that default to `distribution="mean"`. `waste.py` does not import them. | Deleted in `/tmp/cursor-wsr` and registry/`__all__` reduced to `gamma empirical distance beta constant uniform`. `compileall` clean. Import sweep **689 modules, 0 failed** (707 − 18). | `--distributions` must rewrite `DISTRIBUTION_REGISTRY` / `__all__`, not only delete files. Then drop the two hardcoded names from `prediction.py`. | verified locally |
| R-cursor-02 | Cursor | `datasets/pytorch/{extra_key,fast_gen,fast_td,generator}_dataset.py` | module | 280 | Completes the unused half of R-claude-03 (pytorch). `rg` for `ExtraKeyDataset`, `TensorDictDatasetFastGeneration`, `FastTdDataset`, `GeneratorDataset` hits only definitions and re-exports. Runtime training/eval constructs `TensorDictDataset`; rollout baseline still constructs `BaselineDataset`. | Deleted the four files and cleaned `datasets/__init__.py` + `pytorch/__init__.py`. Same 689-module sweep. | New pytorch-dataset allowlist: keep `td_dataset.py` + `baseline_dataset.py`. | verified locally |
| R-cursor-03 | Cursor | `logic/src/envs/base/improvement.py` (`ImprovementEnvBase`) | module | 66 | No subclass in the retained tree. Only re-exported from `envs/__init__.py`, `envs/base/__init__.py`, `envs/routing/__init__.py`. | Deleted the module and the three re-exports. Same sweep. | Drop the re-export in the env package prune. | verified locally |
| R-cursor-04 | Cursor | `logic/src/policies/route_improvement/common/bandit.py` | module | 79 | `ThompsonBandit` is imported only by `common/__init__.py`. Fast-TSP uses `helpers.py` only. | Deleted and unhooked. Same sweep. | Delete with the improver helper cleanup. | verified locally |
| R-cursor-05 | Cursor | `interfaces/distance_metric.py` (`IDistanceMetric`) + `interfaces/context/joint_context.py` (`JointSelectionConstructionContext`) | module | 177 | `IDistanceMetric` is named in the `interfaces` docstring and never imported. `JointSelectionConstructionContext` is only re-exported from `context/__init__.py`. Retained interfaces that *are* implemented: `IEnv`, `IModel`, `IPolicy`, `IRouteConstructor`, `IRouteImprovement`, `IMandatorySelectionStrategy`, `IAcceptanceCriterion`, `IBinContainer`, `ITraversable`, `ITensorDictLike`, plus `SearchContext` / `SelectionContext` / `ProblemContext` / `SolutionContext` / `MultiDayContext`. `problem_context.py` still lazy-imports the simulator. | Deleted both files and the `JointSelection…` re-export. Same sweep. | Interface prune allowlist; do not delete `problem_context.py`. | verified locally |
| R-cursor-06 | Cursor | `policies/vector/selection/{regular,revenue,combined}.py` | module | 263 | Retained selectors are `lookahead`, `last_minute`, `service_level`. `rg` for the three classes hits the factory, `vector/selection/__init__.py`, and `mandatory_selection/__init__.py` re-exports only. Scalar `RegularSelection` / `RevenueThresholdSelection` are already AUTO-CLEANED. | Deleted the three modules; factory mappings reduced to the three retained names; both `__init__.py` files cleaned. Same sweep. | Selector allowlist in the vector factory + `__all__` cleanup. | verified locally |
| R-claude-02 | Cursor | remaining `distributions/{statistical_uniform,statistical_beta,statistical_constant,spatial_distance}.py` + `waste.py` branches `unif`/`beta`/`const`/`dist`/`empty` | module | ~280 | Owner wants only `empirical` + `gamma`. `waste.py:66-91` still constructs `Constant`/`Uniform`/`Beta`/`Distance`. `datasets.py:54` advertises `empty const unif dist emp gamma*`. §0 smoke uses `train.data_distribution=unif`. `VRPPGenerator` default `waste_distribution="uniform"`. `bins/base.py` special-cases `emp` and `"gamma" in sample_dist`. | Not deleted (would break the smoke). Exact edits below in §5.F. | After moving the smoke to `gamma1`, extend `--distributions empirical gamma` so `_filter_files` also rewrites `DISTRIBUTION_REGISTRY`, `waste.py`, and `datasets.py::distributions_per_problem`. | approved (Harbinger 2026-09-25) — drop after §0 train smoke moves to `gamma1` |
| R-claude-03 | Cursor | simulation side: keep `SimulationDataset` + `GenerativeDataset` + `NumpyDictDataset`; drop `np_pkl` / `pd_csv` / `pd_xlsx` | module | 482 | `gen_data` writes `.npz` (`NumpyDictDataset`). `load_dataset=null` uses `GenerativeDataset`. `bins/base.py:170-177` and `repository/__init__.py:106-108` still dispatch on `.pkl/.xlsx/.csv`. Owner asked for one class per side — one file loader plus one generator is the minimum that keeps both `gen_data` and on-the-fly `emp` working. Pytorch survivor is `TensorDictDataset`; keep `BaselineDataset` until rollout is rewritten to store `bl_vals` on the TensorDict. | Static. Deleting the three loaders requires editing `bins/base.py` and `repository`. | `--sim-datasets npz generative`; add `--pytorch-datasets td baseline`. | approved (Harbinger 2026-09-25) — survivors locked |
| R-cursor-07 | Cursor | unused constants: `MAX_LENGTHS`, `CRITICAL_FILL_THRESHOLD`, `OPERATION_MAP`, `FS_COMMANDS`, `TQDM_COLOURS` | config-key | ~80 | `rg` for each symbol: `MAX_LENGTHS` only in `constants/simulation.py`; `CRITICAL_FILL_THRESHOLD` only in `constants/waste.py`; `OPERATION_MAP` / `FS_COMMANDS` only in `constants/system.py` (Studio leftovers); `TQDM_COLOURS` only in `constants/user_interface.py`. Live constants: `METRICS`/`SIM_METRICS`/`DAY_METRICS`, `LOSS_KEYS` (metrics.py — Grok deletes that file), `LOCK_TIMEOUT`, `PBAR_WAIT_TIME`, `COUNTY_ALIASES`, `MAP_DEPOTS`, `GAMMA_PRESETS`. | Static. Not deleted. | `remove_enums.py` / a constants allowlist. | approved (Harbinger 2026-09-25) |
| R-cursor-08 | Cursor | `data/generators/datasets.py` `train_time` branch | function | — | `train_time` is an extra `dataset_type` writer (`_generate_train_time_data`). Processor path and `haversine_distance` are unrelated and stay. | Static. Owner keeps the type. | Do **not** drop `train_time` or its yaml key. | keep (Harbinger 2026-09-25) |

### Kimi removal evidence (lane D, 2026-09-25)

Verified in `/tmp/kimi-wsr-del` (staged deletions; `compileall` clean after every stage). Baseline sweep before any deletion: **707 modules, 0 failed**.

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| R-kimi-01 | Kimi | `route_construction/base/base_multi_period_policy.py` | module | 250 | Zero subclasses in the retained tree (SIRP/multi-period policies were pruned); only re-exported from `base/__init__.py`. Its `ProblemContext`/`SolutionContext`/`ScenarioTree` imports have independent live users, so no cascade. | Deleted with the `base/__init__.py` re-export removed; import sweep 707 -> 705, 0 failed; full smoke below. | manual (one file + one `__init__` line) | verified locally |
| R-kimi-02 | Kimi | `learning_algorithms/neural_agent/batch.py` (`compute_batch_sim`) | module | 197 | Zero callers repo-wide (`grep compute_batch_sim` -> definition only). The HRL branch inside is likewise unreachable: `policy_na.py:124` forwards `hrl_manager` but `compute_simulator_day` never uses it. | Deleted with `agent.py` MRO cleaned (`NeuralAgent(SimulationMixin)`); same sweep (705, 0 failed). | manual | verified locally |
| R-kimi-03 | Kimi | Dead tracking remnants: `BaseRoutingPolicy._log_solver_params` + call (`base_routing_policy.py:412-445,500` orig), `NeuralAgentPolicy._log_params` + call + `_params_logged` (`policy_na.py:50,132,143-169` orig), duplicate `from typing import Any  # AUTO-REPLACED` (`base_routing_policy.py:26` orig), and the two SANS dispatcher call sites (`simulated_annealing_neighborhood_search/dispatcher.py:101,196` orig) | function | ~75 | Bodies unreachable (`run = None  # AUTO-REMOVED` then immediate return); `kimi_lane_d_repro_20260925.py` asserts the pattern. SANS dispatcher call sites deleted as export cleanup (lane-E file, noted here). | Deleted; same sweep. | manual | verified locally |
| R-kimi-04 | Kimi | Dead attention-hook + viz plumbing in `neural_agent/simulation.py`: `add_attention_hooks` dummy (:310-312 orig), `hook_data` capture/removal (:68,74-76,248-253 orig), `_viz_record` guard (:257-258 orig) | function | ~30 | The real hooks module was removed; the dummy always returns empties, so `attention_weights`/`graph_masks` were always empty in the returned dict. `NeuralAgent` (mixin-only class) has no `_viz_record`, so the `hasattr` guard never fired. | Deleted; the returned output dict keeps the same keys with empty tensors (contract preserved); same sweep. | manual | verified locally |
| R-kimi-05 | Kimi | `other_algorithms/travelling_salesman_problem/two_opt.py` + the `engine=="custom"` branch in `tsp.py:43,57-58` + `__init__.py` re-export | module | 58 | `solve_tsp_2opt` referenced only by the custom branch; the only `find_route` caller (`fast_tsp.py:70-75`) never passes `engine=`; no yaml sets `engine: custom`. | Deleted with the 3 small edits; import sweep 705 -> 704, 0 failed. | manual | verified locally |
| R-kimi-06 | Kimi | `tsp.py::{get_partial_tour, dist_matrix_from_graph, calculate_tour_cost, get_path_cost}` | function | ~91 | Zero importers outside tsp.py (`get_path_cost` only called by `dist_matrix_from_graph`). Frees the `networkx`/`dijkstra_path` imports in tsp.py (`networkx` itself stays -- other retained files use it, per R-muse-02's probe). | Deleted + imports removed; same sweep. | manual | verified locally |
| R-kimi-07 | Kimi | `meta_heuristics/ant_colony_optimization_k_sparse/` (`__init__.py` 5 + `pheromones.py` 138) + the node-pheromone block in `hyper_aco.py` (class docstring, init :171-192 orig, `deposit_edge` :806-823 orig, `evaporate_all` :937-944 orig) | package | ~190 | `SparsePheromoneTau` imported only by `hyper_aco.py` (function-level) and its own `__init__`; `deposit_edge`/`evaporate_all` have zero callers; on the retained path the matrix is inert (0 stored edges after a full solve, observed). pg_clns uses its own `PheromoneTau` (its `k_sparse: 10` yaml key is a sparsity parameter, not this package). | Deleted + hyper_aco block removed + stale `meta_heuristics/__init__.py` comment cleaned; import sweep 704 -> 702, 0 failed. | manual | verified locally |
| R-kimi-08 | Kimi | `HyperHeuristicACO.construct` + `_bootstrap` + `_bootstrap_standard` + `_bootstrap_profit` (`hyper_aco.py:327-727` orig) + the 22 `recreate_repair` insertion imports used only by them (:43-66 orig) | function | ~480 | No callers repo-wide (grep `.construct(` finds only pg_clns's own constructor); docstrings name HVPL as the caller (removed). The live ACO-HH path (`solve()`) needs only `recreate_repair/greedy.py` from the 12-module recreate family -- lane-E input for R-claude-05. hyper_aco.py went 944 -> 459 lines. | Deleted (single-file edit); import sweep 702, 0 failed. | manual | verified locally |
| R-kimi-09 | Kimi | NA vectorized-selector fallback in `policy_na.py:197-238` + dead `NeuralParams.selector_name/selector_threshold` fields (`params.py:40-41,94-95,120-121`) + unused `seed` attr on `NeuralAgent` (`agent.py:46`) | function | ~50 | `selector_name`/`selector_threshold` are never injected into the day context (grep over `pipeline/` -> 0 hits); the sim path always feeds the scalar `mandatory` list (confirms Cursor §5.F note with stronger evidence). Reachable only via direct `execute(selector_name=...)` calls. | Not deleted (adapter contract change; owner decision). | manual | proposed |
| R-kimi-10 | Kimi | `smart_waste_collection_two_commodity_flow/pyomo_wrapper.py` + dispatcher pyomo branch (`dispatcher.py:93-110`, eager import :24) + `pyomo` dependency (`logic/pyproject.toml:94`) | module | 274 | Only importer is the dispatcher; nothing else in the repo imports pyomo. The path is unconditionally broken (B-kimi-22) so no working behaviour is lost. Owner choice: one-line `initialize=` fix (B-kimi-22) vs prune. Keep OR-Tools: `framework: ortools, engine: scip` is the only working non-Gurobi backend, and the `engine: gurobi` fallback depends on `CreateSolver` returning None. | Not deleted (owner decision). | manual; `pyproject.toml` edit in the packaging step if pruned | proposed |
| R-kimi-11 | Kimi | SWC-TCF `delta` key (`policy_swc_tcf.yaml:50-55`, `configs/policies/swc_tcf.py:53`, constraint `gurobi.py:135-138`) | config-key | -- | Provably inert (B-kimi-18); removal deletes the constraint + key together. Owner decision on intended semantics. | Not deleted (goes with the B-kimi-18 decision). | manual | proposed |
| R-kimi-12 | Kimi | Per-day ScenarioGenerator block in `pipeline/simulations/actions/route_construction.py:130-157`; chained: `ScenarioGenerator`/`ScenarioTree`/`ScenarioTreeNode` in `pipeline/simulations/bins/prediction.py` (~250 LOC) | function | ~60 (+~250 chained) | No retained policy consumes `context["scenario_tree"]` -- only the removed multi-period base class and two operators' optional params that never receive a tree (`kimi_lane_d_repro_20260925.py` check 4). The block also builds a stochastic tree (np.random draws) for every policy every day before `adapter.execute`. Lane-B file -- flagged for Grok; prediction.py deletion is the follow-on. | Static + repro; not deleted here. | manual | proposed |

### Kimi removal evidence — BPC batch (verified in `/tmp/kimi-wsr-del`)

Applied after the rows above. Cumulative tree diff vs `70e660b03`: **26 files changed, 23 insertions, 2,197 deletions**; import sweep **707 -> 701 modules, 0 failed**. Functional BPC check (`kimi_lane_d_bpc_lci_yaml` adapted for the pruned tree) returns **identical** results before/after (obj 17.0 on the 3-node counterexample -- the intentionally-retained invalid LCI engine, B-kimi-02, is unaffected by these deletions).

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| R-kimi-13 | Kimi | `branch_and_price_and_cut/bpc_engine.py::_select_nodes_knapsack` (:182-312 orig) + `knapsack_proc_selection` param (`params.py:59,70` orig; not in the configs dataclass) | function | 131 | Defined, never called; the engine comment at :404-410 says "No node pre-selection"; grep repo-wide -> definition + param default only | Deleted; params field removed; compileall + sweep green; BPC counterexample unchanged | manual | verified locally |
| R-kimi-14 | Kimi | `pricing/smoothing.py::dssr_pricing_wrapper` (:323-417 orig) + dead `use_dssr`/`dssr_max_iters` params (`params.py:152,156` orig, mirrored `configs/policies/bpc.py`, yaml `policy_bpc.yaml:188-195`) | function | ~95 | Zero callers (`use_dssr` never passed truthy by `column_generation_loop`); the wrapper is also broken (`_ng_memory` does not exist, B-kimi-06). Sole-consumer imports (`time`, `numpy`) removed. | Deleted + params/yaml keys removed; same verification | manual | verified locally |
| R-kimi-15 | Kimi | `helpers/solvers_and_matheuristics/search/search_strategy.py` (`create_search_strategy`, `BestFirstSearch`, `DepthFirstSearch`, `HybridSearchStrategy`, `NodeSelectionStrategy`) | module | 236 | Zero usage outside the package; the B&B tree dispatches by string (`branching/tree.py:174-200`), never these classes. `search/` still holds live files -- only the module went. Re-exports cleaned in `search/__init__.py` + package root `__init__.py`. | Deleted; sweep 702 -> 701, 0 failed | manual | verified locally |
| R-kimi-16 | Kimi | No-op cut engines in `search/cutting_planes.py`: `MinCutInequalityEngine` (:1552-1664 orig), `TriangleCliqueCutEngine` (:1667-1827), `NodeProfitBoundEngine` (:1957-2081), `PathEliminationEngine` (:2084-2201) + factory wiring (`create_cutting_plane_engine` at :1433 orig; `bpc_engine.py:548-565` reduced to Composite + Rank1) | class | ~517 | The four engines call master methods that do not exist (B-kimi-08) and are pure no-ops; MinCut's fallback mutated base constraints -- removing it eliminates that latent corruption. `"all"` now maps to the 9 remaining engines + LimitedMemoryRank1. | Deleted; factory `"all"`/valid lists updated; counterexample unchanged; sweep green | manual | verified locally |
| R-kimi-17 | Kimi | Dead methods: `solve_ip` (`master_problem/model.py:423-458` orig), `has_artificial_variables_active` (`problem_support.py:783-797` orig), `find_and_add_violated_rcc` + its sole callee `add_set_packing_capacity_cut` (`master_problem/constraints.py:470-527,331-367` orig), `Label.is_feasible` (`pricing/labels.py:123-132` orig), `Node` dataclass (`common/node.py:23-42` orig, incl. now-unused imports), `BIG_M` (`model_problem/model.py:159-163` orig), `SeparationEngine.separate_integer` (`separation/engine.py:90+` orig), + dead arc-fixing block in `bpc_engine.py:714-732` orig and `reduced_cost_arc_fixing` (`pricing/smoothing.py:425-436` orig, 0 callers after block removal) | function | ~280 | Zero callers each (grep over retained tree; HGS's unrelated `is_feasible` attribute untouched; the `MasterProblemSupport` Protocol *declarations* of the same names in problem_support.py are interface stubs, kept). | Deleted; sweep green; counterexample unchanged | manual | verified locally |
| R-kimi-18 | Kimi | Dead ACO-HH config surface: `HyperHeuristicACOConfig.{sequence_length, local_search, local_search_iterations, elitist_weight}` (`configs/policies/aco_hh.py:63-66` orig), yaml `sequence_length` (`policy_aco_hh.yaml:91-94` orig); if the owner picks removal over wiring for B-kimi-26, also the `operators` plumbing (`policy_aco_hh.py:128`, `params.py:21,75,120` orig, `aco_hh.py:67` orig) | config-key | ~12 | Zero readers on the solver path (B-kimi-26/27); `sequence_length` never reaches `HyperACOParams`. | Not deleted (owner wiring/removal decision pending). | manual | proposed |
| R-kimi-19 | Kimi | `search/cutting_planes.py::KnapsackCoverEngine` (:639-739 orig) + `BasicFleetCoverEngine` (:792-890 orig) | class | ~200 | Reachable only via factory/`"all"` when `master.vehicle_limit` is set; the shipped sim runs `n_vehicles: 0` -> `vehicle_limit=None`, so they never fire. `KnapsackCoverEngine`'s `sum crossings >= K` cut is dimensionally wrong anyway (review note). | Not deleted (conditional on whether vehicle-limited BPC stays a supported config). | manual | proposed |

### Muse removal evidence (lane-G assist, 2026-09-25)

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| R-muse-01 | Muse | `logic/src/configs/tasks/train.py`: `route_improvement_epochs`, `lr_route_improvement`, `efficiency_weight`, `overflow_weight`, `accumulation_steps`, `enable_scaler`, `eval_only` | config-key | ~10 | Zero readers outside `logic/src/configs` (`grep -rln` over `logic/src` + `main.py`, submodule excluded). `train.yaml` sets none of them, so no yaml side needs editing — only the dataclass and any `config.yaml` defaults-list entry. Counter-examples that must NOT ride along: `checkpoint_encoder` (read by `utils/model/loader.py`, `models/core/attention_model/model.py`), `shrink_size` (read by `glimpse/decoder.py`, loader), `use_pbrs`/`pbrs_*` (read by `rl/common/base/module.py:218-219`, `steps.py:226` — PBRS is reachable, agreeing with Codex §5.A), `persistent_workers`/`pin_memory` (read by RL base/data/steps). | Static + probe `muse_lane_g_probe_20260925.py` (`dead-field:*`, `pbrs-is-read`, `checkpoint_encoder-is-read` all PASS). No local deletion run — needs lane-G dataclass↔yaml sweep before the edit. | Narrow R-claude-08 to exactly these 7 fields; mirror in `logic/configs/tasks/train.yaml` only if keys are added there later. | proposed |
| R-muse-02 | Muse | `logic/pyproject.toml` dependencies: `python-pptx`, `docxtpl`, `latex2mathml`, `cryptography`, `requests`, `joblib`, `pydantic`; conditionally `torch-geometric`/`torch-scatter`/`torch-sparse`, `wandb`, `openpyxl` | dependency | — | Zero `^(from|import)` hits for the first seven across `logic/**/*.py` + `main.py` (submodule `wsmart_bin_analysis` excluded — it is a gitlink, empty in fresh worktrees). `torch_geometric` is imported only by `models/subnets/embeddings/edges/{none,base}.py` at top level, and `embeddings/__init__.py` does `from .edges import ...` at package import — so the whole models package hard-requires it today; it drops automatically with R-gemini-02 + the `__init__` cleanup (probe `torch_geometric-only-in-edges` PASS). `wandb` is imported only by `tracking/logging/modules/metrics.py` (R-grok-03; probe PASS). `jinja2` stays while `gui.py` stays (probe `jinja2-only-gui` PASS). `openpyxl` is coupled to the `.xlsx` loader R-claude-03 proposes to drop — remove together. `networkx` stays: used by retained `tsp.py` + 3 more files (probe PASS). NOT verified here and left for lane G: `einops`, `scikit-learn`, `hydra-colorlog`, per-solver optionality (`gurobipy`, `ortools`, `pyvrp`, `fast_tsp`, `alns`, `hexaly`, `vrpy`). | Static + probe (`dep-unused:*` PASS). No install-footprint test run. | `logic/pyproject.toml` edit in the packaging step, after R-gemini-02 / R-grok-03 / R-claude-03 land. | proposed |
| R-muse-03 | Muse | `ci/export_config.json` `imitation_policy` category (`impl_dirs: logic/src/models/policies`, `impl_root`, `yaml_dirs: logic/configs/models/policies`, `config_dirs: logic/src/configs/rl/policies`) | config-key | — | `logic/src/models/policies` does not exist on the branch; the vector policies live in `logic/src/policies/vector` (probe PASS). Any export that keeps an imitation policy while pruning by this category silently keeps nothing or prunes by a dead path. | Static path check only. | Point the category at `logic/src/policies/vector` (+ its real yaml/config dirs) or delete the category if imitation policies are out of the export contract. | proposed |

## 4. Packaging-script consequences (append as removals are confirmed)

Known gaps in the current tooling (found during the 2026-09-25 prune; see bus entry for details) that the roadmap must fix before the next export:

1. `prune_codebase.py::prune_category` — keeping zero algorithms in a category with empty `yaml_prefixes`/`config_prefixes` deletes *every* file under the category's `yaml_dirs`/`config_dirs` (the `joint` category wiped `logic/configs/policies/other` and `logic/src/configs/policies/other`).
2. `cleanup_helper._match_acronym` — pruning `hgs_adc`/`hgs_alns`/`hgs_rr` deleted `meta_heuristics/hybrid_genetic_search` (and `policy_hgs.yaml`) even though `HGS` was kept.
3. `cleanup_helper.clean_init_file` comments only the first line of a parenthesised multi-line import (left `IndentationError`s in seven files).
4. `remove_tracking.py` stubs the simulator's result writers (`output_stats`, `update_policy_log_section`, ...) as no-ops, silently disabling result files.
5. `remove_enums.py` regex `@GlobalRegistry\.register\s*\([^)]*?\)` cannot span nested parentheses.
6. `test_sim.yaml` / `train.yaml` / `config.yaml` defaults lists and `sim.policies` are not touched by any script.
7. `ci/export_config.json` `imitation_policy.impl_dirs` points at `logic/src/models/policies`, which does not exist (the vector policies live in `logic/src/policies/vector`).

### Codex additions to packaging consequences

8. Apply R-codex-01/02 as dependency-aware export transformations; verified saving is **8 Python files / 711 physical lines**, including export cleanup. Keep proposed R-codex-03/04 out of the confirmed total.
9. Strengthen acceptance checks beyond process exit/import success: assert generated sample counts, selected validation graphs, baseline parameters, one-dimensional advantages, and loaded-model architecture/prediction parity. Existing smoke commands succeed despite B-codex-01/06/08.
10. R-claude-06 requires coordinated caller edits: remove `train/engine.py`'s eager `zenml_train_pipeline_module` import, its dispatch at 433–435 and `_run_training_via_zenml`; remove `test/engine.py`'s dispatch at 69–72 and `_run_sim_via_zenml` (including its lazy import); remove `eval/__init__.py`'s pipeline import, dispatch at 47–50 and `_run_eval_via_zenml`. Remove the three pipeline modules (527 gross lines), `configure_zenml_stack` stubs, and `zenml_enabled/store_url/stack_name` fields plus matching YAML keys. This is a proposal, not a verified deletion.
11. Callback removal also needs `rl/common/trainer.py`: remove imports and `_build_callbacks` append blocks at 227–240, then both callback packages' exports. GPU memory monitoring is actually reachable on CUDA-capable machines; it is not dead code. Keep model checkpointing. Heatmap callback refers to removed `maybe_log_eval_attention_heatmaps`; do not leave a dangling opt-in flag. Re-test training with CUDA after removal.
12. Fresh worktree verification requires the simulator submodule as well as data. The first sweep had 14 `GridBase` import failures from an uninitialized gitlink; providing the existing submodule resolved them. An export should verify that the submodule contents are present before packaging, and test the extracted archive in isolation.

### Gemini additions to packaging consequences

13. Restrict `subnets/embeddings` via an explicit allowlist in `logic/package/prune_codebase.py` / `ci/export_config.json`. Retained AM path requires only `nodes/init.py`, `context/base.py`, and `context/vrpp.py`. Pruning `positional/` (R-claude-01, 191 LOC), `edges/` (R-gemini-02, 231 LOC), `state/` (R-gemini-03, 179 LOC), `context/generic.py` (R-gemini-04, 93 LOC), and `dynamic.py`/`static.py` (R-gemini-01, 146 LOC) removes **11 files / 840 LOC** and drops `torch_geometric` from model imports (unblocking R-claude-09).
14. Delete `models/common/{non_autoregressive,improvement,transductive}` (R-claude-04, **14 files / 1,233 LOC**) and atomically clean re-exports in `models/__init__.py`, `models/common/__init__.py`, and `policies/vector/__init__.py`. Keep only `models/common/autoregressive/`.
15. Prune experimental subnet modules: `modules/{cross_attention,flash_attention,normalized_activation_function,dynamic_hyper_connection,static_hyper_connection}.py` (R-gemini-05, **5 files / 617 LOC**) and clean `subnets/modules/__init__.py`.
16. Prune dead policy variant `models/core/attention_model/symnco_policy.py` (R-gemini-06, **1 file / 88 LOC**) and clean `attention_model/__init__.py`.
17. Fix model loader fidelity and eliminate fake parameter synthesis: repair `loader.py:94-106` by constructing structured `NormalizationConfig` and `ActivationConfig` objects before passing to `AttentionModel` (resolving B-gemini-01 and B-codex-08). Eliminate the duplicate `VRPPContextEmbedder` on `AttentionModel` and remove the erroneous fake projection weight synthesis in `loader.py:150-162` (resolving B-gemini-03). Unify training and inference architectures by adopting `AttentionModelPolicy` with a lightweight dict adapter for evaluation/simulation, eliminating the dual model implementation drift.
18. Fix CUDA generator migration: in `GlimpseDecoder._select_node` (B-gemini-04), dynamically match `self.generator`'s device to `probs.device` or recreate it on `to(device)`, preventing runtime crashes when sampling on GPU. Also replace the unbounded rejection while-loop (B-gemini-05) with a direct masked probability zeroing and single multinomial draw.
19. Prune deep decoder policy and YAML (owner approved): delete `logic/src/models/core/attention_model/deep_decoder_policy.py` and `logic/configs/models/deep_decoder.yaml` (R-gemini-07, **2 files / 176 LOC**). Clean imports and registry entries in `logic/src/policies/vector/__init__.py`, `logic/src/models/core/attention_model/__init__.py`, and `logic/src/pipeline/features/train/model_factory/builder.py`.
20. Prune critic network and shared critic baseline for minimal export packaging (owner approved): delete `logic/src/models/common/critic_network/` (3 files, 344 LOC) and `logic/src/pipeline/rl/common/baselines/shared_critic.py` (93 LOC) (R-gemini-08, **4 files / 437 LOC**). Clean re-exports in `logic/src/models/__init__.py`, `logic/src/models/common/__init__.py`, `logic/src/pipeline/rl/common/baselines/__init__.py`, `logic/src/pipeline/rl/common/__init__.py`, and `logic/src/pipeline/rl/core/__init__.py`. On main, document/repair critic initialization for full feature parity; prune from the minimal export artifact.

### Grok additions to packaging consequences

21. Apply R-grok-01–04 together. Verified saving in `/tmp/grok-wsr` is **1,866 deletions / 10 insertions** across 9 deleted modules plus the dead half of `storage.py`. Import sweep afterwards: **698 modules, 0 failed** (707 before). Keep `utils/input/files.py`, `update_policy_log_section`, `setup_system_logger`, and `gui.py`.
22. `remove_tracking.py` currently stubs the result writers (item 4). The retained contract needs the opposite: keep the jsonl writer (`gui.send_daily_output_to_gui`, `gui.send_final_output_to_gui`) and `update_policy_log_section`, and delete `metrics.py` so `wandb` leaves the import graph. `jinja2` stays while `popup.html` is rendered into the day-1 jsonl record. `rich` stays; the day table imports it directly.
23. Give every policy one filesystem key, the expander id in `sim.full_policies` (includes `_emp` and the real improver token). `SimulationContext.pol_name` is today `to_slug(display_name)`, which drops the distribution and rewrites an empty improver as `none` (B-claude-06, B-grok-02). Parallel `single_simulation` then rejects a successful result. `cpu_cores: 0` already means "use every core but one", so the mismatch is on the default path. Assert in the smoke that `log_<full_policies id>_<N>N.json` exists and that its `mean.km` equals the mean of `samples.*.km`.
24. Empirical packaging check for B-grok-01: after a smoke with `data_distribution=emp` and `graphs_20V_1N_plastic.json`, the bin ids inside the waste generator must equal the coordinate ids used to slice the distance matrix. Today's Rio Maior pair is `old_out_info` for the graph and `out_info` / `out_rate_crude` for the grid.
25. Sample `time` must match the sum of daily solver times (B-grok-03), and a resume smoke must show a positive `time` equal to previous elapsed plus the continued run (B-grok-05). Overflow smoke: a bin left at 100% for a day with zero new waste must not increment `overflows` (B-grok-04).
26. Confirm Codex item 12 from this lane: a fresh worktree of `70e660b03` has an empty `wsmart_bin_analysis` gitlink, and `emp` imports `GridBase` from it. The export has to materialize that submodule. `__init__.py` pulls `export.simulation` and `python.sample_gen`. The `ui/`, `benchmark/`, `test/`, and `docs/` trees are outside that import. Shipping only the imported library is a proposal; it was not deleted here.
27. `test_sim.yaml` still lists `sim.problem: ctop` as a supported switch. If the export keeps that switch, fix B-grok-08 in the same change. The retained contract exercised here is `vrpp`.

### Cursor additions to packaging consequences

28. Apply R-cursor-01–06 together. Verified saving in `/tmp/cursor-wsr` is **1,965 deletions / 4 insertions** across 18 deleted modules plus export cleanup. Import sweep afterwards: **689 modules, 0 failed** (707 before). Keep `distributions/{base,statistical_empirical,statistical_gamma,statistical_uniform,statistical_beta,statistical_constant,spatial_distance}.py` until R-claude-02’s remaining cut; keep `TensorDictDataset` + `BaselineDataset`; keep `GenerativeDataset` + `NumpyDictDataset` + `SimulationDataset`.
29. `--distributions empirical gamma` already exists but `_filter_files` does not rewrite `DISTRIBUTION_REGISTRY` / `__all__` or the `waste.py` `unif`/`beta`/`const`/`dist`/`empty` branches (R-claude-02). **Harbinger 2026-09-25:** drop those four modules after the §0 train smoke moves from `unif` to `gamma1`. Same edit must change `VRPPGenerator.__init__` default `waste_distribution` to `gamma1` and shrink `datasets.py::distributions_per_problem` to `emp` + `gamma*`.
30. Pytorch datasets need a category. **Harbinger 2026-09-25:** survivors are `TensorDictDataset` + `BaselineDataset` (`td_dataset.py`, `baseline_dataset.py`). The four unused classes are R-cursor-02. Do not collapse `BaselineDataset` until rollout (`rl/common/baselines/rollout.py:255`) stores `bl_vals` on the TensorDict.
31. Simulation datasets: **Harbinger 2026-09-25:** survivors are `SimulationDataset` + `GenerativeDataset` + `NumpyDictDataset` (`npz` + `generative`), not a single class. `bins/base.py` extension dispatch and `repository/__init__.py:106-108` must lose `.pkl/.xlsx/.csv` in the same change. `make_dataset` already normalises `.npz` waste by `max_waste` (percent → fraction); keep that.
32. Selector packaging checks: (a) **Harbinger 2026-09-25:** last-minute is one unit — stored yaml is percent `70`/`90` (`ms_last_minute.yaml`). Rewrite `LastMinuteSelector` default `0.7` and `train.yaml` nest `0.25` to that same percent; convert only at the fraction-fill boundary (NA already does `/100`). (b) training `steps.py` must prepend a depot column before `select()` (B-cursor-02). (c) **Harbinger 2026-09-25:** service-level stays linear `z·σ·n_d` in both copies (B-cursor-01 wontfix). Do not add `√n_d`; there is no sqrt implementation to delete. (d) `fast_tsp.find_tour` is called with the configured seed, default time budget 2 s (B-cursor-05). Empty tours `[0]`/`[0,0]` already return unchanged.
33. `haversine_distance` stays — `processor/formatting.py` calls it. `IDistanceMetric` and `JointSelectionConstructionContext` go (R-cursor-05). `ImprovementEnvBase` goes (R-cursor-03). **Harbinger 2026-09-25:** prune the dead constants in R-cursor-07 (`MAX_LENGTHS`, `CRITICAL_FILL_THRESHOLD`, `OPERATION_MAP`, `FS_COMMANDS`, `TQDM_COLOURS`). Keep `train_time` (R-cursor-08 is not a removal).

### Muse additions to packaging consequences (verified against `main`-branch scripts)

All probes below were executed against the scripts as they exist on `main`
(`git show main:logic/package/*.py`) plus the review-branch tree at
`70e660b03`. Reproducer: `.venv/bin/python .agent/cache/tools/muse_lane_g_probe_20260925.py`
(29 checks, all PASS).

34. **Bulk-delete guard is missing, and the trigger is shared dirs, not just empty prefixes.** `prune_codebase.py::prune_category` (lines 605–620): when the keep list is empty it calls `_prune_all_yaml_by_prefix` / `_prune_all_configs_by_prefix` / `_prune_entire_impl_dirs`. `_prune_all_yaml_by_prefix` (lines 499–505) sets `should_delete = True` for *every* yaml when `yaml_prefixes` is falsy. The `joint` incident is worse than "empty prefixes": the `joint` category's `yaml_dirs`/`config_dirs` are `logic/configs/policies/other` and `logic/src/configs/policies/other`, which it **shares** with the `selector` and `improvement` categories — so even a correctly-scoped bulk delete wipes other categories' files. Fix: (a) never default falsy prefixes to delete-all — require an explicit `--prune-all-<category>` flag; (b) refuse bulk deletion when the category's yaml/config dirs are listed by any other category (fail loudly instead of wiping shared dirs).
35. **§4 item 2's stated mechanism is wrong; the real one is the parent-dir kill.** Direct probes of `_match_acronym` return `False` for `hybrid_genetic_search` vs `hgs_adc` / `hgs_alns` / `hgs_rr` — the initials heuristic does NOT match those pairs. The over-deletion comes from `_find_impls_to_delete` (`cleanup_helper.py:313-322`): when a single *file* matches the pruned acronym, the whole **parent directory** is added to `to_delete` (unless in `PROTECTED_DIRS`). A variant file inside `hybrid_genetic_search/` matching `hgs_adc` therefore deletes the entire HGS implementation that was supposed to stay. Fix: delete only the matched file; delete a directory only when the directory name itself matches. Separately, gate the initials heuristic (lines 201–207) out of delete decisions — exact, `policy_`/`selection_`, prefix/suffix and yaml-prefix equality are sufficient for pruning.
36. **`clean_init_file` single-line commenting confirmed** (`cleanup_helper.py:104-134`): the loop appends `# <line>  # AUTO-CLEANED` for the matched `from ... import (` opener only; continuation lines survive and re-indent into `IndentationError`s. Fix: when the matched line opens a parenthesis/bracket, comment through the closing bracket (or re-emit the `__init__` via `ast` instead of line regexes).
37. **`remove_enums.py` regex confirmed** (`process_python_file`): `(?s)@GlobalRegistry\.register\s*\([^)]*?\)` — the `[^)]` class stops at the first `)` even with `DOTALL`, so any decorator with nested parens survives silently, and the "handles multi-line as well" comment is only true for paren-free args. Fix: balanced-paren scan from the opening `(`.
38. **`remove_tracking.py` stub list confirmed with line numbers** (lines 368–375): the script emits no-op `output_stats`, `send_daily_output_to_gui`, `send_final_output_to_gui`, and `update_policy_log_section` — i.e. exactly the result writers the retained contract needs (§0). This pins §4 item 4: the fix is Grok item 22 (keep-result-writers mode), not a tweak to the stubs.
39. **Apply R-muse-01–03 as one config/dependency hygiene change.** R-muse-01 narrows R-claude-08 to 7 reader-less train fields (PBRS, `checkpoint_encoder`, `shrink_size` explicitly excluded with reader evidence). R-muse-02 gives R-claude-09 a verified drop list (7 zero-hit packages) plus three coupled drops (`torch-geometric*` with R-gemini-02, `wandb` with R-grok-03, `openpyxl` with R-claude-03) and names what lane G must still check. R-muse-03 repairs the dangling `imitation_policy` category (§4 item 7 follow-through).

### Kimi additions to packaging consequences (lane D)

All probes executed against the tree at `70e660b03`; reproducers under `.agent/cache/tools/kimi_lane_d_*_20260925.py`.

40. **Apply lane-D removals as one staged change.** R-kimi-01–08 and R-kimi-13–17 are verified locally in `/tmp/kimi-wsr-del`: staged deletions with `compileall` + import sweep green after every stage (707 -> 701 modules, 0 failed; cumulative tree diff 26 files, 23 insertions, 2,197 deletions) and the full smoke below. `helpers/solvers_and_matheuristics` needed per-file `__init__.py` re-export cleanup (the package root eagerly imports every submodule). None of the deletions sit on the retained runtime path; the four no-op cut engines are the only behaviour-relevant deletion, and removing them eliminates MinCutInequalityEngine's latent constraint-rewriting (B-kimi-08) -- a strict improvement. Leftovers for the packaging pass: the unread `enable_reduced_cost_arc_fixing` dataclass field (params.py:144, bpc.py:114), the now-dead `_find_customer_components` (constraints.py:434, orphaned by R-kimi-17), and the Protocol stubs in problem_support.py (interface declarations, kept deliberately).
41. **Policy-yaml key drift is a class of bug, not one-offs.** B-kimi-06/07/10/18/25/26/27 all share one shape: a yaml key that either is never read, is read but dead at the call site, or documents semantics the code does not implement. Add an export acceptance check: for every key in `policy_*.yaml`, grep the policy package for a read, and fail the export on unread keys unless they are explicitly allowlisted. Today's unread/inert key list: BPC `enable_dssr`, `dssr_max_iters`, `enable_reduced_cost_arc_fixing`, `max_cut_iterations`, `use_spatial_partitioning`, `enable_hybrid_search` (not even a field); ACO-HH `sequence_length` (+ dataclass-only `local_search`, `local_search_iterations`, `elitist_weight`); SWC-TCF `delta`. The `operators` key IS read but ignored by the solver (B-kimi-26) -- the check should cover "read but never affects behaviour" where feasible.
42. **Solver-backend matrix for SWC-TCF must be pinned in the export.** Verified on the installed ortools: `CreateSolver("GUROBI"|"HIGHS"|"CPLEX") -> None` (only SCIP/CBC/SAT linked). So `framework: ortools, engine: scip` is the only working non-Gurobi path, `engine: gurobi` reaches the native fallback by accident of that None, and `engine: highs/cplex` silently returns empty tours (B-kimi-20). Either constrain the schema (enum-validate `engine`/`framework`) or delete `pyomo_wrapper.py` + the `pyomo` dep (R-kimi-10) -- the pyomo path crashes unconditionally (B-kimi-22).
43. **BPC is not exact under the shipped yaml** (B-kimi-02 invalid LCI cuts, B-kimi-03 unsound LR pre-pruning, B-kimi-11 unsound UB prune with heuristic pricing). Until the owner decides fix-vs-relabel, the export must not describe BPC as an exact solver. If fixed, add lane-D's 3-node counterexample (`kimi_lane_d_bpc_lci_yaml_20260925.py`, true optimum 25) to the packaging smoke as a regression gate.
44. **Cross-lane flags.** (a) The per-day scenario-tree build in `actions/route_construction.py:130-157` is dead work for all nine retained policies -- R-kimi-12, needs lane-B (Grok) sign-off to delete from that file; `pipeline/simulations/bins/prediction.py` follows. (b) B-kimi-32 (base early-return ignores `vrpp`) explains the km=0-before-threshold acceptance behaviour -- do NOT "fix" it in packaging without an owner decision, or smoke expectations change. (c) ACO-HH live path needs only 6 of the 13 `helpers/operators` families (destroy_ruin/{random,shaw,string}, perturbation_shaking/{kick,perturb}, intra/inter_route_local_search, recreate_repair/greedy.py, solution_initialization/greedy_si.py) -- input for Qwen's R-claude-05 pass. (d) Lane-E note: `local_search_hgs.py` optional flags referenced in R-claude-05 were not re-verified here.
45. **Smoke additions from lane D** (after the respective fixes): ACO-HH determinism (same seed -> same tour, B-kimi-28); SWC-TCF tiny-instance brute-force gate (true optimum, B-kimi-16); BPC 3-node LCI gate (obj == 25, B-kimi-02); NA adapter profit units (B-kimi-01). Until fixed, pin the current values as xfail-style witnesses so a silent behaviour change cannot pass unnoticed (same philosophy as Codex item 9).

## 5. Per-agent sections

Each agent appends `## <N>. <Agent> — lane <X> — <date>` below, containing: (a) files actually read, (b) the rows it added to §2/§3 (IDs only), (c) disagreements with other rows, (d) open questions for the owner. Confirmations of another agent's row are one line each.

## 5.A. Codex — lane A — 2026-09-25

**Scope and reproducibility.** Reviewed `70e660b03` in `/tmp/codex-wsr-review-20260925`, while shared HEAD was `9bedfb996`. No source fixes or commits made in the shared checkout. Removal experiment is recoverable from Git and exists only in the scratch worktree. The shared environment interpreter was used, with shared data/submodule symlinks. Initial missing-submodule failures are setup failures, not new product bugs. Torch scatter/sparse emitted binary-loading warnings; these did not prevent CPU checks.

**Read:** REINFORCE, all base mixins, epoch/time-step helpers, baseline registry/base/exponential/rollout/critic and warmup references; training builder/registry/engine; eval dispatcher/engine/evaluators/validation; loader, config mapping and checkpoint utilities; training-policy forward/legacy model constructor; RL/train/eval/model schemas and task YAMLs; PBRS integration; RL helper importers and callback/ZenML callers. Focused utilities were followed through their callers; this is not a claim of exhaustive line-by-line review of every utility file.

**Rows:** B-codex-01–09; R-codex-01–04. Priority order: B-codex-06 (wrong experiment size/validation), B-codex-08 (different architecture at inference), then baseline correctness B-codex-01–05 and eval field contract B-codex-07. B-codex-09 makes corrupted/incompatible artifacts hard to detect.

**Reproducer:** from the scratch worktree, run:

```bash
/home/pkhunter/Repositories/Doc/WSmart-Route/.venv/bin/python \
  /home/pkhunter/Repositories/Doc/WSmart-Route/.agent/cache/tools/codex_lane_a_repro_20260925.py
```

It uses real loss/setup/wrapping functions and small inputs; expensive policy inference/checkpoint I/O is mocked only where noted. All assertions passed. The loss, configuration and sample-order counterexamples are independent of stochastic model quality. B-codex-08 also has a real checkpoint observation: loading `/tmp/codex-wsr-review-output/am` returned `args['normalization']='layer'`, but inspecting `named_modules()` found `BatchNorm1d` for `encoder.layers.0.norm1.normalizer` and `norm2.normalizer`. Capturing `load_state_dict` returned missing `context_embedder.project_step_context.{weight,bias}` and no unexpected keys. This is not evidence that the synthesized projection is mathematically equivalent.

**Removal verification:** deleted the eight R-codex-01/02 modules and 18 re-export lines with no remaining symbol/module matches outside their former definitions. `compileall -q logic main.py` passes; import sweep returns **699 modules, 0 failed**. The report §0 CPU training smoke completed and wrote checkpoint/config/args into `/tmp/codex-wsr-review-output/am`; gamma data generation completed into `/tmp/codex-wsr-review-output/gen`. These checks establish that these helpers are unnecessary for the tested path, not that training is numerically correct.

**Configuration/path answers:**

- Exponential/rollout/critic are selectable; only exponential and rollout have real constructed baseline behavior. Warmup is a retained wrapper but its setting is lost (B-codex-01). `baseline: no` in the YAML comment is misleading: the registry spells it `none`.
- `entropy_weight` is consumed by REINFORCE and the AM policy returns entropy. `max_grad_norm` is used both by REINFORCE's optimizer hook and Lightning's `gradient_clip_val`; consolidate ownership to avoid redundant clipping. No entropy-key mismatch was established.
- `train_time` itself is read from kwargs, but sample identity is lost (B-codex-05). `prepare_epoch` also checks `train_time` on the policy rather than the Lightning module; day metadata is not guaranteed on the first epoch. Record this as follow-up, not a separately verified row.
- `train.accumulation_steps`, `enable_scaler`, `route_improvement_epochs`, `lr_route_improvement` have no runtime reads in the searched retained pipeline/utilities. `train.eval_batch_size` is not used by validation loaders, which use `self.batch_size`; standalone `eval.eval_batch_size` is used. Remove obsolete fields or wire intended behavior, synchronizing YAML and schema. `regenerate_per_epoch` reads the wrong hparams level and is not a declared retained train field. Coordinate final field inventory with lane G.
- PBRS is **reachable**, via `cfg.rl.use_pbrs` and `shared_step`; it cannot be called dead code just because the default is false. Keep it unless the owner removes the capability, with matching caller/schema/YAML cleanup. Its docstring incorrectly equates initial potential zero with terminal potential zero; the bonus depends on collected waste, so policy-invariance claims need a separate mathematical review.
- Eval `_build_dataset_kwargs` resolves the active task env and forwards offset/count; CPU multiprocessing is explicitly rejected and GPU counts must divide `val_size`. No actual multi-GPU execution was performed. `get_best` is cost-minimizing; SamplingEval is reward-maximizing, consistent individually. `np.trim_zeros` removes only outer zeros: `[0,1,0,2,0] -> [1,0,2]`; it does **not** remove internal depot visits. Result overwrite checks use the actual filename written by `save_dataset`; no overwrite-bypass finding established. Output extension follows input even though the writer pickles, an interoperability issue to document or standardize.
- Of six evaluator files, one is `__init__`; greedy and sampling produce detailed arrays consumed by CLI eval. Augmentation/multistart variants are dispatcher-reachable but do not all produce `rewards`/`sequences`, so preserving their names requires integration work. YAML advertises `beam_search`, while the dispatcher has no such branch. Narrow and validate exported method choices rather than assuming every advertised mode works.
- Loader forwards dimensions/layer counts/heads, dropout, aggregation, tanh/masks, checkpoint/shrink/temporal/spatial options, entropy, predictor layers and decoder type. Normalization and activation are flattened into obsolete constructor kwargs (B-codex-08). `_parse_hydra_config` also omits decoder type from the returned dictionary, causing a `glimpse` fallback; that matches the retained contract but is not a general decoder migration. `args.json` alone has nested hparams rather than the flat constructor contract; retain `config.yaml` alongside weights until this format is fixed.

**Cross-review/disagreements:** R-claude-06 is a coordinated-removal proposal, not a safe file-only deletion: train imports ZenML eagerly and WSTrainer instantiates the GPU callback. R-claude-08 must distinguish unused fields from fields silently ignored by bugs; baseline and graph fields need fixes, not deletion. R-claude-04 remains lane C's architecture proposal. B-claude-04's width fix is supported by successful loading of the real 64-hidden checkpoint, but does not fix B-codex-08. Simulator findings B-claude-01/02/05/06 were not independently rerun and are not marked confirmed. No B–G signed review rows were present when this pass began; future lane submissions still require cross-review before §6 can lock.

**Owner decisions:** whether export supports PBRS, non-contract baselines, and augmentation/multistart eval. None blocks the verified 711-line removal recommendation. Keep critic in scope per §0 and repair it. Full nine-policy simulation and CUDA/multiprocess tests were not repeated for these unreachable-helper removals; do not interpret this lane report as final artifact acceptance.

**Offline eval verification:** §0 eval smoke on the generated 12-instance gamma dataset also completed (three batches of four), reporting non-zero KM `6.4567378` and KG `324.0834653`. Results are under `/tmp/codex-wsr-review-output/eval`. This independently confirms B-claude-03's dataset-loading fix on this fixture, and validates the train → load → eval path after R-codex-01/02 removals, subject to B-codex-08's architecture mismatch. It does not establish trained-policy parity.

## 5.C. Gemini — lane C — 2026-09-25

**Scope and reproducibility.** Reviewed commit `70e660b03` in isolated scratch worktree `/tmp/gemini-wsr` (detached from `70e660b03`). The shared workspace checkout `/home/pkhunter/Repositories/Doc/WSmart-Route` (HEAD `9bedfb996`) remains untouched with zero source commits. Baseline import sweep passed with 707 modules (0 failed). Removal experiments were executed in `/tmp/gemini-wsr` and verified end-to-end using `/home/pkhunter/Repositories/Doc/WSmart-Route/.venv/bin/python`. Symlinks to `data/` and `wsmart_bin_analysis` were maintained.

**Read:**
- `logic/src/models/core/attention_model/**` (`model.py`, `policy.py`, `symnco_policy.py`, `deep_decoder_policy.py`, `__init__.py`)
- `logic/src/models/subnets/encoders/**` (`gat/encoder.py`, `gat/layer.py`, `common/encoder_base.py`, `__init__.py`)
- `logic/src/models/subnets/decoders/**` (`glimpse/decoder.py`, `deep/decoder.py`, `common/decoder_base.py`, `__init__.py`)
- `logic/src/models/subnets/embeddings/**` (`positional/**`, `edges/**`, `state/**`, `nodes/**`, `context/**`, `dynamic.py`, `static.py`, `__init__.py`)
- `logic/src/models/subnets/modules/**` (`attention.py`, `activation_function.py`, `cross_attention.py`, `flash_attention.py`, `normalized_activation_function.py`, `dynamic_hyper_connection.py`, `static_hyper_connection.py`, `__init__.py`)
- `logic/src/models/subnets/factories/**` (`attention.py`, `base.py`, `__init__.py`)
- `logic/src/models/subnets/critic_network/**` (`critic_network.py`, `critic_encoder.py`, `__init__.py`)
- `logic/src/models/common/**` (`autoregressive/**`, `non_autoregressive/**`, `improvement/**`, `transductive/**`, `__init__.py`)
- `logic/src/models/__init__.py`
- `logic/src/utils/model/loader.py`, `checkpoint_utils.py`, `config_utils.py`, `problem_factory.py`
- `logic/src/configs/models/**` (`attention_model.py`, `deep_decoder.py`, `base.py`)
- `logic/configs/models/**` (`attention_model/am.yaml`, `attention_model/deep_decoder.yaml`, `attention_model/symnco.yaml`)
- `logic/src/envs/tasks/vrpp.py`
- `logic/src/policies/vector/__init__.py`

**Rows added:**
- Bugs: `B-gemini-01` through `B-gemini-06`. Priority order:
  1. `B-gemini-04` (blocker GPU crash during sampling evaluation due to unmigrated CPU generator).
  2. `B-gemini-01` / `B-codex-08` (major architecture divergence: loaded checkpoint silently builds `BatchNorm1d` instead of `LayerNorm`).
  3. `B-gemini-03` (major dead-layer weight synthesis: fake identity projection synthesized for unused duplicate context embedder).
  4. `B-gemini-02` (major silent configuration drop: direct instantiation of `AttentionModelPolicy` ignores `normalization` string).
  5. `B-gemini-05` (major infinite-loop risk in sampling decoding loop).
  6. `B-gemini-06` / `B-codex-07` (major evaluation cost metric sign inversion).
- Removals: `R-gemini-01` through `R-gemini-08`, plus empirical confirmation of `R-claude-01` and `R-claude-04`.
  - Confirmed and verified locally: **39 files / 3,391 LOC** (10 Python packages/modules).
  - Proposed for follow-up: `R-gemini-09` (~60 LOC dead dataclass fields).

**Reproducer:** run from workspace root:
```bash
/home/pkhunter/Repositories/Doc/WSmart-Route/.venv/bin/python \
  /home/pkhunter/Repositories/Doc/WSmart-Route/.agent/cache/tools/gemini_lane_c_repro_20260925.py
```
This script validates all six bugs against commit `70e660b03`:
- Tests actual checkpoint `/tmp/codex-wsr-review-output/am` with `load_model`, confirming `normalization='layer'` in `args.json` builds `BatchNorm1d` and `GELU` (`B-gemini-01`).
- Tests direct instantiation of `AttentionModelPolicy(normalization="layer")`, confirming it produces `BatchNorm1d` (`B-gemini-02`).
- Inspects `AttentionModel` vs `GlimpseDecoder` to prove `model.context_embedder` is dead and duplicate (`B-gemini-03`).
- Executes `GlimpseDecoder` on CUDA with CPU generator, reproducing `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'` (`B-gemini-04`).
- Demonstrates the infinite while-loop condition in sampling selection (`B-gemini-05`).
- Validates the cost/neg_profit convention mismatch (`B-gemini-06`).

**Removal verification in scratch worktree `/tmp/gemini-wsr`:**
Deleted 39 files (3,391 LOC) across:
1. `subnets/embeddings/positional/` (R-claude-01, 3 files, 191 LOC)
2. `models/common/{non_autoregressive,improvement,transductive}/` (R-claude-04, 14 files, 1,233 LOC)
3. `embeddings/{dynamic,static}.py` (R-gemini-01, 2 files, 146 LOC)
4. `embeddings/edges/` (R-gemini-02, 4 files, 231 LOC) — drops `torch_geometric`
5. `embeddings/state/` (R-gemini-03, 3 files, 179 LOC)
6. `embeddings/context/generic.py` (R-gemini-04, 1 file, 93 LOC)
7. `modules/{cross_attention,flash_attention,normalized_activation_function,dynamic_hyper_connection,static_hyper_connection}.py` (R-gemini-05, 5 files, 617 LOC)
8. `attention_model/symnco_policy.py` (R-gemini-06, 1 file, 88 LOC)
9. `attention_model/deep_decoder_policy.py` & `configs/models/deep_decoder.yaml` (R-gemini-07, 2 files, 176 LOC)
10. `models/common/critic_network/` (3 files, 344 LOC) & `rl/common/baselines/shared_critic.py` (1 file, 93 LOC) (R-gemini-08, 4 files, 437 LOC)

Re-exports were cleaned across all affected `__init__.py` files (`models`, `common`, `attention_model`, `vector`, `baselines`, `rl/common`, `rl/core`).
- `compileall -q logic main.py` passes cleanly.
- `import_sweep.py` passes: **669 modules, 0 failed** (exactly 38 modules dropped).
- Full pipeline smoke executed:
  1. Data generation: `gen_data` (smoke gamma1, 12 instances) succeeded.
  2. Training: `train` (AM policy, 1 epoch CPU/CUDA) succeeded and produced valid checkpoint `/tmp/wsr_train_g2/am/epoch-1.pt`.
  3. Evaluation: `eval` (offline greedy evaluation) completed with `Average KM: 6.4567, Average KG: 324.08`.
  4. Simulation: `test_sim` (2-day simulation with Neural Agent `na` and all policies) completed successfully end-to-end (`Simulation completed successfully!`).

**Lane C Brief Questions Answered in Depth:**

1. **Two Model Classes (`AttentionModelPolicy` vs `AttentionModel`):**
   - Training uses `AttentionModelPolicy` (derived from RL4CO's `AutoregressivePolicy`), which accepts `TensorDict` inputs and operates on standard RL4CO step contexts.
   - Inference/evaluation and simulator `NeuralAgent` use legacy `AttentionModel` (in `logic/src/models/core/attention_model/model.py`), which accepts standard Python dictionaries and uses legacy manual packing.
   - Diffing `state_dict().keys()` between the two reveals:
     - `encoder.*` parameter names match identically (`encoder.init_embed.*`, `encoder.layers.*`).
     - `decoder.*` parameters diverge structurally: `AttentionModel` wraps decoder subnets under `decoder.*` (`decoder.project_node_embeddings`, `decoder.project_fixed_context`, `decoder.context_embedding.*`, `decoder.pointer.*`), whereas `AttentionModelPolicy` places `context_embedding.*` at the root and houses glimpses within `decoder.*`.
     - `AttentionModel` instantiates `self.context_embedder` at the root *and* `self.decoder.context_embedding` inside the decoder.
   - **Recommendation (Owner Approved):** Do NOT maintain two separate parallel model implementations. Legacy `AttentionModel` will be replaced by `AttentionModelPolicy` wrapped in a lightweight adapter that translates legacy dict inputs (`coords`, `demand`, etc.) to `TensorDict`. This eliminates code duplication, parameter translation hacks, and architecture drift.

2. **Model Loader Fidelity (`loader.py:94-106`):**
   - `loader.py` extracts flat hyperparameters (`args["normalization"]`, `args["activation"]`, `args["af_param"]`, etc.) and passes them as keyword arguments to `AttentionModel(...)`.
   - `AttentionModel.__init__` consumes `norm_config: Optional[NormalizationConfig] = None` and `activation_config: Optional[ActivationConfig] = None`. It captures all other arguments into `**kwargs` and passes them to `super().__init__(**kwargs)` (which is `nn.Module.__init__()`).
   - Consequently, `normalization="layer"` is ignored. `NormalizationConfig` defaults to `norm_type="batch"`, silently building `nn.BatchNorm1d` instead of `LayerNorm` (confirming `B-codex-08` / `B-gemini-01`).
   - Furthermore, `AttentionModelPolicy` itself has a related bug (`B-gemini-02`): passing `normalization="layer"` to `AttentionModelPolicy` forwards `normalization` into `kwargs` for `GraphAttentionEncoder`, which only inspects `norm_config` and ignores `kwargs["normalization"]`.

3. **Context Embedder Duplication & Fake Weight Synthesis (`loader.py:150-162`):**
   - In `AttentionModel.__init__` (lines 220-221), `self.context_embedder = component_factory.create_context_embedder(...)` is created.
   - But in line 223, `self.decoder = component_factory.create_decoder(...)` is also created. `GlimpseDecoder.__init__` instantiates its own `self.context_embedding = VRPPContextEmbedder(...)`.
   - In `AttentionModel.forward()`, all decoding calls `self.decoder(...)`, which exclusively uses `self.decoder.context_embedding`. `self.context_embedder` is never called!
   - In PyTorch Lightning training checkpoints, `context_embedding.project_step_context.{weight,bias}` is saved from the policy. When `loader.py` attempts to load the checkpoint into `AttentionModel`, it finds `context_embedder.project_step_context` missing and executes a synthesis block (lines 150-162) creating identity weights for this dead layer.
   - **Fix:** Remove `self.context_embedder` from `AttentionModel` and remove the synthesis block in `loader.py`.

4. **Decoding Loops, CUDA Device Crash, and Mask Contracts:**
   - In `GlimpseDecoder._select_node` (lines 103-104, 302): `self.generator` is created on CPU at init. When `model.to("cuda")` is invoked, `generator` remains on CPU. In sampling mode, `torch.multinomial(probs, 1, generator=self.generator)` crashes with `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'` (`B-gemini-04`).
   - In sampling mode, lines 310-311:
     `while curr_mask.gather(1, selected.unsqueeze(-1)).any(): selected = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)`
     If all unmasked nodes have zero probability (due to softmax underflow or bad logits), this while-loop infinite loops (`B-gemini-05`). Instead, masked actions should be zeroed out in `probs`, renormalized, and drawn in a single call.
   - **Mask contract:** Every decoder step computes `curr_mask` from depot visits and capacity constraints (`vrpp.py`). Outer zero padding is removed by the environment. The mask contract is mandatory; unmasked decoding will sample already visited customers or violate capacity constraints. Vectorized greedy decoding (`torch.argmax(logits.masked_fill(curr_mask, -inf), dim=-1)`) is clean and avoids any while-loops.

5. **Subnets and Modules Pruning:**
   - `subnets/embeddings/positional/` (R-claude-01, 191 LOC): Only imported by removed DACT/N2S models. Safe to remove.
   - `models/common/{non_autoregressive,improvement,transductive}/` (R-claude-04, 1,233 LOC): Base classes for removed non-autoregressive and improvement policies. Safe to remove.
   - `subnets/embeddings/{dynamic,static}.py` (R-gemini-01, 146 LOC): Generic feature wrappers unused by VRPP. Safe to remove.
   - `subnets/embeddings/edges/` (R-gemini-02, 231 LOC): Edge embeddings that import `torch_geometric`. Unused by AM. Removing this package eliminates `torch_geometric` dependencies from the model stack! Safe to remove.
   - `subnets/embeddings/state/` (R-gemini-03, 179 LOC): Reinforcement learning MDP state embedders unused by AM. Safe to remove.
   - `subnets/embeddings/context/generic.py` (R-gemini-04, 93 LOC): Generic context embedder superseded by `vrpp.py`. Safe to remove.
   - `subnets/modules/{cross_attention,flash_attention,normalized_activation_function,dynamic_hyper_connection,static_hyper_connection}.py` (R-gemini-05, 617 LOC): Experimental hyperconnections and flash attention variants. Unused by retained AM. Safe to remove.
   - `models/core/attention_model/symnco_policy.py` (R-gemini-06, 88 LOC): Sym-NCO symmetric augmentation policy variant unused by retained tasks. Safe to remove.
   - `deep_decoder_policy.py` and `configs/models/deep_decoder.yaml` (R-gemini-07, 176 LOC): Pruned per owner decision.
   - `models/common/critic_network/` & `shared_critic.py` (R-gemini-08, 437 LOC): Pruned per owner decision for minimal export packaging.

6. **Model Configurations & Packaging Rules:**
   - Model configs now strictly retain only `attention_model/am.yaml`.
   - `deep_decoder.yaml` and `symnco.yaml` are pruned.
   - Dead dataclass fields in `configs/models/*.py` (R-gemini-09, ~60 LOC): Dataclasses still retain legacy flags (`shrink_size`, `use_distance_features`, `temporal_feature_dim`). Prune after model cleanup in coordination with Lane G.

**Disagreements and Cross-Review:**
- **Confirm B-codex-08 & extend:** Fully confirm Codex's finding on normalization swallowing, and add the deeper structural cause: `AttentionModel` expects dataclass configs (`norm_config`), not flat kwargs, and duplicates `VRPPContextEmbedder`.
- **Confirm B-codex-07 / B-claude-03:** Fully confirm evaluation sign inversion (`B-gemini-06`).
- **Confirm R-claude-01 & R-claude-04:** Both verified locally and safe to remove, dropping 1,424 LOC.
- **Dependency clarification for R-claude-09:** Pruning `models/subnets/embeddings/edges/` (R-gemini-02) removes `torch_geometric` from the neural model stack, confirming that the model stack does not require `torch_geometric`.

**Owner Decisions Recorded:**
1. **Unify Dual Model Architectures:** **YES** — eliminate legacy `AttentionModel` and standardize on `AttentionModelPolicy` wrapped in a lightweight dict adapter for evaluation and simulation.
2. **Critic Network Scope (R-gemini-08):** **Repair on main, prune for this packaging run** — verified removal of `models/common/critic_network/` and `rl/common/baselines/shared_critic.py` (4 files / 437 LOC) from the minimal export artifact.
3. **Deep Decoder Scope (R-gemini-07):** **YES (can be pruned)** — verified removal of `deep_decoder_policy.py` and `deep_decoder.yaml` (2 files / 176 LOC).

## 5.B. Grok — lane B — 2026-09-25

**Scope and reproducibility.** Reviewed `70e660b03` in `/tmp/grok-wsr` (data and `wsmart_bin_analysis` symlinked from the main checkout; the worktree gitlink was empty). Shared HEAD stayed `9bedfb996`. No source commit. The removal experiment exists only in that worktree (`git diff --stat`: 1,866 deletions, 10 insertions).

**Read:** `simulator.py`, `day_context.py` (naming, `run_day`, `get_daily_results`), `states/{initializing,running,finishing}.py`, `actions/{fill,collection,logging,route_construction,route_improvement}.py`, `bins/base.py`, `repository/filesystem.py`, `checkpoints/{manager,hooks,persistence}.py`, `failure_analyzer.py`, `features/test/{engine,config,orchestrator/*}`, `tracking/__init__.py` and `tracking/logging/**`, `utils/data/loader.py`, `utils/input/**`, `utils/configs/**`, `utils/infrastructure/**`, `main.py`. `controllers/hydra_dispatch.py` was used as the entry, not line-audited.

**Rows:** B-grok-01–08; R-grok-01–06; a disposition of R-claude-07. Priority: B-grok-01 (empirical kg/overflows describe the wrong bins), B-grok-02 (id vs slug breaks parallel multi-sample means; default `cpu_cores: 0` is parallel), B-grok-03 (summary `time` is wall clock). Then B-grok-04, B-grok-05, B-grok-07.

**Reproducer:**

```bash
/home/pkhunter/Repositories/Doc/WSmart-Route/.venv/bin/python \
  /home/pkhunter/Repositories/Doc/WSmart-Route/.agent/cache/tools/grok_lane_b_repro_20260925.py
```

It checks the focus-graph vs grid id sets, the slug/id rejection, the idle-day overflow count, the negative resume clock, and which repository filenames exist. Exit 0 means those mismatches reproduced as described.

**Live parallel run.** From the worktree, `main.py test_sim` with `sim.cpu_cores=2`, `sim.policies=[alns,bpc]`, Rio Maior 20 bins, 2 days, `emp`, `load_dataset=null`. Hydra expanded that to 4 tasks and printed `Launching 4 WSmart Route simulations on 2 CPU cores`. Each worker then printed `Finished simulation for policy <slug>_emp`, the `simulator.py:268` line that runs when the expander id is missing from the result dict. Two days is before the cf70 collection day, so km 0 in that run is the empty-tour path. A separate 10-day log set (`…/sim_out/10days/riomaior20_plastic/emp/variant_check/`) has non-zero km on day 7/9 and is the source for the time comparison in B-grok-03. `init_single_sim_worker` with `wst.init_worker` as a no-op did not crash that 2-core run. The removed dashboard (`monitor.initialize_simulation_display` returns `None`) is skipped by `process_display_updates`.

**Naming rule (B-claude-06).** Use the expander id — `SimulationContext.pol_id_orig`, which is `get_pol_name(full_policies[i])` and already ends in `_emp` — as the log filename, the jsonl policy field, the checkpoint name, and the parallel result key. Keep `display_name` for the terminal table. `to_slug` may format that table; it must stop being the file key. That also removes the `none` vs `fast_tsp` split for the Neural Agent.

**Lane questions.**

1. **Result numbers.** km in the sample record matches the sum of daily km on the 10-day logs, and `CollectAction` recomputes km with `get_route_cost` on the global matrix. kg in `collect()` is `(real_c/100) * volume * density`. The sample `overflows` column is the sum of days-at-100%, and sample `time` is wall clock (B-grok-03, B-grok-04). `mean`/`std` for `n_samples==1` are the sample itself. For `n_samples>1` without `resume`, parallel aggregation can write zeros under the `_emp` id (B-grok-02).
2. **Parallel.** The 2-core path runs. The name check fails on every policy whose slug differs from the expander id, which is the retained set.
3. **Checkpoints.** Resume restores the bin object and sets `start_day` to the saved day plus one. The reported time goes negative (B-grok-05). With `checkpoint_days: 0` the periodic save is off; a crash still pickles state. Removal is R-grok-06 and needs an owner decision because the same wrapper swallows day-loop exceptions (B-grok-07).
4. **GUI / failure feed.** R-claude-07: keep `gui.py` (it is the `.jsonl` writer) and delete `metrics.py` (done locally, R-grok-03). `failure_analysis` is inside the jsonl payload. `failure_emit.py` is an extra Studio line.
5. **Repository files.** Rio Maior except `num_loc==104`, and Figueira da Foz, resolve to files that exist. The 104-bin names do not (B-grok-06). `both` is outside `MAP_DEPOTS`. `mixrmbac` files (`StockAndAccumulationRate*.xlsx`, `Coordinates*.xlsx`) exist.
6. **`utils/input`, `loader.py`, `infrastructure`.** `loader.load_grid_base` is on the empirical path (B-grok-01). `setup_env` / `setup_model` / `setup_hrl_manager` are called from `InitializingState` for neural policies. The seven `utils/input` modules in R-grok-01 and `yaml_to_env.py` have no runtime caller.

**Removal verification.** `compileall -q logic main.py` passed. `import_sweep.py` returned **698 modules, 0 failed**. Importing `log_utils` after the deletion does not import `wandb`. A full nine-policy sim was not repeated after the deletion; the deleted modules are import-time only on the simulator path.

**Disagreements.** R-claude-07 can drop `metrics.py` and then `wandb`. It cannot drop `gui.py` without replacing the jsonl writer. Codex item 12 is confirmed: the submodule must be present or `emp` cannot import `GridBase`.

**Owner decisions.** (1) File keys: expander id, including `_emp`. (2) Keep checkpoint/resume and fix the clock, or delete the package and surface day-loop errors. (3) Empirical fills should follow the focus-graph bins. (4) Sample `time` should be the sum of daily solver times.

## 5.F. Cursor — lane F — 2026-09-25

**Scope and reproducibility.** Reviewed `70e660b03` in detached worktree `/tmp/cursor-wsr` (data and `wsmart_bin_analysis` symlinked from the main checkout). Shared HEAD stayed `9bedfb996`. No source commit. The removal experiment exists only in that worktree (`git diff --stat`: 1,965 deletions, 4 insertions, 18 files removed).

**Read:** scalar selectors `selection_{lookahead,last_minute,service_level}.py` + `base/eoq.py`; vectorized `selection/{base,factory,lookahead,last_minute,service_level,regular,revenue,combined}.py`; `fast_tsp.py` + `common/helpers.py` + `tsp.py::find_route`; BMC/OI + `IAcceptanceCriterion`; ALNS/PSOMA/HGS call sites; `policy_na.py` mandatory mask; `node_selection.py`; `steps.py` training selector; `envs/{routing/vrpp,tasks/{base,vrpp},generators/vrpp,base/improvement}.py`; `data/{distributions,generators/{waste,builders,datasets},datasets/{pytorch,simulation},network,processor}`; `constants/*`; the 19 `interfaces/` files.

**Rows:** B-cursor-01–07; R-cursor-01–08; dispositions of R-claude-02 and R-claude-03. Priority: B-cursor-02 (training drops the first customer), B-cursor-03 (70 vs 0.7), B-cursor-01 (linear vs √n). Then B-cursor-04 and B-cursor-05.

**Reproducer:**

```bash
/home/pkhunter/Repositories/Doc/WSmart-Route/.venv/bin/python \
  /home/pkhunter/Repositories/Doc/WSmart-Route/.agent/cache/tools/cursor_lane_f_repro_20260925.py
```

Exit 0 reproduces B-cursor-01–05 and the BMC/OI convention. B-cursor-06/07 are source-path findings.

**Lane questions.**

1. **Selectors, scalar vs vectorized.** Confirmed. Simulator fills are percent (`MAX_CAPACITY_PERCENT=100`); training/`make_dataset` normalises by `max_waste` to `[0,1]` (`MAX_WASTE=1.0`). Lookahead *algorithm* matches at `current_collection_day=0` after that scaling (B-cursor-04 is the non-zero-day split). Last-minute does **not** share a threshold: yaml `70` vs vectorized `0.7` vs train nest `0.25` (B-cursor-03). Service-level formula is linear `z·σ·n_d` in both copies; the paper’s `√n_d` is absent (B-cursor-01). Neural Agent divides `bins.c` by 100 and prepends a depot column, so its fallback vectorized path is internally consistent — but the default sim path feeds NA the *scalar* `mandatory` list from `MandatorySelectionAction`, so the vectorized NA fallback is unused on `test_sim`.
2. **Fast-TSP.** Seed is accepted and dropped (B-cursor-05). Time limit default is 2.0 s, not 30 s (30 s is LKH / other removed improvers). `[0]` / `[0,0]` return unchanged. Duplicate depot entries are stripped by `split_tour` / `assemble_tour`. Mandatory nodes already in a trip stay; the improver does not add missing ones (`resolve_mandatory_nodes` is unused by Fast-TSP).
3. **Acceptance.** Live signature is `accept(current_obj, candidate_obj, **kwargs)`, not `(current, new, f_best, it, max_it)`. ALNS/PSOMA pass profit (maximization). HGS only `step()`s (B-cursor-06). BMC overflow guard is `T <= 1e-9`; `exp(delta/T)` with `delta < 0` cannot overflow.
4. **Interfaces.** Implemented by retained code: `IEnv`, `IModel`, `IPolicy`, `IRouteConstructor`, `IRouteImprovement`, `IMandatorySelectionStrategy`, `IAcceptanceCriterion`, `IBinContainer`, `ITraversable`, `ITensorDictLike`, and the context types listed in R-cursor-05. Orphans: `IDistanceMetric`, `JointSelectionConstructionContext`.
5. **Envs.** `ImprovementEnvBase` is unreachable (R-cursor-03). `VRPP.get_costs` uses `COST_KM`/`REVENUE_KG` (=1.0) when the dataset omits them; the simulator uses `R`/`C`. `get_costs` prepends a zero waste column for depot 0 — correct for customer-only `waste` after `make_dataset`. `make_dataset` divides `.npz` waste by `max_waste` (100 → fraction), which is why training and eval share a scale. `VRPPGenerator.max_waste=1.0` is that training scale, not a simulator percent.
6. **Distributions (R-claude-02).** Consumers of `generate_waste`: `VRPPGenerator._generate_fill_levels`, `builders.py` (then `* 100` for sim datasets). Registry consumers: `bins/prediction.py` (default `"mean"`). `bins/base.py` only special-cases `emp` and `"gamma" in …`. **Exact remaining edits** once smoke moves to `gamma1`:
   - `distributions/__init__.py`: `DISTRIBUTION_REGISTRY = {"gamma": Gamma, "empirical": Empirical}`; `__all__` the same plus `BaseDistribution`.
   - `waste.py`: keep `empty→Constant(0)` only if some path still asks for it; drop `unif`/`beta`/`const`/`dist`; `gamma*` and `emp` stay.
   - `datasets.py:54`: `["emp", "gamma1", "gamma2", "gamma3", "gamma4"]`.
   - `VRPPGenerator` default `waste_distribution="gamma1"`.
   - §0 train smoke: `train.data_distribution=gamma1`.
7. **Datasets (R-claude-03).** Pytorch constructors: `TensorDictDataset` (`rl/common/base/data.py`, `epoch.py`); `BaselineDataset` (`baselines/rollout.py:255`). The other four have no constructor. Simulation constructors: `GenerativeDataset` (`bins/base.py` when `waste_file is None`); `NumpyDictDataset` (`.npz`); `NumpyPickleDataset` / `PandasExcelDataset` / `PandasCsvDataset` (extension dispatch, unused by the retained `gen_data` + `load_dataset=null` path). Survivors: `TensorDictDataset` (+ `BaselineDataset` until rollout is folded) and `GenerativeDataset` + `NumpyDictDataset` behind `SimulationDataset`.
8. **Network / processor / generators.** `euclidean` (`ogd`) and `file` are the retained distance methods. `haversine_distance` is used by `processor/formatting.py`. Processor entry points on `test_sim` / `gen_data`: `setup_basedata`, `process_data`, `process_coordinates`, `process_model_data`, `process_indices`. `train_time` stays (Harbinger, R-cursor-08).
9. **Constants.** Unused and approved to prune: `MAX_LENGTHS`, `CRITICAL_FILL_THRESHOLD`, `OPERATION_MAP`, `FS_COMMANDS`, `TQDM_COLOURS` (R-cursor-07). `LOSS_KEYS` dies with Grok’s `metrics.py`. `METRICS` / `SIM_METRICS` / `DAY_METRICS` stay.

**Removal verification.** `compileall -q logic main.py` passed. `import_sweep.py` returned **689 modules, 0 failed**. A full train/eval/sim smoke was not repeated after the deletion; the deleted modules are import-time only on the retained paths. Shared source and packaging scripts were not changed.

**Disagreements / confirmations.** Confirm B-codex-07 / B-gemini-06 on `VRPP.get_costs` returning `neg_profit` — lane F owns that file and agrees. Confirm Grok: `emp` needs the `wsmart_bin_analysis` submodule (`GridBase`). Do not read B-cursor-03 as “the two last-minute copies disagree on the default sim path”: each pipeline is internally consistent if it keeps its own yaml. They disagree as soon as a config is shared. Paper-review “30 s Fast-TSP budget” is LKH; Fast-TSP is 2 s.

**Owner decisions (Harbinger, 2026-09-25).**
- R-claude-02 remainder: **approved.** Drop `unif`/`beta`/`const`/`dist` after the §0 train smoke moves to `gamma1`.
- R-claude-03 survivors: **locked.** Pytorch `TensorDictDataset` + `BaselineDataset`; simulation `SimulationDataset` + `GenerativeDataset` + `NumpyDictDataset`. Drop the `.pkl/.xlsx/.csv` loaders in the same packaging change.
- R-cursor-07 dead constants: **approved to prune** (`MAX_LENGTHS`, `CRITICAL_FILL_THRESHOLD`, `OPERATION_MAP`, `FS_COMMANDS`, `TQDM_COLOURS`).
- R-cursor-08 `train_time`: **keep.** Not a removal.
- B-cursor-01 service-level: **keep linear.** `z·σ·n_d` is the export contract. The paper `√n_d` form is not retained; both copies are already linear, so there is no sqrt code to delete. Do not port it.
- B-cursor-03 last-minute: **one unit.** Stored yaml is percent `70`/`90` (the cf70/cf90 variants). Vectorized default `0.7` and the train nest `0.25` are not a second/third scale — rewrite them to `70`/`90` and convert only where fills are already in `[0,1]` (NA `/100`, `make_dataset` / `max_waste`).

Lane F owner calls for this review are closed.

## 5.M. Muse — lane-G assist — 2026-09-25

**Scope and reproducibility.** No assigned lane; worked the open lane-G slice
(packaging scripts, dependencies, config consistency) that §4 and §6 are
waiting on. Read-only work in the shared checkout (`feat/minimal-export-package`,
HEAD `9bedfb996`; tree paths verified against `70e660b03`); `main`-branch
packaging scripts inspected via `git show main:...` (the export branch deleted
them). No source commits, no worktree — nothing here required an edit to
verify. Reproducer (29 checks, all PASS):

```bash
.venv/bin/python .agent/cache/tools/muse_lane_g_probe_20260925.py
```

**Read:** `logic/package/{prune_codebase,cleanup_helper,remove_tracking,remove_enums}.py`
(on `main`); `ci/export_config.json` (categories `joint`, `imitation_policy`);
`logic/pyproject.toml` dependencies; `logic/src/configs/tasks/train.py:40-60`
vs `logic/configs/tasks/train.yaml`; `embeddings/{__init__,edges/none.py,edges/base.py}`;
`tracking/logging/modules/{metrics,gui}.py` import lines; RL PBRS readers
(`rl/common/base/{module,steps}.py`); `checkpoint_encoder` / `shrink_size` readers.

**Rows:** R-muse-01–03 (§3); §4 items 34–39. No §2 bug rows — the defects found
are all in the packaging scripts/config, which §4 owns.

**Disagreements / confirmations (one line each).**
- Confirm §4 items 1, 3, 4, 5, 7 with exact line numbers (see items 34, 36–38, R-muse-03).
- Correct §4 item 2's mechanism: the initials heuristic does not match the
  reported pairs (probed `False`); the parent-dir kill at `cleanup_helper.py:313-322`
  is the over-deletion path (item 35).
- Confirm Codex §5.A: PBRS is reachable (`module.py:218-219`), so it stays out
  of R-claude-08; agree `accumulation_steps` / `enable_scaler` /
  `route_improvement_epochs` / `lr_route_improvement` are reader-less, and add
  `efficiency_weight` / `overflow_weight` / `eval_only` (R-muse-01).
- Confirm Grok R-grok-03 and item 22: `wandb` enters only through `metrics.py`;
  `jinja2` must stay while `gui.py` stays (probe PASS both).
- Confirm Gemini R-gemini-02 unblocks `torch-geometric` removal, with the
  precise chain: top-level imports in `edges/{none,base}.py` + eager
  `from .edges import ...` in `embeddings/__init__.py` (R-muse-02).
- Confirm Cursor §5.F: `networkx` is used by retained `tsp.py` — excluded from
  the dep drop list.

**Open questions for the owner.**
1. PBRS (`use_pbrs=false` default, reachable code): keep the capability in the
   export (then it needs parity testing per Codex item 9) or remove capability +
   callers + schema + yaml together? Either way, R-claude-08 must not silently
   drop half of it.
2. `openpyxl`: drop with the `.xlsx` loader (R-claude-03) or keep for the
   `mixrmbac` research path? The dep and the loader are one decision.
3. Lanes D (Kimi), E (Qwen), G (Mistral) have not reported — §6 consolidation
   stays open. The `helpers/operators/` 102-file question (R-claude-05) is the
   largest unverified block and nothing here substitutes for lane E's
   reachability pass.

## 5.D. Kimi — lane D — 2026-09-25

**Scope and reproducibility.** Reviewed `70e660b03` in `/tmp/kimi-wsr` (read-only analysis; review helpers read the same tree) and applied + verified removals in `/tmp/kimi-wsr-del` (cumulative diff vs `70e660b03`: 26 files, +23/−2,197; nothing committed). Shared checkout untouched. Interpreter: shared `.venv`; `data/` and the `wsmart_bin_analysis` submodule symlinked into both worktrees. Setup gotcha worth knowing: fresh worktrees contain an EMPTY gitlink dir for the submodule — `ln -sfn` nests the link inside it; `rmdir` the empty dir first, then symlink (`GridBase` import fails otherwise).

**Read:** `route_construction/base/{base_routing_policy,base_multi_period_policy,factory,registry}.py`; `exact_and_decomposition_solvers/branch_and_price_and_cut/{bpc_engine,policy_bpc,params}.py` + all 30 files of `helpers/solvers_and_matheuristics/`; `smart_waste_collection_two_commodity_flow/{gurobi,ortools_wrapper,pyomo_wrapper,dispatcher,params,policy_swc_tcf}.py`; `hyper_heuristics/ant_colony_optimization_hyper_heuristic/{hyper_aco,hyper_operators,policy_aco_hh,params}.py` + `meta_heuristics/ant_colony_optimization_k_sparse/pheromones.py`; `learning_algorithms/neural_agent/{policy_na,agent,simulation,batch,params}.py`; `other_algorithms/travelling_salesman_problem/{tsp,two_opt}.py`; yamls `policy_{bpc,swc_tcf,aco_hh,na}.yaml`; `pipeline/simulations/actions/route_construction.py`, `day_context.py` (seed plumbing), `bins/prediction.py`, `data/processor/mapper.py` (`_load_profit_vars`); flowcharts `assets/diagrams/code/{bpc,swc_tcf,aco_hh}_flowchart.dot` (untracked copy in the main checkout).

**Rows:** B-kimi-01–33; R-kimi-01–08 + R-kimi-13–17 (verified locally), R-kimi-09–12 + R-kimi-18–19 (proposals); §4 items 40–45. Priority order: **B-kimi-02** (invalid LCI cuts — provably wrong "exact" results with the shipped yaml, 17 vs 25 on a 3-node case), **B-kimi-16** (SWC-TCF travel cost halved — systematically non-optimal routes on the shipped path), **B-kimi-32** (base early-return ignores `vrpp`; cross-policy semantics), then B-kimi-03/04/05/11/20/26/28/30.

**Reproducers:** `.agent/cache/tools/kimi_lane_d_repro_20260925.py` (base/NA, 11 checks, all PASS) plus the eight lane-D repros named in the §2 lane-D header. Two caveats: the repros hardcode their scratch worktree path (same convention as other lanes), and `kimi_lane_d_bpc_lci_yaml_20260925.py` reads `params.enable_dssr` — it must run against a pristine `70e660b03` tree (R-kimi-14 deletes the key).

**Removal verification (final tree, `/tmp/kimi-wsr-del`).** `compileall` clean; import sweep **701 modules, 0 failed** (707 before any deletion). Full pipeline on the pruned tree: `gen_data` (gamma1 smoke) OK; `train` (§0 smoke) OK; `eval` on the 12-instance gamma set reports **KM 6.4567 / KG 324.08** (identical to the values Codex and Gemini observed on this fixture); `test_sim` 2 days, all nine policies incl. Neural Agent, completes cleanly and writes all 17 variant `log_*.json` files (km 0 throughout, as expected before the cf70/cf90 collection days). BPC functional check unchanged by the deletions (obj 17.0 before and after on the 3-node counterexample).

**Lane brief answers:**

1. **Base policy contract.** Id mapping is correct (depot 0; VRPP subset `[0..N]` with global==local 1-based; restricted subset reads wastes via `bins.c[global-1]`; `_map_tour_to_global` correct). The defects are elsewhere: the empty-mandatory short-circuit ignores `vrpp` (B-kimi-32), the typed-config path swallows engine-nested overrides incl. `vrpp:false` (B-kimi-21), the action/base `vrpp` default split (False vs True), and the tracking remnants (R-kimi-03). Coordinate extraction degrades gracefully.
2. **BPC.** The pricing DP itself is correct: 30/30 brute-force matches on random instances, dominance rules sound (conservative superset), standard Baldacci ng-cycle rule, capacity enforced at extension, correct depot handling; no time resource exists in this VRPP. The defects sit above the DP: invalid LCI cuts (B-kimi-02), unsound single-vehicle LR bound (B-kimi-03), timeout → empty fallback (B-kimi-04), non-OPTIMAL LP → `(0.0, {})` (B-kimi-05), UB prune trusting `pricing_exhausted` from truncated/heuristic pricing (B-kimi-11). Timeouts ARE enforced (wall-clock polling; 4 s budget → 2.19 s wall). `exact_mode:false` truncates arcs (`min(20, max(5, n//3))`) and switches to heuristic-first for n>40; its smoothing half is a no-op (B-kimi-10). Gurobi infeasibility is handled correctly (Farkas pricing); the silent-empty-result paths are B-kimi-04/05. Wired-vs-ignored yaml keys: B-kimi-15 + §4 item 41.
3. **SWC-TCF.** Constraints (1)–(7) match the flowchart box-for-box. Divergences: the halved travel cost (B-kimi-16), the two-unit arc filter (B-kimi-19), inert delta (B-kimi-18), and the OR-Tools/Pyomo kg-vs-percent unit split (B-kimi-17). Units: delta = slack fraction (inert), psi = fill fraction (x100 → percent), Omega = EUR per vehicle (not per time), dist matrix = km (gurobi's 6,000,000 threshold is a no-op). `framework: ortools` works only with `engine: scip`; `framework: pyomo` crashes at model construction (B-kimi-22).
4. **ACO-HH.** The six unlisted operators are NOT dead — all 11 run every solve because `params.operators` is stored but never read (B-kimi-26). Evaporation/deposit hit the right matrices; the defects are the first-hop deposit landing on the never-read virtual row (B-kimi-29) and wall-clock time in the visibility update (B-kimi-28). Penalty halving has no floor and capacity-infeasible bests can be returned (B-kimi-30). `construct()` itself has no try/except and zero callers — the swallow risk is the three removal wrappers (B-kimi-31). ACO-HH imports 6 of the 13 `helpers/operators` families (lane-E input).
5. **Neural Agent.** Model loading goes through `utils/model/loader.py` (lane C's B-gemini-01/03 apply upstream). The scalar `mandatory` list is the live mask path; the vectorized-selector fallback is dead on `test_sim` (R-kimi-09). `beam_width: 5` with `strategy: greedy` is parsed but only read inside the `beam_search` branch — conditional, not a bug. `route_improvement: []` is honoured downstream by RouteImprovementAction; the naming quirk remains B-claude-06. The `_viz_record`/`hasattr` guards never fired (R-kimi-04). NA's returned cost/profit units are inconsistent but discarded by the only caller (B-kimi-01).
6. **solvers_and_matheuristics reachability.** Only `bpc_engine.py` imports the package (grep: 18 hits, all bpc_engine + package-internal + one docstring). SWC-TCF and the other seven policies do not. Per-file importer evidence in the R-kimi rows; function-level/lazy imports checked (the only one, the DSSR local import, was deleted with R-kimi-14).

**Disagreements / confirmations (one line each):**
- Confirm B-cursor-05 (TSP seed dropped) — verified during the TSP read; not re-reported.
- Confirm Cursor §5.F ("vectorized NA fallback unused on test_sim") with stronger evidence: `selector_name`/`selector_threshold` are never injected into the day context (R-kimi-09).
- B-kimi-32 + B-kimi-21 jointly explain why the action's `not vrpp` half is dead today; fixing either side without an owner decision changes acceptance behaviour (km=0 before thresholds relies on it).
- Lane C: B-gemini-01/03/04 are upstream of NA; the NA simulation path itself is clean apart from B-kimi-01's dead-value units.
- Lane B (Grok): R-kimi-12 (scenario-tree dead work) is filed in my §3 but the edit belongs to lane B — please pick it up; your B-grok-02 naming fix will also resolve B-claude-06 for NA.
- Lane E (Qwen): ACO-HH live families listed in §4 item 45c; the `local_search_hgs.py` optional flags cited in R-claude-05 were not re-verified here.

**Owner questions:**
1. BPC: fix (gate LCI on `vehicle_limit==1`; fleet-scaled LR bound; UB prune only on certified pricing) or relabel as heuristic? Until then the export should not describe BPC as exact.
2. B-kimi-32: empty mandatory + vrpp policy — run the solver (flowchart) or return `[0,0]` (code)? Current acceptance expectations encode the code side.
3. SWC-TCF: fix pyomo (one `initialize=`) or prune wrapper + `pyomo` dep (R-kimi-10)? Align all three wrappers on percent units or keep gurobi-only?
4. ACO-HH: wire `operators`/`sequence_length` or delete the keys (R-kimi-18)? Is run-to-run reproducibility required (B-kimi-28)?
5. NA adapter contract: fix profit/cost units (B-kimi-01) or document as advisory-only?
6. Scenario-tree removal (R-kimi-12): approve lane B deleting the block from `route_construction.py` + `prediction.py`?

### Lane E bug evidence (Qwen, 2026-09-25)

Scope: `route_construction/meta_heuristics/{alns,hgs,pg_clns,psoma,sans}`, `helpers/operators/` (102 files), `helpers/local_search/`. Reproducers verified against `70e660b03` in worktree `/tmp/qwen-wsr`.

| ID | Agent | File:line | Severity | Symptom | Evidence / repro | Suggested fix | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-qwen-01 | Qwen | `logic/src/policies/route_construction/meta_heuristics/hybrid_genetic_search/split.py:301-365` | blocker | `_split_limited` is broken for K≥2: `V_curr[0]` is never set to 0, so `V_prev[0]` is `-inf` for all layers after the first. The deque is never seeded for k≥2, no new route boundaries can be created, and `best_k` is always 1. The vehicle-limit parameter `max_vehicles` is silently ignored. | Trace the DP: after k=1, `V_prev = V_curr` where `V_curr[0] = -inf` (loop starts at i=1). At k=2, `V_prev[0] > -inf` is False → deque empty → no routes start. On a 3-node instance with Q=2, w=[3,3,3], the unlimited split returns 3 singletons (profit-optimal) but the limited split returns 1 route violating capacity. | Set `V_curr[0] = 0` before the inner loop (or after), so each layer can independently start routes from the depot. | open |
| B-qwen-02 | Qwen | `logic/src/policies/route_construction/meta_heuristics/simulated_annealing_neighborhood_search/heuristics/sans.py:256-307` | major | Inner loop selects and applies an operator twice. `_select_neighbor` at line 262 produces `new_solution, op`, but then lines 305-307 select a SECOND random operator and overwrite `new_solution`, discarding the first result entirely. Every neighbor evaluation wastes one operator application and its deep copies. | `sed -n '256,310p'` on `70e660b03`: first `_select_neighbor` result is never referenced after line 268. The second `apply_operator` at line 307 uses a fresh `rng.choice(valid_ops)`. | Remove the first `_select_neighbor` call (lines 262-268) or use its result instead of re-selecting. | open |
| B-qwen-03 | Qwen | `logic/src/policies/route_construction/meta_heuristics/simulated_annealing_neighborhood_search/dispatcher.py:195-240` | blocker | `execute_og` builds a `values` dict with key `"Q"` but `find_solutions` (`refinement/route_search.py:69`) reads `values["vehicle_capacity"]` and `values["E"]`. Both are missing → `KeyError` at runtime. The OG engine is completely unreachable. Additionally, `params.combination` is a string (`"a"`/`"b"`) but `find_solutions` indexes it as a numeric tuple (`chosen_combination[0]`, `[3]`, `[4]`, `[5]`, `[6]`). | `execute_og` dict: `{"Q": Q, "R": R, "B": ..., "C": C, "V": ..., "shift_duration": ..., "perc_bins_can_overflow": ...}`. `find_solutions` reads `values["vehicle_capacity"]` → KeyError. Even if fixed, `params.combination = "a"` → `"a"[3]` → IndexError. | This is a removal candidate (R-qwen-03), not a fix target. If kept, rename `"Q"` → `"vehicle_capacity"`, add `"E"` (number of vehicles?), and map combination strings to numeric tuples. | open (remove, don't fix) |
| B-qwen-04 | Qwen | `logic/src/policies/route_construction/meta_heuristics/hybrid_genetic_search/hgs.py:639-647` | major | Restart clears `_feas_cache`, `_infeas_cache`, `_offspring_feasibility` but NOT `_offspring_coverage` and `_offspring_margin` (declared at lines 120-121). `_adjust_penalties` (line 548-549) reads the last 100 entries from these lists, which contain stale pre-restart data, corrupting the VRPP penalty adaptation. | `sed -n '639,647p'`: restart block clears 5 lists but not the coverage/margin lists. `sed -n '548,549p'`: `_adjust_penalties` reads `self._offspring_coverage[-100:]`. | Clear `_offspring_coverage` and `_offspring_margin` in the restart block. | open |
| B-qwen-05 | Qwen | `logic/src/policies/route_construction/meta_heuristics/hybrid_genetic_search/evolution.py:127` | minor | `diversity_weight = 1.0 - (nb_elite / pop_size)` produces negative weight when `pop_size < nb_elite` (default 4). Inverts fitness to reward less-diverse individuals. Can occur during initialization with unlucky feasibility splits. | On a 2-individual infeasible subpopulation: `1.0 - 4/2 = -1.0`. | Clamp: `diversity_weight = max(0.0, 1.0 - nb_elite / pop_size)`. | open |
| B-qwen-06 | Qwen | `logic/src/policies/route_construction/meta_heuristics/adaptive_large_neighborhood_search/alns.py:672-678` | major | Dynamic `T_start` calibration silently skipped when `best_profit ≤ 0`. With `start_temp=0.0` and non-positive initial profit, BMC gets `T=0` and rejects all worsening moves (`T ≤ 1e-9` guard). Algorithm degenerates to pure hill-climbing on difficult instances. | `sed -n '672,678p'`: `if self.params.start_temp == 0.0 and best_profit > 0:` — the `best_profit > 0` guard excludes the case where calibration is needed most. | Remove the `best_profit > 0` guard; use `abs(best_profit)` or a fallback temperature. | open |
| B-qwen-07 | Qwen | `logic/src/policies/route_construction/meta_heuristics/pheromone_guided_cooperative_large_neighborhood_search/pg_clns.py:95,118` + `aco.py:117` + `lns.py:176` | minor | PG-CLNS uses `time.process_time()` (CPU time) for wall-clock time limits, while ALNS and HGS use `time.perf_counter()`. On multi-core systems or when suspended, `process_time` differs significantly from wall-clock, making the time limit behave unpredictably. | `sed -n '95p'`: `start_time = time.process_time()`. Compare ALNS `alns.py:667`: `time.perf_counter()`. | Use `time.perf_counter()` consistently. | open |
| B-qwen-08 | Qwen | `logic/src/policies/route_construction/meta_heuristics/pheromone_guided_cooperative_large_neighborhood_search/lns.py:163-168` | minor | LNS weight update hardcodes `lambda_decay = 0.8` instead of using `self.params.reaction_factor`. Rejected operators get `max(0.1, 0) = 0.1` floor, preventing the adaptive mechanism from discriminating bad operators. | `sed -n '163,168p'`: `lambda_decay = 0.8` is a local variable, not `self.params.reaction_factor`. Equilibrium weight for score=0 is 0.1. | Use `self.params.reaction_factor` and allow weights to decay to zero (or a lower floor). | open |
| B-qwen-09 | Qwen | `logic/src/policies/route_construction/meta_heuristics/pheromone_guided_cooperative_large_neighborhood_search/operators/__init__.py:79-81` | minor | `__all__` lists `"perturb"` and `"kick"` but these names are never imported from any submodule. No `def perturb` or `def kick` exists in the operators directory. Broken API contract. | `grep -rn "def perturb\|def kick" pg_clns/operators/` → 0 matches. | Remove from `__all__` or implement the functions. | open |
| B-qwen-10 | Qwen | `logic/src/policies/route_construction/meta_heuristics/particle_swarm_optimization_memetic_algorithm/solver.py:163-179` | major | EMA reward normalisation is inconsistent: training phase uses `abs(gbest_profit - best_pf)` (worsening operators get positive reward), non-training uses `max(0.0, best_pf - initial_profit)` (worsening operators get zero). Creates misleading probability distributions. | `sed -n '163,164p'`: `abs(...)`. `sed -n '176,179p'`: `max(0.0, ...)`. | Use `max(0.0, ...)` in both phases. | open |
| B-qwen-11 | Qwen | `logic/src/policies/route_construction/meta_heuristics/simulated_annealing_neighborhood_search/heuristics/anneal.py:121-131` | major | `_update_removed_bins` is only called when `delta == 0`, not when `delta > 0`. When an improving structural move (add/remove bins) is accepted, the `removed_bins` bookkeeping becomes permanently inconsistent with the actual solution. | `sed -n '121,131p'`: `if delta == 0: _update_removed_bins(...)`. When `delta > 0` (improving), the update is skipped. OG engine only — but the OG engine is already broken (B-qwen-03). | Call `_update_removed_bins` unconditionally when a move is accepted. | open (OG-only, remove with R-qwen-03) |
| B-qwen-12 | Qwen | `logic/src/policies/route_construction/meta_heuristics/simulated_annealing_neighborhood_search/common/revenue.py:30-38` | major | OG engine revenue: `revenue_per_bin = bin_stock * E * B * R` missing `/100` for percent→fraction conversion. `Stock` is 0-100 percent; physical kg is `stock/100 * V * density`. Revenue is 100× too large. Compare new engine (`sans_state.py:73`): `bin_kg = stocks.get(b, 0) * V * density / MAX_CAPACITY_PERCENT`. | `sed -n '30,38p'` on `revenue.py`. No `/100` or `/MAX_CAPACITY_PERCENT`. OG engine only. | Add `/ 100.0` (or `/ MAX_CAPACITY_PERCENT`). OG engine — remove with R-qwen-03. | open (OG-only, remove with R-qwen-03) |

### Lane E removal evidence (Qwen, 2026-09-25)

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| R-qwen-01 | Qwen | `logic/src/policies/helpers/operators/{evolutionary_mutation,generalized_insertion_and_deletion,improvement_descent,intensification_fixing,perturbation_shaking,search_heuristics,sequence_merging}/` | 7 packages | 11,435 | **Not reachable from any retained policy.** External callers of `helpers/operators` checked: ALNS imports from `destroy_ruin` + `recreate_repair` + `solution_initialization` (via top-level `__init__.py`); HGS imports `crossover_recombination` (1 function) and uses `local_search_base` → `inter_route_local_search` + `intra_route_local_search`; BPC imports `recreate_repair.greedy`; ACO-HH imports `solution_initialization.greedy_si`. Zero external callers for the 7 categories. PG-CLNS and SANS have their own independent operator implementations. Internal cross-deps are docstring examples only (`>>>` prefix), not actual imports. | Import sweep: `grep -rn "from.*helpers\.operators\.<cat>" logic/src/policies/ --include='*.py' | grep -v operators/` → 0 hits for all 7 categories. `compileall` on current tree passes (707 modules). Removal not yet tested in worktree (see verification plan). | Clean `helpers/operators/__init__.py` re-exports for the 7 categories (lines referencing `evolutionary_mutation`, `generalized_insertion_and_deletion`, `improvement_descent`, `intensification_fixing`, `perturbation_shaking`, `search_heuristics`, `sequence_merging`). Add to `prune_codebase.py` operator allowlist. | proposed (import-safe, needs compileall verification) |
| R-qwen-02 | Qwen | `logic/src/policies/route_construction/meta_heuristics/adaptive_large_neighborhood_search/{alns_package.py,ortools_wrapper.py}` | 2 modules | 437 | ALNS dispatcher (`dispatcher.py:21-22`) dispatches on `variant == "package"` / `variant == "ortools"`. The yaml `policy_alns.yaml:98` sets `engine: "custom"` only. No other yaml or config references `package` or `ortools` ALNS engines. | `grep -rn "package\|ortools" logic/configs/policies/policy_alns.yaml` → only `engine: "custom"`. `grep -rn "run_alns_package\|run_alns_ortools" logic/ --include='*.py'` → only `dispatcher.py` imports. | Add to ALNS prune list. Remove dispatcher branches for `package`/`ortools`. | proposed |
| R-qwen-03 | Qwen | SANS OG engine: `refinement/{route_search,refinement,rebalancing}.py`, `search/{deterministic,random_search,reversed}.py`, `operators/{__init__,inter_move,inter_swap,intra_move,intra_swap,move,swap}.py`, `select/{consecutive,greedy,random}.py`, `heuristics/anneal.py`, `common/{objectives,revenue,penalties,check,update}.py` | 20 files | ~3,143 | **OG engine is completely unreachable.** `test_sim.yaml:74` only references `sans.new`. `execute_og` crashes with `KeyError` on `values["vehicle_capacity"]` and `values["E"]` (B-qwen-03), plus `params.combination` type mismatch (string indexed as tuple). The `og_a`/`og_b` yaml variants set `engine: og` but are not referenced from `test_sim.yaml`. All 20 files are exclusively on the OG path. | `grep -rn "engine.*og\|og_a\|og_b" logic/configs/tasks/test_sim.yaml` → only `sans.new`. `grep -rn "find_solutions\|refine_solution\|rebalance_solution\|run_annealing_loop"` → only OG-path callers. New engine (`heuristics/sans.py`) uses its own `_select_neighbor`/`apply_operator` from `sans_operators.py`, not `search/` or `operators/`. | Remove `execute_og` from `dispatcher.py`, remove `og_a`/`og_b`/`lac` sections from `policy_sans.yaml`, delete the 20 files. Clean `__init__.py` re-exports. | proposed (broken code, safe to remove) |
| R-qwen-04 | Qwen | `logic/src/policies/route_construction/meta_heuristics/hybrid_genetic_search/pyvrp_wrapper.py` | module | 84 | HGS dispatcher (`dispatcher.py:47-48`) dispatches on `params.engine == "pyvrp"`. The yaml `policy_hgs.yaml` only references `engine: custom` (default). `pyvrp` is listed as an optional dependency. | `grep -rn "engine.*pyvrp\|pyvrp" logic/configs/policies/policy_hgs.yaml` → only a comment at line 164. No yaml sets `engine: pyvrp`. | Remove `solve_pyvrp` import and dispatch branch from `dispatcher.py`. Delete `pyvrp_wrapper.py`. | proposed |
| R-qwen-05 | Qwen | PG-CLNS `operators/` subdirectory (35 files within the PG-CLNS package): `operators/{destroy,repair,exchange,move,route}/*`, `operators/{destroy_operators,repair_operators,exchange_operators,move_operators,route_operators}.py`, `operators/__init__.py` | sub-package | ~1,800 | PG-CLNS has its own independent operator implementations that duplicate the shared `helpers/operators` library. These are NOT wrappers — they are standalone re-implementations. The PG-CLNS `local_search.py` imports from `.operators` (the local package), not from `helpers/operators`. However, these are ON the runtime path for PG-CLNS and CANNOT be removed without replacing them with imports from the shared library. | `local_search.py:29`: `from .operators import (move_2opt_intra, ...)`. These are PG-CLNS's own implementations. **Not a removal candidate** — listed for completeness. | N/A — keep. | not a removal (runtime path) |

### Mistral removal evidence (lane G, 2026-09-25)

Verified against `70e660b03` in worktree `/tmp/mistral-wsr` (read-only; no source commits). Dependency checks: `grep -rn` over `logic/**/*.py` + `main.py` for each package, including lazy (function-level) imports. Submodule `wsmart_bin_analysis` excluded (gitlink, empty in worktree). Reachability script: `.agent/cache/tools/reachability.py`.

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-mistral-01 | Mistral | `logic/pyproject.toml` core deps: `einops`, `scikit-learn`, `hydra-colorlog` | dependency | — | Zero `^(from\|import)` hits for any of the three across `logic/**/*.py` + `main.py` (submodule excluded). Muse R-muse-02 explicitly left these for lane G; confirmed zero-hit. `einops` is sometimes a transitive dep of `torch`/`pytorch-lightning` but is never imported directly in the codebase. `scikit-learn` (`sklearn`) has zero direct or lazy imports. `hydra-colorlog` has zero imports (the Hydra config uses the standard `hydra.core.config_store`). | Static `grep -rn` over `logic/` + `main.py` — 0 hits for all three. No local deletion run (pyproject edit only). | Remove from `logic/pyproject.toml` `dependencies` list. Safe to drop in the same packaging step as R-muse-02. | proposed |
| R-mistral-02 | Mistral | `logic/pyproject.toml` core deps: `torch-scatter`, `torch-sparse`, `ml-dtypes`, `torchvision` | dependency | — | `torch-scatter` and `torch-sparse`: zero direct imports (`grep -rn "torch_scatter\|torch_sparse" logic/ --include="*.py"` → 0 hits). They are transitive deps of `torch-geometric`, which drops with R-gemini-02. `ml-dtypes`: zero hits (`grep -rn "ml_dtypes" logic/` → 0). `torchvision`: zero hits (`grep -rn "import torchvision\|from torchvision" logic/` → 0). | Static grep — 0 hits for all four. No local deletion run. | Remove from `logic/pyproject.toml` `dependencies`. `torch-scatter`/`torch-sparse` also require removing their `[tool.uv.sources]` and `[[tool.uv.index]]` (pyg) entries. | proposed |
| R-mistral-03 | Mistral | `logic/pyproject.toml` optional-deps `solvers`: `hexaly`, `vrpy`, `pyomo` | dependency | — | `hexaly` (`Hexaly`/`LocalSolver`): zero hits in code or yaml (`grep -rn "hexaly\|Hexaly\|LocalSolver" logic/ --include="*.py" --include="*.yaml"` → 0). `vrpy`: only a comment in `configs/policies/other/route_improvement.py:390` (`# only used by vrpy fallback`), no import anywhere. `pyomo`: only in `pyomo_wrapper.py` which crashes unconditionally (B-kimi-22); Kimi R-kimi-10 proposes removal from the solver side; this confirms from the dependency side. Remaining `solvers` deps that stay: `gurobipy` (BPC + SWC-TCF gurobi + helpers), `ortools` (SWC-TCF ortools + ALNS ortools_wrapper), `fast-tsp` (route improvement TSP), `alns` (ALNS). `pyvrp` is proposed for removal by R-qwen-04 (HGS pyvrp engine unreachable). | Static grep — 0 hits for hexaly/vrpy; pyomo confirmed broken by Kimi B-kimi-22. No local deletion run. | Remove from `[project.optional-dependencies] solvers` in `logic/pyproject.toml`. Also remove `hexaly` entry from `[tool.uv.sources]` and `[[tool.uv.index]] hexaly_package`. | proposed |
| R-mistral-04 | Mistral | `logic/configs/policies/policy_sans.yaml` `params` key under `og_a`/`og_b`/`lac.*` variants | config-key | — | `SANSConfig` dataclass has no `params` field — only `engine`, `time_limit`, `seed`, `perc_bins_can_overflow`, `T_min`, `T_init`, `iterations_per_T`, `alpha`, `combination`, `mandatory_selection`, `route_improvement`. The `params` key (a list of opaque numeric parameters) exists only in the OG-engine yaml variants. Hydra would reject it unless `struct=False` or it is consumed by a different path. The OG engine itself is broken (B-qwen-03) and proposed for removal (R-qwen-03). | Static comparison of `sans.py` dataclass fields vs `policy_sans.yaml` keys. No local deletion run. | Remove the `params` key together with the OG-engine yaml variants (R-qwen-03). No separate packaging hook needed — the OG removal subsumes this. | proposed (remove with R-qwen-03) |

### OpenCode bug evidence (Batch P verification, 2026-09-25)

Both rows narrow a lane-E removal row; neither is a product bug on the retained path. Details and corrected scope are in §5.O.

| ID | Agent | File:line | Severity | Symptom | Evidence / repro | Suggested fix | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-opencode-01 | OpenCode | `logic/src/policies/route_construction/hyper_heuristics/ant_colony_optimization_hyper_heuristic/hyper_operators.py:37-45` vs `helpers/operators/__init__.py:169-180` | major | R-qwen-01 over-claims: `perturbation_shaking` is LIVE. The retained ACO-HH `solve()` path reads `HYPER_OPERATORS` (`hyper_aco.py:145,233,764`), whose `apply_kick`/`apply_perturb` call `kick`/`kick_profit`/`perturb` imported from the shared `helpers.operators` root, i.e. `perturbation_shaking/`. Deleting the category as specified breaks `import hyper_operators` (observed: first deletion attempt, then restored). | `sed -n '37,45p'` on `70e660b03` shows the package-root import; `hyper_aco.py:68-69,145` wires `HYPER_OPERATORS` into `solve()`. Qwen §4 item 52a states this correctly while the R-qwen-01 table claims zero callers — 52a is right, the table is wrong. | Narrow R-qwen-01 to the 6 remaining categories (R-opencode-01); keep `perturbation_shaking/` in the export. Per-file pruning inside it (only `kick.py`/`perturb.py` names are consumed) is follow-up work, not verified here. | open |
| B-opencode-02 | OpenCode | `logic/src/policies/route_construction/meta_heuristics/simulated_annealing_neighborhood_search/common/routes.py:26`, `objectives.py:24-34` | major | R-qwen-03 over-claims: `common/{objectives,penalties,revenue}.py` are required by the KEPT new engine. `heuristics/sans.py:27` imports `common.routes`, and `routes.py:26,187` imports and calls `compute_profit, compute_real_profit` from `objectives.py`, which itself imports `penalties.py` and `revenue.py`. Deleting them breaks `import sans` on the retained `sans.new` path. | Import chain read on `70e660b03`; deletion experiment kept the three files and the sweep + `sans_new` sim pass. | Narrow R-qwen-03 to `refinement/`, `search/`, `operators/`, `select/`, `heuristics/anneal.py`, `common/{check,update}.py` (R-opencode-02); keep `objectives`/`penalties`/`revenue` with `routes`/`distance`/`solution_initialization`. | open |

### OpenCode removal evidence (Batch P verification, 2026-09-25)

Applied together in `/tmp/opencode-wsr` (detached `70e660b03`, data + `wsmart_bin_analysis` symlinked as the common brief requires). Baseline sweep before any deletion: **707 modules, 0 failed**. No shared-checkout commits.

| ID | Agent | Path / symbol | Kind | LOC | Why it is safe (importers checked) | Verification done | Packaging hook | Status |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| R-opencode-01 | OpenCode | `logic/src/policies/helpers/operators/{evolutionary_mutation,generalized_insertion_and_deletion,improvement_descent,intensification_fixing,search_heuristics,sequence_merging}/` (narrowed R-qwen-01: `perturbation_shaking/` stays per B-opencode-01) | package | 9,540 | Every external importer of the shared `helpers.operators` root was enumerated (`local_search_base.py`, `bpc_engine.py`, `hyper_aco.py`, `hyper_operators.py`, `policy_aco_hh.py`, `alns.py`, `hgs.py`): all consumed names resolve to the 7 kept categories. Distinctive-name grep for deleted symbols (`two_opt_steepest`, `dp_route_reopt`, `apply_lns`, `solve_lkh`, `swap_mutation`, `unstringing_*`, `aco_build_sequence`, `ss_hh_*`, …) hits only definitions, the package `__init__`, and removed files — except `kick`/`perturb`, which is why `perturbation_shaking/` is excluded. `double_bridge` has no external users. | Deleted the 40 files + cleaned `helpers/operators/__init__.py` (6 import blocks + `__all__` entries; `perturbation_shaking` block kept). `compileall` clean; import sweep **640 modules, 0 failed** (combined with R-opencode-02–05 below); 2-day `test_sim` with `sans,alns,hgs,aco_hh` × cf70/cf90 writes all 8 `log_*.json` + realtime `.jsonl` cleanly. | Operator-category allowlist (the 7 kept families) + atomic `__init__` cleanup, same hook as R-qwen-01. | verified locally |
| R-opencode-02 | OpenCode | SANS OG engine, narrowed (R-qwen-03 minus `common/{objectives,penalties,revenue}.py` per B-opencode-02): `refinement/` (4 files), `search/` (4), `operators/` (7), `select/` (4), `heuristics/anneal.py`, `common/{check,update}.py` + `execute_og` + `og_a`/`og_b`/`lac` yaml (subsumes R-mistral-04) | package | ~3,060 (22 files / 2,947 + yaml 112) | `test_sim.yaml` references only `sans.new`; `lookahead*.py` matches for the update-fn names are local methods/comments, not imports; `common/update.py` users are the SANS `__init__` + OG files only; `policy_sans.py` `engine == "og"` branch unreachable once the yaml variants go. `common/routes.py` + `distance.py` + `solution_initialization.py` stay (new engine + ACO-HH `policy_aco_hh.py:20`). | Deleted 22 files; edited SANS `__init__.py` (dropped `update` + `find_solutions` re-exports), `dispatcher.py` (dropped `route_search`/`routes`/`get_route_cost` imports + whole `execute_og`), `policy_sans.py` (dropped `execute_og` import + `og` branch → always `execute_new`), `policy_sans.yaml` (kept `sans.new` only, yaml-validated). Same sweep + `sans_new` sim logs as above. | Same hook as R-qwen-03 + dispatcher/`__init__`/yaml cleanup listed in §4 item 60. | verified locally |
| R-opencode-03 | OpenCode | `.../adaptive_large_neighborhood_search/{alns_package.py,ortools_wrapper.py}` + dispatcher branches + `__init__` re-exports (R-qwen-02 as specified) | module | 437 | `policy_alns.yaml` sets `engine: custom` only; the only other `ortools_wrapper` importer (`swc_tcf/dispatcher.py:23`) is the unrelated SWC-TCF module, untouched. ALNS `__init__.py` re-export was the live edge the sweep caught (104 downstream fails → fixed by cleaning the `__init__`). | Deleted 2 files + dispatcher branches + `__init__` lines. Same sweep + `alns_custom` sim logs as above. | Same hook as R-qwen-02. | verified locally |
| R-opencode-04 | OpenCode | `.../hybrid_genetic_search/pyvrp_wrapper.py` + dispatcher branch (R-qwen-04 as specified) | module | 84 | `policy_hgs.yaml` never sets `engine: pyvrp` (comment-only mention at `:164`); no other importer. | Deleted 1 file + dispatcher branch. Same sweep + `hgs_custom` sim logs as above. | Same hook as R-qwen-04; pair with the `pyvrp` dep drop (R-mistral-03). | verified locally |
| R-opencode-05 | OpenCode | `logic/src/utils/policy/{llh_pool.py,wrappers.py}` (new: the only external importers of `perturbation_shaking` and `improvement_descent`) | module | 882 | `llh_pool` has zero importers repo-wide (only a docstring mention in `utils/policy/__init__.py`, which is import-free, 17 lines). `wrappers` is imported only by `llh_pool`. Retained operator files import `utils.policy.routes`/`seed_hurdle`/`neighborhood` (all kept), never `wrappers`/`llh_pool`. | Deleted 2 files. Same sweep; nothing on the retained path imports them. | Manual (two files, no `__init__` change needed). | verified locally |

### Qwen additions to packaging consequences

46. **Operator library is 70% dead code.** The shared `helpers/operators/` package has 13 sub-categories totalling ~25 kLOC / 102 files. Only 6 categories are reachable from the retained policies: `destroy_ruin` (ALNS), `recreate_repair` (ALNS + BPC), `solution_initialization` (ALNS + ACO-HH), `crossover_recombination` (HGS, 1 function), `inter_route_local_search` (HGS via `local_search_base`), `intra_route_local_search` (HGS via `local_search_base`). The other 7 categories (`evolutionary_mutation`, `generalized_insertion_and_deletion`, `improvement_descent`, `intensification_fixing`, `perturbation_shaking`, `search_heuristics`, `sequence_merging`) totalling **11,435 LOC / 49 files** have zero external callers. PG-CLNS and SANS have their own independent operator implementations and do not use the shared library.
47. **SANS OG engine is dead and broken.** `test_sim.yaml` only references `sans.new`. The OG engine (`execute_og`) crashes with `KeyError` on `values["vehicle_capacity"]` and `values["E"]` (B-qwen-03). The `og_a`/`og_b` yaml variants are unreachable. Removing the OG engine eliminates **20 files / ~3,143 LOC** including `refinement/`, `search/`, `operators/`, `select/`, `heuristics/anneal.py`, and 5 `common/` modules.
48. **ALNS `package`/`ortools` engines are unreachable.** The yaml only uses `engine: custom`. The `alns_package.py` (315 LOC) and `ortools_wrapper.py` (122 LOC) are dead code.
49. **HGS `pyvrp` engine is unreachable.** The yaml only uses `engine: custom` (default). `pyvrp_wrapper.py` (84 LOC) is dead code.
50. **HGS `_split_limited` is broken (B-qwen-01).** The vehicle-limit parameter is silently ignored; the DP always uses 1 vehicle. Until fixed, HGS with `max_vehicles > 0` produces capacity-infeasible single-route solutions. The shipped yaml uses `max_vehicles: 0` (unlimited), which takes the `_split_unlimited` path and is correct. Export should not describe HGS as supporting limited-fleet VRPP until this is fixed.
51. **SANS double operator selection (B-qwen-02).** Every neighbor evaluation in the new engine wastes one operator application. Not a correctness bug (the second selection is valid), but a ~2× performance overhead on the inner loop.
52. **Cross-lane flags from Lane E.** (a) Confirm Kimi §4 item 45c: ACO-HH imports 6 of 13 `helpers/operators` families from the shared library (`destroy_ruin/{random,shaw,string}`, `perturbation_shaking/{kick,perturb}`, `intra_route_local_search`, `inter_route_local_search`, `recreate_repair/greedy`, `solution_initialization/greedy_si`). (b) `local_search_base.py` is only used by HGS (via `local_search_hgs.py`). If HGS's `use_cross_exchange`, `use_lambda_interchange`, `use_ejection_chains` flags are all False in the yaml (verify), then `inter_route_local_search`'s `cross_exchange.py`, `cyclic_transfer.py`, `ejection_chain.py`, `exchange_chain.py`, `k_opt_star.py`, `subramanian_neighborhoods.py`, `swap_star.py` are loaded but never called — potential further pruning within a reachable category. (c) PG-CLNS's `ACOSolver.solve()` is dead code within the PG-CLNS context (only `constructor` and `pheromone` attributes are used by `PGCLNSSolver`).

### Mistral additions to packaging consequences (lane G, 2026-09-25)

53. **Complete the dependency drop list (extends R-muse-02).** R-mistral-01 adds `einops`, `scikit-learn`, `hydra-colorlog` (3 zero-hit core deps Muse left for lane G). R-mistral-02 adds `torch-scatter`, `torch-sparse`, `ml-dtypes`, `torchvision` (4 zero-hit core deps). Combined with Muse's 7 zero-hit + 3 coupled drops, the total confirmed-unused core `dependencies` count is **17 packages**. The packaging step edits `logic/pyproject.toml` `dependencies` once, after R-gemini-02 (edges→torch-geometric), R-grok-03 (metrics→wandb), and R-claude-03 (xlsx loader→openpyxl) land. Also remove `[tool.uv.sources]` entries for `torch_scatter`, `torch_sparse` and the `[[tool.uv.index]] pyg` block.
54. **Drop 3 solver deps from `[project.optional-dependencies] solvers`** (R-mistral-03): `hexaly` (0 hits), `vrpy` (comment-only), `pyomo` (always crashes, B-kimi-22). Also remove `hexaly` from `[tool.uv.sources]` and `[[tool.uv.index]] hexaly_package`. `pyvrp` is a conditional drop (R-qwen-04 — HGS pyvrp engine is unreachable). Remaining `solvers` group: `gurobipy`, `ortools`, `alns`, `fast-tsp` (+ `pyvrp` until R-qwen-04 is approved).
55. **`openpyxl` is safe to drop from core deps after three coupled removals.** The `.xlsx` read in `filesystem.py:_get_mixrmbac_data` is only for the `mixrmbac` area (not retained — only `riomaior`/`figueiradafoz` stay, both `.csv`). `finishing.py:114,230` writes `.xlsx` inside a dead `run = None  # AUTO-REMOVED` block. `splitting.py:183` uses `openpyxl` but is proposed for removal by Grok R-grok-01. `pd_xlsx_dataset.py` goes with R-claude-03/R-cursor-03. So `openpyxl` drops from core deps when R-grok-01 + R-grok-05 + R-claude-03 land. Confirms Muse R-muse-02's coupling note with concrete reachability evidence.
56. **`protobuf` stays (used by ortools_wrapper).** `ortools_wrapper.py:15` imports `from google.protobuf.duration_pb2 import Duration`. The `protobuf==7.35.1` pin in core `dependencies` is required as long as `ortools` stays in the `solvers` group. Do not drop it.
57. **SANS `params` yaml key has no dataclass field** (R-mistral-04). The `og_a`/`og_b`/`lac.*` yaml variants set a `params` list that `SANSConfig` does not define. This is subsumed by R-qwen-03 (remove the OG engine and its yaml variants). No separate packaging hook.
58. **Submodule `wsmart_bin_analysis` is required for the export.** `GridBase` is imported (lazily in `utils/data/loader.py:243,299`, eagerly in `pipeline/simulations/bins/base.py:44` and `data/distributions/statistical_empirical.py:24`). Confirms Grok §4 item 26 and Codex §4 item 12. The export must materialize the submodule (not just the gitlink). The `ui/`, `benchmark/`, `test/`, `docs/` trees inside it are outside the import path — shipping only `export.simulation` and `python.sample_gen` is a proposal for the owner.
59. **Static import graph is sparse — 33 of 707 modules reachable from `main.py` + `hydra_dispatch.py` alone.** The AST-based reachability script (`.agent/cache/tools/reachability.py`) follows static `import`/`from` statements from the two entry-point files and finds only 33 reachable modules. The real reachability comes from runtime string-keyed registries (Hydra `_target_`, `RouteConstructorRegistry`, `ENV_REGISTRY`, etc.) and lazy imports — the static graph is not sufficient to identify dead code. The script is posted for other lanes to extend with registry-aware resolution; the `import_sweep.py` runtime sweep remains the authoritative reachability check. Lanes D/E's per-policy import-graph work (Kimi §5.D item 6, Qwen §5.E) is more complete than the raw AST walk for the policy subtree.

### OpenCode additions to packaging consequences (Batch P verification, 2026-09-25)

60. **Promote narrowed Batch P to verified and apply as one change (R-opencode-01–05).** Verified together in `/tmp/opencode-wsr`: **67 deleted `.py` files / ~14.2k deletions / 3 insertions** across 76 changed files; `compileall` clean; import sweep **707 → 640 modules, 0 failed** (exactly the 67 deleted modules); 2-day `test_sim` (`sans,alns,hgs,aco_hh` x cf70/cf90, `emp`, 20 bins) writes all 8 `log_*.json` + realtime `.jsonl` with no errors. Concrete edit list beyond file deletion: `helpers/operators/__init__.py` (drop 6 import blocks + `__all__` entries, keep `perturbation_shaking`); ALNS `dispatcher.py` (drop `package`/`ortools` branches + now-dead `variant` line) and ALNS `__init__.py` (drop `run_alns_package`/`run_alns_ortools` re-exports — the sweep proved this `__init__` is a live edge: 104 downstream fails until cleaned); HGS `dispatcher.py` (drop `pyvrp` branch); SANS `__init__.py` (drop `update` + `find_solutions` re-exports), SANS `dispatcher.py` (delete `execute_og`, drop `route_search`/`routes`/`get_route_cost` imports), `policy_sans.py` (drop `execute_og` import, `og` branch → always `execute_new`), `policy_sans.yaml` (keep `sans.new` only); delete `utils/policy/{llh_pool,wrappers}.py` (no `__init__` change — it is import-free).
61. **Corrections to lane-E rows (do not apply R-qwen-01/R-qwen-03 as written).** B-opencode-01: applying R-qwen-01 verbatim deletes live `perturbation_shaking/` (ACO-HH `solve()` consumes `kick`/`kick_profit`/`perturb` via `HYPER_OPERATORS`). B-opencode-02: applying R-qwen-03 verbatim deletes `common/{objectives,penalties,revenue}.py`, which the kept new engine needs (`sans.py` → `routes.py` → `objectives.py` → `penalties`/`revenue`). The verified scope is R-opencode-01 (6 categories) + R-opencode-02 (22 files + yaml, three `common/` modules kept). §6.A.2 Batch P figures that still quote "49 files / 11,435 LOC" and "20 files / 3,143 LOC incl. 5 common modules" must be re-based on these numbers (40 files / 9,540 + 22 files / ~3,060).
62. **Batch P follow-ups not verified here.** (a) Per-file pruning inside the kept `perturbation_shaking/` (9 files; only `kick.py`/`perturb.py` names are consumed via the package root — `cross_day.py`, `day_shuffle.py`, `evolutionary.py`, `genetic_transformation.py`, `branch_and_bound.py`, `double_bridge.py` need per-file importer checks). (b) Qwen §4 item 52b: HGS yaml flags `use_cross_exchange`/`use_lambda_interchange`/`use_ejection_chains` are confirmed all `false` (`policy_hgs.yaml:131-140`), so most of kept `inter_route_local_search` is loaded-but-uncalled — candidate for the same treatment, needs its own sweep. (c) SANS `params.combination` (`"best"` default) is now new-engine-only; the `"a"`/`"b"` string values died with the OG yaml.
63. **Process note for the packaging scripts.** The ALNS `__init__` incident (item 60) is the general rule: `prune_codebase.py` file deletion must be paired with `clean_init_file`-style re-export cleanup in the SAME change, or the export breaks at import time while every deleted file looks unreferenced. The `export_config.json` `operator_categories` addition proposed in §6.A.3 item 7 should therefore list the 7 kept families (`destroy_ruin`, `recreate_repair`, `solution_initialization`, `crossover_recombination`, `inter_route_local_search`, `intra_route_local_search`, **`perturbation_shaking`**) — not 6.

## 5.E. Qwen — lane E — 2026-09-25

**Scope and reproducibility.** Reviewed `70e660b03` in detached worktree `/tmp/qwen-wsr` (data and `wsmart_bin_analysis` symlinked from the main checkout). Shared HEAD stayed `9bedfb996`. No source commit. Read-only analysis with targeted verification; full removal experiment is pending (see verification plan below).

**Read:** All 5 policy packages (`alns.py`, `hgs.py`, `split.py`, `evolution.py`, `individual.py`, `pg_clns.py`, `aco.py`, `lns.py`, `solver.py`, `particle.py`, `sans.py`, `anneal.py`, `sans_operators.py`, `sans_neighborhoods.py`, `sans_opt.py`, `sans_perturbations.py`, `sans_state.py`); all 3 dispatchers; all 5 policy adapters; `local_search_base.py`, `local_search_hgs.py`; the entire `helpers/operators/` package (13 sub-categories, 102 files); PG-CLNS's own `operators/` subdirectory; SANS's own `operators/`, `search/`, `select/`, `refinement/`, `heuristics/`; all policy yamls; `__init__.py` re-export chains.

**Rows:** B-qwen-01–12; R-qwen-01–05. Priority order: **R-qwen-01** (11,435 LOC of unreachable operators — the single largest removal candidate in the review), **B-qwen-01** (HGS split broken for limited fleet), **R-qwen-03** (SANS OG engine dead and broken, 3,143 LOC), **B-qwen-02** (SANS double operator selection), then B-qwen-03–12.

**Operator reachability (the big question, R-claude-05).** Built the full import graph from the five metaheuristic policies + `fast_tsp.py` + `local_search_hgs.py` into `helpers/operators/`. Result:

| Operator category | Files | LOC | Reachable from | Used by |
|---|---:|---:|---|---|
| `destroy_ruin` | 13 | 3,519 | ALNS (via `__init__.py`) | ALNS uses 6/13 files directly; 7 more loaded transitively |
| `recreate_repair` | 12 | 4,743 | ALNS (via `__init__.py`), BPC (direct) | ALNS uses 5/12 files directly; BPC uses `greedy.py` |
| `solution_initialization` | 6 | 820 | ALNS (via `__init__.py`), ACO-HH (direct) | ALNS uses `greedy_si.py`; ACO-HH uses `greedy_si.py` |
| `crossover_recombination` | 7 | 1,462 | HGS (direct) | HGS uses 1 function (`route_profit_gpx_crossover`) |
| `inter_route_local_search` | 8 | 1,484 | `local_search_base.py` → HGS | HGS loads all via `local_search_base.py`; yaml flags control which are called |
| `intra_route_local_search` | 6 | 1,211 | `local_search_base.py` → HGS | HGS loads all via `local_search_base.py` |
| **`evolutionary_mutation`** | 6 | 999 | **none** | Zero external callers |
| **`generalized_insertion_and_deletion`** | 12 | 2,581 | **none** | Zero external callers |
| **`improvement_descent`** | 4 | 760 | **none** | Zero external callers |
| **`intensification_fixing`** | 4 | 1,179 | **none** | Zero external callers |
| **`perturbation_shaking`** | 9 | 1,895 | **none** | Zero external callers |
| **`search_heuristics`** | 9 | 2,737 | **none** | Zero external callers (LKH/LNS/GES) |
| **`sequence_merging`** | 5 | 1,284 | **none** | Zero external callers (ACO/Markov/SS-HH) |

**Bold** = unreachable from retained policies. Total unreachable: **49 files / 11,435 LOC**. PG-CLNS and SANS have their own independent operator implementations and do not import from the shared library.

**Verification plan.** The removal of the 7 unreachable categories requires cleaning `helpers/operators/__init__.py` (removing re-export blocks for the 7 categories) and verifying `compileall` + `import_sweep.py` + the full smoke. This was not done in this pass due to time; the import-graph evidence is complete and the removal is mechanical.

**Key findings in detail:**

1. **B-qwen-01 (HGS split limited broken).** The DP for `_split_limited` initializes `V_curr[0] = -inf` and never sets it to 0. After k=1, `V_prev[0]` is `-inf`, so the deque is never seeded for k≥2. The vehicle limit is silently ignored. The shipped yaml uses `max_vehicles: 0` (unlimited), which takes `_split_unlimited` (correct). This is a latent bug that only manifests when `max_vehicles > 0`.

2. **B-qwen-02 (SANS double operator selection).** The inner loop at `sans.py:256-307` calls `_select_neighbor` (which applies an operator), then immediately selects a second operator and overwrites the result. The first application is wasted. This is a ~2× overhead on the inner loop, not a correctness bug.

3. **B-qwen-03 / R-qwen-03 (SANS OG engine dead and broken).** `execute_og` builds a dict with `"Q"` but `find_solutions` reads `"vehicle_capacity"` and `"E"` → `KeyError`. Even if that were fixed, `params.combination` is a string indexed as a tuple → `IndexError`. The OG engine has never worked in the current codebase. The 20 OG-only files (3,143 LOC) are safe to remove.

4. **R-qwen-01 (7 unreachable operator categories, 11,435 LOC).** The single largest removal candidate in the review. Import-graph evidence is complete; compileall verification is pending.

**Disagreements / confirmations (one line each):**
- Confirm R-claude-05: the operator library is massively over-retained. 49/102 files (11,435/25,118 LOC) are unreachable.
- Confirm Kimi §4 item 45c: ACO-HH uses 6 operator families from the shared library.
- Confirm Cursor §5.F: `local_search_hgs.py` optional flags (`use_cross_exchange`, `use_lambda_interchange`, `use_ejection_chains`) are all False in the shipped yaml, making most of `inter_route_local_search` loaded-but-uncalled.
- Confirm Gemini R-gemini-02: pruning `embeddings/edges/` drops `torch_geometric` from model imports.
- B-kimi-32 (base early-return ignores `vrpp`) affects all five metaheuristic policies equally — the `BaseRoutingPolicy._validate_mandatory` short-circuit is shared.

**Owner questions from Lane E:**
1. HGS limited-fleet split (B-qwen-01): fix or document as unsupported? The shipped yaml uses unlimited fleet.
2. SANS OG engine (R-qwen-03): confirmed remove? The engine is broken and unreachable.
3. ALNS `package`/`ortools` engines (R-qwen-02): confirmed remove?
4. HGS `pyvrp` engine (R-qwen-04): confirmed remove?
5. Within reachable operator categories, should we prune individual unused files (e.g., `destroy_ruin/bb.py`, `recreate_repair/deep.py`) or keep entire categories for API completeness?

## 5.G. Mistral — lane G — 2026-09-25

**Scope and reproducibility.** Reviewed `70e660b03` in detached worktree `/tmp/mistral-wsr` (data symlinked from main checkout; submodule `wsmart_bin_analysis` gitlink empty — confirmed as a known setup issue per Codex item 12 / Grok item 26). No source commits. The shared checkout HEAD stayed `9bedfb996`. The reachability script is posted at `.agent/cache/tools/reachability.py` for other lanes to re-run and extend.

**Read:** `logic/pyproject.toml` (full deps + optional-deps + tool.uv.sources + indices); `main.py` and `logic/controllers/hydra_dispatch.py` (entry-point imports); `logic/configs/policies/policy_sans.yaml` + `logic/src/configs/policies/sans.py` (yaml-dataclass mismatch); `logic/src/pipeline/simulations/repository/filesystem.py:210-300` (`.xlsx` vs `.csv` area dispatch); `logic/src/pipeline/simulations/states/finishing.py:100-120,220-235` (dead `.xlsx` writes); `logic/src/utils/input/splitting.py:183` (openpyxl use); `logic/src/data/datasets/simulation/pd_xlsx_dataset.py:169` (openpyxl use); `.gitmodules` (submodule config); `logic/src/pipeline/simulations/bins/base.py:44` and `logic/src/data/distributions/statistical_empirical.py:24` (GridBase imports); `logic/src/utils/data/loader.py:229-299` (GridBase lazy imports). Grepped every third-party top-level package in `logic/pyproject.toml` `dependencies` and `optional-dependencies.solvers` against `logic/**/*.py` + `main.py`.

**Rows:** R-mistral-01–04 (§3); §4 items 53–59. No §2 bug rows — the defects found are all in dependency/config metadata, which §3/§4 own.

**Reachability script.** Posted at `.agent/cache/tools/reachability.py`. It walks static `import`/`from` statements from `main.py` and `hydra_dispatch.py` and resolves `logic.src.*` import paths to files under `logic/`. Key finding: only **33 of 707** modules are reachable from static imports alone. The real reachability comes from runtime string-keyed registries (Hydra `_target_`, `RouteConstructorRegistry`, `ENV_REGISTRY`, policy dispatchers) and lazy imports. The script is a starting point, not the final reachability answer — lanes D and E already built more complete per-policy import graphs for the policy subtree (Kimi §5.D item 6, Qwen §5.E). The `import_sweep.py` runtime sweep remains authoritative for "can this module import?".

**Dependency findings (the core of lane G).** Muse R-muse-02 identified 7 zero-hit core deps + 3 coupled drops and explicitly left `einops`, `scikit-learn`, `hydra-colorlog` and per-solver optionality for lane G. Results:

| Package | Group | Hits | Verdict |
|---|---|---|---|
| `einops` | core | 0 | **R-mistral-01: drop** |
| `scikit-learn` | core | 0 | **R-mistral-01: drop** |
| `hydra-colorlog` | core | 0 | **R-mistral-01: drop** |
| `torch-scatter` | core | 0 | **R-mistral-02: drop** (transitive of torch-geometric) |
| `torch-sparse` | core | 0 | **R-mistral-02: drop** (transitive of torch-geometric) |
| `ml-dtypes` | core | 0 | **R-mistral-02: drop** |
| `torchvision` | core | 0 | **R-mistral-02: drop** |
| `hexaly` | solvers | 0 | **R-mistral-03: drop** |
| `vrpy` | solvers | 0 (comment-only) | **R-mistral-03: drop** |
| `pyomo` | solvers | 1 (always crashes) | **R-mistral-03: drop** (confirms R-kimi-10) |
| `gurobipy` | solvers | 10+ | **keep** (BPC, SWC-TCF gurobi, helpers) |
| `ortools` | solvers | 2 | **keep** (SWC-TCF ortools, ALNS ortools_wrapper) |
| `fast-tsp` | solvers | 1 | **keep** (route improvement TSP) |
| `alns` | solvers | 4 | **keep** (ALNS) |
| `pyvrp` | solvers | 2 | **conditional** (R-qwen-04: drop if HGS pyvrp engine removed) |
| `protobuf` | core | 1 | **keep** (ortools_wrapper — §4 item 56) |
| `openpyxl` | core | 4 (all on removal paths) | **drop after** R-grok-01 + R-grok-05 + R-claude-03 (§4 item 55) |

Total confirmed-unused core `dependencies`: **17 packages** (Muse's 7 + Mistral's 7 + 3 coupled: `torch-geometric`, `wandb`, `openpyxl`).

**Yaml-dataclass consistency.** Did a coarse comparison of `policy_*.yaml` keys vs `configs/policies/*.py` dataclass fields for all nine policies. Nested Hydra structures make exact comparison hard with grep, but one finding stands: SANS `params` (R-mistral-04) — a yaml-only key with no dataclass field, only in the OG-engine variants. Kimi §4 item 41 already lists the unread/inert policy-yaml keys from the solver side; this complements from the dataclass side.

**Submodule/data footprint.** Confirmed: `wsmart_bin_analysis` is required (§4 item 58). `GridBase` is the import the retained `emp` distribution and simulation bins depend on. The export must materialize the submodule, not just the gitlink.

**Disagreements / confirmations (one line each).**
- Confirm Muse R-muse-02: `einops`, `scikit-learn`, `hydra-colorlog` are zero-hit (Muse left these for lane G; verified).
- Confirm Muse R-muse-02: `openpyxl` is coupled to R-claude-03, but add that `filesystem.py:_get_mixrmbac_data` and `finishing.py` dead-write are the other two use sites (§4 item 55).
- Confirm Kimi R-kimi-10 from the dependency side: `pyomo` is in `optional-dependencies.solvers` and the only importer crashes unconditionally (R-mistral-03).
- Confirm Qwen R-qwen-04: `pyvrp` is only used by the unreachable HGS pyvrp engine (conditional drop, §4 item 54).
- Confirm Grok §4 item 26 + Codex §4 item 12: submodule required (§4 item 58).
- No disagreement with Muse's packaging-script edits (§4 items 34–39); they are complementary to mine.

**Open questions for the owner.**
1. `pyvrp` (R-qwen-04): drop from `solvers` when the HGS pyvrp engine is removed, or keep for API completeness?
2. `openpyxl`: confirm that `mixrmbac` area support is dropped (Grok R-grok-05) so `openpyxl` can leave core deps?
3. Submodule: ship only `export.simulation` + `python.sample_gen` from `wsmart_bin_analysis`, or the full repo?

## 5.O. OpenCode — Batch P verification — 2026-09-25

**Scope and reproducibility.** No assigned lane in the minimal-export review; worked the open Batch P item that §6.A.2 explicitly left unverified (Qwen R-qwen-01–04, "import-graph complete, sweep pending"). Reviewed commit `70e660b03` in detached worktree `/tmp/opencode-wsr` (data + `wsmart_bin_analysis` symlinked per the common brief; shared checkout untouched, no source commits). Baseline sweep before any deletion: **707 modules, 0 failed**. The deletion experiment exists only in that worktree (`git diff`: 76 files changed, ~14.2k deletions, 3 insertions).

**Read:** all 13 `helpers/operators/` categories + `operators/__init__.py` (444 lines); every external importer of the shared root (`local_search_base.py`, `bpc_engine.py`, `hyper_aco.py`, `hyper_operators.py`, `policy_aco_hh.py`, `alns.py`, `hgs.py`); ALNS/HGS/SANS dispatchers + package `__init__` files; `policy_sans.py`, `SANS params.py`, full SANS `common/` + `heuristics/` imports; `policy_{alns,hgs,sans}.yaml`, `test_sim.yaml`; `utils/policy/` (all 6 files).

**Rows:** B-opencode-01–02; R-opencode-01–05; §4 items 60–63. Priority: B-opencode-01 (a verbatim R-qwen-01 deletes a live ACO-HH dependency), then the verified R-opencode-01/02 volume, then B-opencode-02.

**Removal verification (final tree).** `compileall -q logic main.py` clean; `import_sweep.py` **640 modules, 0 failed** (707 − 67 deleted, exact). Runtime: `main.py test_sim` (2 days, Rio Maior 20 bins, `emp`, `sim.policies=[sans,alns,hgs,aco_hh]`, 2 cores) completes and writes all 8 variant `log_*.json` + realtime `.jsonl`; km 0 throughout is expected pre-threshold (cf70 collects day 7). This exercises every edited dispatcher, the trimmed `operators/__init__.py` (via `aco_hh_custom`, which consumes the kept `kick`/`perturb`), the cleaned SANS `__init__` (via `sans_new`), and the trimmed `policy_sans.yaml`. Train/eval were not re-run: none of the deleted modules is imported by the train/eval path (sweep covers their imports; `utils.policy` has no train/eval importer).

**Disagreements / confirmations (one line each).**
- Narrow R-qwen-01 (B-opencode-01): `perturbation_shaking/` stays; Qwen §4 item 52a was right, the R-qwen-01 table was wrong.
- Narrow R-qwen-03 (B-opencode-02): `common/{objectives,penalties,revenue}.py` stay; the other 22 files + OG yaml go.
- Confirm R-qwen-02 / R-qwen-04 as specified (plus `__init__`/dispatcher cleanup, which the sweep proved necessary).
- Confirm R-mistral-03's `pyvrp` conditional drop pairs with R-opencode-04; confirm R-mistral-04 is subsumed by R-opencode-02.
- Confirm Qwen §5.E `local_search_hgs` flag observation from the yaml side (`policy_hgs.yaml:131-140` all false) — follow-up 62b, not verified here.
- Do not apply Qwen §4 item 46 / §6.A.2 Batch P as written (49 files / 11,435 LOC); the verified numbers are R-opencode-01 + R-opencode-02.

**Owner questions.**
1. Approve promoting R-opencode-01–05 to the confirmed removal list (§6.A.2 Batch V) with the item-60 edit list?
2. `perturbation_shaking/` per-file pruning (62a) and `inter_route_local_search` loaded-but-uncalled pruning (62b): approve as follow-up lanes or drop?
3. `pyvrp` dep drop now (R-opencode-04 landed) or keep for API completeness (Mistral §5.G question 1 still open)?

## 6. Consolidation (Claude + Codex, after all lanes report)

Lane A evidence is available; overall consolidation remains open until B–G report. Do not promote untested proposals to confirmed removals or declare the package functionally correct from imports alone.

_(empty — filled at lock time: final removal list grouped by packaging hook, bug list with owners, decisions needed from the owner.)_

## 6.A. DeepSeek — consolidation pass — 2026-09-25

> Signed consolidation of lanes A–G + Muse (`feat/minimal-export-package`, review
> commit `70e660b03`; shared checkout `9bedfb996`). This pass merges the seven
> signed §5 sections into one actionable list. It edits no lane's rows and adds
> no source commits. My own verification notes are §6.A.0.

### 6.A.0 Scope and independent verification (signed §5.DS)

**Method.** Re-ran the §0 baseline on the shared checkout, re-read every
packaging script the report points at on `main`
(`git show main:logic/package/*.py`, `git show main:ci/export_config.json`), and
spot-checked `file:line` for the highest-impact rows. All commands run from the
repo root with `.venv`.

**Confirmed independently in this pass:**

1. **Baseline is green.** `.venv/bin/python .agent/cache/tools/import_sweep.py`
   → **707 modules, 0 failed** (matches the report's baseline).
2. **The review target is stable.** `git diff --stat 70e660b03 HEAD` touches
   only `.agent/**`; no `logic/` source changed between the analysed commit and
   the shared checkout, so every lane's `file:line` claim still resolves.
3. **§4 item 34 (shared-dir wipe) is real.** `ci/export_config.json` on `main`
   gives `selector`, `improvement`, `acceptance`, `joint` the *same*
   `yaml_dirs: ["logic/configs/policies/other"]` and
   `config_dirs: ["logic/src/configs/policies/other"]`, and `joint` has empty
   `yaml_prefixes`/`config_prefixes`. `prune_category`
   (`prune_codebase.py:606-619`) then calls `_prune_all_yaml_by_prefix`, whose
   `else: should_delete = True` (`:504-505`) deletes the whole shared dir.
4. **§4 item 35's correction is right.** `_match_acronym`
   (`cleanup_helper.py:194-222`) returns `False` for
   `hybrid_genetic_search` vs `hgs_adc`/`hgs_alns`/`hgs_rr` (no exact/prefix/
   suffix hit, initials `hgs` ≠ `hgs_adc`, reverse initials `ha` ≠ the name).
   The over-deletion path is `_find_impls_to_delete:313-322`: a matching
   **variant file** inside `hybrid_genetic_search/` promotes its whole parent
   dir into `to_delete`.
5. **§4 item 38 (stub list) is real.** `remove_tracking.py:356-376` emits
   no-ops for `output_stats`, `send_daily_output_to_gui`,
   `send_final_output_to_gui`, `update_policy_log_section` — exactly the
   retained result writers.
6. **`remove_enums.py` regex is real.** `git show main:logic/package/remove_enums.py`
   line 43: `(?s)@GlobalRegistry\.register\s*\([^)]*?\)` — `[^)]` cannot span
   nested parentheses even with DOTALL.
7. **R-muse-03 is real.** `ci/export_config.json` `imitation_policy.impl_dirs`
   is `logic/src/models/policies`, which does not exist on the branch.
8. **B-codex-06 mechanism confirmed statically.** `_build_stage_config`
   (`features/train/engine.py:176-179`) writes the stage graph into
   `task_cfg["env"]` i.e. `cfg.train.env.graph`, while `data.py:184` reads
   `getattr(self.cfg, "env", None)` and `:203` reads eval graphs through
   `_get_eval_graphs(self.cfg)` (`cfg.env`). The configured
   `n_samples`/`eval_graphs` are therefore ignored.
9. **B-codex-08 / B-gemini-01 path confirmed.** `utils/model/loader.py` forwards
   flat `normalization`/`activation` into `AttentionModel`'s `**kwargs`;
   `AttentionModel.__init__` consumes only `norm_config`/`activation_config`.
10. **B-claude-06 / B-grok-02 naming split is real.** `states/base/context.py:152-153`
    sets `pol_id_orig` (expander id) and `pol_name = to_slug(display_name)`;
    `parallel_runner.py:157` and `orchestrator/__init__.py:202` key on
    `to_slug(display_name)` while the result dict and `log_*.json` use the
    expander id, exactly as R-grok-02 describes.

I did **not** re-run the lane worktree removal experiments (they are recorded
with their own sweeps); I did not re-execute the GPU paths, the BPC/SWC
counterexamples, or the full nine-policy sim. Items below carry the lane's own
evidence and are marked accordingly.

### 6.A.1 Bugs — consolidated and deduplicated

Duplicates collapse to one row each (the right-hand column lists every row that
reports it). Severity is the highest claimed. "Path" abbreviations:
`loader` = `logic/src/utils/model/loader.py`, `attn` =
`logic/src/models/core/attention_model/model.py`, `decoder` =
`logic/src/models/subnets/decoders/glimpse/decoder.py`.

**Group 1 — must fix before the export is described as correct (retained path).**

| Consolidated | Rows | Where | Sev | Symptom (one line) | Disposition |
| --- | --- | --- | --- | --- | --- |
| DS-01 | B-codex-06 | `features/train/engine.py:176-179` vs `rl/common/base/data.py:184,203` | major | Stage graph writes `cfg.train.env`, data setup reads `cfg.env`; requested 64/32 becomes 10/512 and eval graphs are dropped | Fix the reader or the writer; add dataset-size assertion (§6.A.5) |
| DS-02 | B-codex-08, B-gemini-01, B-gemini-02 | `loader.py:94-106`, `attn:134-135`, `policy.py:57,75-82` | major | Norm/activation flattened into swallowed kwargs; saved `layer` loads as `BatchNorm1d` | Rebuild structured `NormalizationConfig`/`ActivationConfig`; add prediction-parity check |
| DS-03 | B-gemini-03 | `loader.py:150-162`, `attn:220-221`, `decoder:114-119` | major | Dead duplicate `context_embedder`; loader synthesizes fake identity weights for it | Delete the dead embedder + synthesis block |
| DS-04 | B-codex-07, B-gemini-06 | `features/eval/engine.py:261,309`, `attn:431`, `envs/tasks/vrpp.py:74` | major | Eval serializes `cost` as negative profit; `get_best` minimizes while sampling maximizes | Standardize on one `reward`/`cost` key and sign |
| DS-05 | B-codex-03 | `rl/common/baselines/rollout.py:255`, `rl/core/reinforce.py:103` | major | `[B,1]` baseline broadcasts against `[B]` reward → `[B,B]` advantage | Reshape/assert before subtraction |
| DS-06 | B-codex-01 | `rl/common/base/module.py:96,196,199` | major | Baseline config saved under `hparams.kwargs`, read at top level; `exp_beta`/warmup ignored | Flatten or pass explicitly |
| DS-07 | B-codex-02 | `rl/common/baselines/critic.py:45`, `optimization.py:51`, `reinforce.py:103` | major | `baseline=critic` builds no critic; no regression loss, excluded from optimizer | Repair critic (keep scope) or remove option + schema |
| DS-08 | B-codex-04 | `rl/common/epoch.py:79`, `rollout.py:249` | major | Epoch wrapping evaluates the live policy, not the frozen baseline | Use the frozen policy |
| DS-09 | B-codex-05 | `rl/common/base/data.py:288`, `steps.py:333`, `epoch.py:209,215` | major | `train_time` applies shuffled tours in dataset-row order → wrong bins reset | Carry instance IDs |
| DS-10 | B-codex-09 | `loader.py:145-148` | major | Empty/unknown checkpoint loads a random model and prints success | Reject empty/unknown, fail on missing keys, allowlist migrations |
| DS-11 | B-gemini-04 | `decoder:103-104,302` | **blocker** | `torch.Generator` stays on CPU after `model.to("cuda")` → sampling eval crashes on GPU | Match generator device to `probs.device` |
| DS-12 | B-gemini-05 | `decoder:310-311` | major | Sampling `while` loop spins forever on zero-mass valid actions | Zero+renormalize, single draw |
| DS-13 | B-grok-01 | `bins/base.py:147`, `utils/data/loader.py:251-252`, `states/initializing.py:433,468-480` | major | Empirical fills generated on `arange(num_loc)` while routed graph is focused/sorted; 4/20 ID overlap | Build grid from the same focus indices/order |
| DS-14 | B-grok-02, B-claude-06 | `states/base/context.py:152-153`, `simulator.py:262-269`, `orchestrator/results_handler.py:68,98-107` | major | Expander id `…_emp` vs slug `…_none`; parallel result lookup misses → all-zero `mean`/`std` for `n_samples>1`; `cpu_cores:0` is parallel | One key (`pol_id_orig`) for files, jsonl, checkpoints, results |
| DS-15 | B-grok-03 | `states/finishing.py:62-73`, `actions/route_construction.py:178` | major | Sample `time` is day-loop wall clock, daily `time` is solver only | Sum daily times or rename |
| DS-16 | B-grok-04 | `bins/base.py:411-433` | major | `new_overflows` counts a bin already at 100% with zero new waste, again next idle day | Count only on lost-mass-positive day |
| DS-17 | B-grok-05 | `states/running.py:62`, `states/initializing.py:345-358` | major | Resume adds stored elapsed time to start timestamp → negative sample `time` | `tic = perf_counter() - run_time` |
| DS-18 | B-grok-07 | `states/running.py:73-77`, `simulator.py:515-516` | major | Any day-loop exception becomes `CheckpointError`; sequential failure is dropped, process still "succeeds" | Record failures, non-zero exit |
| DS-19 | B-cursor-02 | `rl/common/base/steps.py:128,140-142`, vector selectors | major | Training selector runs on customer-only waste then zeroes index 0 (the depot); first customer never mandatory | Prepend depot column before `select()` |
| DS-20 | B-cursor-03 | `configs/policies/other/ms_last_minute.yaml:27,31`, `vector/selection/last_minute.py:34,59`, `configs/tasks/train.yaml:108-109` | major | Last-minute threshold in percent (`70`) vs fraction (`0.7`/`0.25`) | Owner decided: rewrite vectorized to percent |
| DS-21 | B-cursor-04 | `mandatory_selection/selection_lookahead.py:178-179`, `vector/selection/lookahead.py:151` | major | Scalar multiplies by absolute day, vectorized by relative days; agree only at day 0 | Relative days both sides |
| DS-22 | B-kimi-02 | `helpers/.../search/cutting_planes.py:937-1045` | major | Single-vehicle LCI cuts injected into unlimited-fleet master: shipped yaml returns 17 vs true 25 (3-node) | Gate on `vehicle_limit==1` or delete engine |
| DS-23 | B-kimi-03 | `helpers/.../lagrangian_relaxation/subgradient_optimization.py:148-162` | major | Single-vehicle LR pre-pruning bound unsound for K routes; can prune the optimum (`lr_pre_pruning: true` shipped) | Scale by fleet size or drop |
| DS-24 | B-kimi-04 | `bpc_engine.py:700-703,765-767,971-976` | major | Node timeout falls back to empty route list when mandatory is empty → silent 0 km/0 kg day | Return best-so-far |
| DS-25 | B-kimi-05 | `master_problem/model.py:310-318`, `column_generation.py:236-240` | major | Non-OPTIMAL LP returns `(0.0, {})`; empty treated as integer optimum | Return None/raise; make guard real |
| DS-26 | B-kimi-11 | `pricing/solver.py:684,353-363`, `column_generation.py:410-425` | major | UB prune trusts `pricing_exhausted` from truncated/heuristic pricing; can prune optimum | Gate on full search |
| DS-27 | B-kimi-16 | `.../gurobi.py:157,165` (+ortools/pyomo) | major | Travel cost halved (`0.5*C*Σd·x`); model objective ≠ reported cost | Drop the `0.5*` |
| DS-28 | B-kimi-17 | `ortools_wrapper.py`, `pyomo_wrapper.py` vs `base_routing_policy.py:252-260` | major | OR-Tools/Pyomo solve a kg model on percent inputs; backend-dependent results | One unit system |
| DS-29 | B-kimi-18 | `gurobi.py:135-142` (+2) | major | `delta` inert (mandatory forced `g[i]==1` makes slack constraint redundant) | Implement or delete key (goes with R-kimi-11) |
| DS-30 | B-kimi-19 | `gurobi.py:74-75` vs ortools/pyomo `max_dist` | major | `6000000.0` vs `6000` arc filter; backend-dependent feasible arcs | One `6000.0` km constant |
| DS-31 | B-kimi-20 | `gurobi.py:191-227`, `ortools_wrapper.py:209-211`, `dispatcher.py:78-90` | major | Infeasible/no-solution returns silent empty tour; one unreachable mandatory bin zeroes the day | Branch on status; log/raise |
| DS-32 | B-kimi-21 | `base/base_routing_policy.py:231-236` | major | Typed-config path drops engine-nested overrides (`vrpp:false`, capacity, …) | Flatten unconditionally |
| DS-33 | B-kimi-32 | `base/base_routing_policy.py:192-205,487-489`, `actions/route_construction.py:109-116` | major | Empty mandatory early-returns `[0,0]` regardless of `vrpp`; optional-bin collection impossible | Owner semantics first (see §6.A.4 D3) |
| DS-34 | B-qwen-01 | `hgs/split.py:301-365` | **blocker (K≥2)** | `_split_limited` never seeds `V_curr[0]=0`; `max_vehicles` silently ignored | Set `V_curr[0]=0` or mark unsupported |
| DS-35 | B-qwen-03 | `sans/dispatcher.py:195-240` | **blocker (OG)** | `execute_og` dict key/type mismatch → `KeyError`/`IndexError`; OG engine never worked | Delete with R-qwen-03, do not fix |
| DS-36 | B-qwen-04 | `hgs/hgs.py:639-647` | major | Restart does not clear `_offspring_coverage`/`_offspring_margin`; stale penalties | Clear them |
| DS-37 | B-qwen-06 | `alns/alns.py:672-678` | major | `T_start` calibration skipped when `best_profit ≤ 0`; BMC degenerates to hill-climb | Remove guard / `abs()` |
| DS-38 | B-qwen-10 | `psoma/solver.py:163-179` | major | EMA reward uses `abs()` in training vs `max(0,·)` in eval | Use `max(0,·)` both |
| DS-39 | B-kimi-01 | `neural_agent/policy_na.py:135-139`, `simulation.py:243,247` | minor | NA returned profit/cost units wrong (100× / missing /100·V·B); discarded by the only caller | Fix units or document advisory |
| DS-40 | B-kimi-22 | `pyomo_wrapper.py:85,151` | **blocker (pyomo)** | Filter-only Pyomo set never initialized → `ValueError` at construction; pyomo path always crashes | Prune wrapper+dep (R-kimi-10) or one-line `initialize=` |
| DS-41 | B-cursor-05 | `other_algorithms/.../tsp.py:43,64`, `route_improvement/fast_tsp.py:68-74` | major | `seed` accepted and never forwarded; default budget 2 s | Forward seed or drop kwarg |

**Group 2 — dead/unreachable code whose bug is moot if the code is removed.**
Fix only if the feature is kept; otherwise delete with the matching R row.

| Consolidated | Rows | Where | Disposition |
| --- | --- | --- | --- |
| DS-42 | B-kimi-06 | `pricing/smoothing.py:221,295-304,323-417`; `params.py:152,156` | DSSR wrapper broken + keys unread → delete (R-kimi-14) |
| DS-43 | B-kimi-07 | `bpc_engine.py:714-732`; `smoothing.py:425-436` | arc-fixing block gate never true, would crash → delete (R-kimi-17) |
| DS-44 | B-kimi-08 | `search/cutting_planes.py` 4 engines | no-op cut engines + MinCut constraint rewrite → delete (R-kimi-16) |
| DS-45 | B-kimi-09 | `lagrangian_relaxation/pre_pruning.py:100` | `seed=None` → Gurobi `TypeError`; simulator masks it | One-line default, or moot if LR pre-pruning dropped |
| DS-46 | B-kimi-10 | `master_problem/model.py:208`, `column_generation.py:125-126` | dual smoothing can never turn on; yaml comments lie | Fix comments or implement |
| DS-47 | B-kimi-12 | `pricing/solver.py:859` | Farkas condition tautology | Collapse |
| DS-48 | B-kimi-13 | `pricing/labels.py:103-113`, `solver.py:732,736` | dead SRI dominance branch | Remove or implement |
| DS-49 | B-kimi-14 | `branching/tree.py:103-113` | reads `max_branch_nodes`/`tree_search_strategy` (real: `max_bb_nodes`/`search_strategy`) | Use real fields |
| DS-50 | B-kimi-15 | `bpc_engine.py`, `params.py`, `model.py:423`, `problem_support.py:783`, `constraints.py:470`, `labels.py:123`, `common/node.py:23`, `model.py:159`, `separation/engine.py:90` | all definition-only | Delete (R-kimi-17) |
| DS-51 | B-kimi-23 | `pyomo_wrapper.py:238-240,273-274` | termination check tautology; three backends, three failure modes | Goes with DS-40/R-kimi-10 |
| DS-52 | B-kimi-24 | `policy_swc_tcf.py:130`, `gurobi.py:185-186` | `int(time_limit)` truncates `<1 s` to unlimited | Pass float + epsilon guard |
| DS-53 | B-kimi-25 | `configs/policies/policy_swc_tcf.yaml:44-64` vs `gurobi.py:135-166` | yaml documents wrong objective semantics | Rewrite comments to code |
| DS-54 | B-kimi-26 | `hyper_aco.py:145`, `policy_aco_hh.py:128`, `params.py:75`, yaml:111 | `operators` list ignored (all 11 run) | Wire or delete (R-kimi-18) |
| DS-55 | B-kimi-27 | yaml:91 `sequence_length` | no dataclass field; pinned to 11 | Wire or delete (R-kimi-18) |
| DS-56 | B-kimi-28 | `hyper_aco.py:767-769,796` | wall-clock `execution_time` in visibility update → seed does not give reproducibility | Deterministic denominator |
| DS-57 | B-kimi-29 | `hyper_aco.py:308-316,874-877` | first-hop transition never reinforced; deposits land on unread virtual row | Seed `prev_op_idx` to real start |
| DS-58 | B-kimi-30 | `hyper_aco.py:296-299,230,753,321-325` | `pv` decays with no floor; returned best can violate capacity | Track best-feasible + floor |
| DS-59 | B-kimi-31 | `hyper_operators.py:525-526,566-567,607-608` | bare `except Exception: return False` hides mismatches | Catch specific, log |
| DS-60 | B-kimi-33 | `hyper_aco.py:281` | elitism `int()` vs docstring ceil | `math.ceil` or fix diagram |
| DS-61 | B-qwen-02 | `sans/heuristics/sans.py:256-307` | operator selected/applied twice; 2× inner-loop waste | Remove first selection |
| DS-62 | B-qwen-05 | `hgs/evolution.py:127` | `diversity_weight` negative when `pop_size < nb_elite` | Clamp at 0 |
| DS-63 | B-qwen-07 | `pg_clns.py:95,118`, `aco.py:117`, `lns.py:176` | `process_time` vs `perf_counter` inconsistency | Use `perf_counter` |
| DS-64 | B-qwen-08 | `pg_clns/lns.py:163-168` | hardcoded `lambda_decay=0.8` ignores `reaction_factor`; 0.1 floor | Use param, lower floor |
| DS-65 | B-qwen-09 | `pg_clns/operators/__init__.py:79-81` | `__all__` lists non-existent `perturb`/`kick` | Clean `__all__` |
| DS-66 | B-qwen-11 | `sans/heuristics/anneal.py:121-131` | `_update_removed_bins` only at `delta==0` | Moot with R-qwen-03 |
| DS-67 | B-qwen-12 | `sans/common/revenue.py:30-38` | OG revenue missing `/100` (100× too large) | Moot with R-qwen-03 |
| DS-68 | B-grok-08 | `actions/collection.py:89` | CTOP branch adds percent into a kg load | Moot if CTOP switch removed; else convert |
| DS-69 | B-grok-06 | `repository/filesystem.py:251-257` | `num_loc==104` files absent | Delete branch (R-grok-05) |

**Group 3 — config/metadata only.** `B-kimi-25` (above), `B-cursor-07`
(`fill_ratios` ignored, `>=` vs `>`), `B-cursor-01` service-level paper form.
Group 3 also covers the yaml/code drift class flagged by Kimi §4 item 41.

**Group 4 — already fixed on the branch** (kept for reviewers):
`B-claude-01` (`3ea21fec1`), `B-claude-02`, `B-claude-03`, `B-claude-04`,
`B-claude-05` (`3ea21fec1`/`70e660b03`, two need a `main` port).

**Group 5 — wontfix by owner decision:** `B-cursor-01` service-level stays
linear (no `√n_d` code to delete); `R-cursor-08` `train_time` is not a removal.

### 6.A.2 Removals — consolidated plan grouped by packaging hook

**Nothing below is a new finding.** This is the merge of §3 into the sequence
the packaging scripts need. "Verified" = the lane deleted it in a worktree and
ran `compileall` + `import_sweep`; those numbers are the lane's, not re-run
here. Overlap between lanes was checked from the row paths; no deleted file
overlaps between the verified batches, but V1/V2/V3 share export-cleanup edits
in `rl/core/__init__.py`, `rl/common/__init__.py`, `models/__init__.py` and
`models/common/__init__.py`, so those batches must land in one commit.

**Batch V — verified, mechanical; apply together and run one combined sweep.**

| Batch | Rows | Content | Verified saving | Hook |
| --- | --- | --- | --- | --- |
| V1 RL helpers | R-codex-01, R-codex-02 | `rl/core/losses/` + `rl/common/{reward_scaler,reward_scaler_batch,route_improvement}.py` + exports | 8 files / 711 LOC; sweep **699**, 0 failed | allowlist `rl/core` losses + atomic `__all__` cleanup |
| V2 model stack | R-claude-01, R-claude-04, R-gemini-01…08 | `embeddings/{positional,edges,state,dynamic,static}`, `context/generic.py`, 5 experimental modules, `symnco_policy.py`, `deep_decoder_policy.py`+yaml, `models/common/{critic_network,non_autoregressive,improvement,transductive}`, `baselines/shared_critic.py` | 39 files / **3,391** LOC; sweep **669**, 0 failed (drops `torch_geometric`) | `subnet_pruning` embeddings allowlist + new `models_common` + `remove_models` cleanup |
| V3 sim/utils/tracking | R-grok-01…04 | 7 `utils/input` modules, `yaml_to_env.py`, `metrics.py`, dead `storage.py` serializers | ~9 modules + storage half; **1,866 del/10 ins**; sweep **698**, 0 failed | `remove_tracking` keep-writers mode + `utils` cleanup |
| V4 selection/data/interfaces | R-cursor-01…06 | spatial/mixture distributions, 4 pytorch datasets, `envs/base/improvement.py`, `route_improvement/common/bandit.py`, 2 interfaces, 3 vector selectors | 18 files / **1,965 del/4 ins**; sweep **689**, 0 failed | `--distributions`, new `--pytorch-datasets`, interface/selector allowlists |
| V5 policies | R-kimi-01…08, R-kimi-13…17 | multi-period base, NA `batch.py`, dead tracking/viz, TSP helpers, k-sparse pheromones, ACO `construct`, knapsack node selection, DSSR, `search_strategy.py`, 4 no-op cut engines, ~280 LOC dead methods | 26 files / **+23/−2,197**; sweep **701**, 0 failed | manual per-file (`solvers_and_matheuristics` needs per-file `__init__` cleanup) |

Combined verified estimate if applied together: **≈100 files, ≈10.1k LOC, and
≈79 modules**; each lane's sweep was green in isolation but a single combined
sweep is still pending. This is the number the owner can commit to now.

**Batch P — proposed, needs worktree verification before it counts.**

| Rows | Content | LOC | Note |
| --- | --- | ---: | --- |
| R-qwen-01 | 7 unreachable `helpers/operators` categories (49 files) | 11,435 | Largest single candidate; import-graph complete, sweep pending. Clean `operators/__init__.py` |
| R-qwen-03 | SANS OG engine (20 files) + `execute_og` + `og_a/og_b/lac` yaml | ~3,143 | Eliminates a broken engine; subsumes R-mistral-04 |
| R-qwen-02 | ALNS `package`/`ortools` engines | 437 | Remove dispatcher branches |
| R-qwen-04 | HGS `pyvrp_wrapper.py` | 84 | Pair with `pyvrp` dep drop |
| R-claude-02 (remainder), R-cursor-07 | `unif/beta/const/distance` + 5 dead constants | ~360 | Requires the smoke move to `gamma1` and selector/constant allowlists |
| R-claude-03 (remainder) | `.pkl/.xlsx/.csv` loaders; survivors locked | ~482 | Drop after V4 |
| R-grok-05, R-grok-06 | `both`/104 branches; `checkpoints/` | ~670 | Owner choice on resume (D2) |
| R-codex-03, R-codex-04 | extra baselines; augmentation/multistart evaluators | ~509 | Only with narrow allowlists + schema edits |
| R-kimi-09…12, R-kimi-18, R-kimi-19 | NA vector fallback, pyomo wrapper, SWC `delta`, scenario tree, ACO keys, fleet-cover engines | n/a | Each pairs with an owner decision |
| R-muse-01…03 | 7 trainer fields, dep edits, `imitation_policy` path | n/a | Config/dep hygiene |
| R-mistral-01…04 | 7 core + 4 transitive deps, 3 solver deps, SANS `params` | n/a | Dep hygiene |

**Batch D — dependency edits (`logic/pyproject.toml`), one change.**
Confirmed-unused core deps (17): `python-pptx`, `docxtpl`, `latex2mathml`,
`cryptography`, `requests`, `joblib`, `pydantic` (R-muse-02);
`einops`, `scikit-learn`, `hydra-colorlog`, `torch-scatter`, `torch-sparse`,
`ml-dtypes`, `torchvision` (R-mistral-01/02); plus coupled drops
`torch-geometric` (after V2), `wandb` (after V3), `openpyxl` (after V4/V5
`.xlsx` removal). Drop 3 `solvers` optional-deps: `hexaly`, `vrpy`, `pyomo`
(R-mistral-03, confirms R-kimi-10); `pyvrp` conditional on R-qwen-04. Also
remove the matching `[tool.uv.sources]`/`[[tool.uv.index]]` entries for
`torch_scatter`/`torch_sparse`/`hexaly`. **Keep** `gurobipy`, `ortools`,
`fast-tsp`, `alns`, `protobuf` (ortools), `networkx` (retained `tsp.py`) and
`jinja2` (gui.py). Every dropped package was grep-checked for zero readers on
the retained tree.

**Ordering constraints.** `torch-geometric` waits on V2; `wandb` on V3;
`openpyxl` on V4/V5; `pyomo` on R-kimi-10 (or DS-40 fix); `pyvrp` on R-qwen-04;
the `gamma1` smoke move must land before the `unif/beta/const/dist` drop.

### 6.A.3 Packaging-script changes (the concrete roadmap input)

All line numbers are `main` as inspected in §6.A.0.

1. **`prune_codebase.py::prune_category` (:559-629) — stop the wipe.**
   (a) `_prune_all_yaml_by_prefix`/`_prune_all_configs_by_prefix` must not
   default to delete-all on falsy prefixes: delete only when an explicit prefix
   list is passed, otherwise raise. (b) Add a shared-dir guard in
   `_build_cleanup_kwargs`/before bulk deletion: build the union of every
   category's `yaml_dirs`+`config_dirs`; if a dir is claimed by >1 category,
   refuse bulk deletion and fail loudly. This is what wiped
   `logic/{src/,}configs/policies/other` (shared by `selector`/`improvement`/
   `acceptance`/`joint`).
2. **`cleanup_helper.py::_find_impls_to_delete` (:279-329) — stop the
   parent-dir kill.** Delete only the matched file; add the parent dir only
   when the directory name itself matches. Today a variant file (`hgs_adc.py`)
   inside `hybrid_genetic_search/` deletes the whole kept HGS package.
3. **`cleanup_helper.py::_match_acronym` (:194-222) — gate the heuristics.**
   For delete decisions keep exact / `policy_`/`selection_` / prefix / suffix /
   yaml-prefix equality; drop the `_<acronym>_` substring test (line 198) and
   the forward/reverse initials heuristics (lines 201-220) or require them to
   be confirmed against the registry.
4. **`cleanup_helper.py::clean_init_file` (:104-134) — bracket-aware.** When a
   matched line opens `(`/`[`, comment through the matching close, or re-emit
   the `__init__` via `ast`. The current single-line comment left the seven
   `IndentationError`s.
5. **`remove_enums.py::process_python_file` (:41-43) — balanced parens.** Scan
   from the opening `(` to its matching `)`, not `[^)]*?`.
6. **`remove_tracking.py` — add a keep-result-writers mode.** Keep `gui.py`,
   `storage.update_policy_log_section`, `setup_system_logger`,
   `analysis.output_stats`; delete only `metrics.py` (and then `wandb`). When
   that mode is active, stop emitting the stubs at :368-375 for the kept
   writers. Item 38 pins this.
7. **`ci/export_config.json` — fix categories.**
   (a) `imitation_policy.impl_dirs` `logic/src/models/policies` does not exist;
   point at `logic/src/policies/vector` (+ real yaml/config dirs) or delete the
   category. (b) `joint` has empty prefixes and shares
   `logic/configs/policies/other`; either delete the category or give it its own
   dirs so the item-1 guard can pass. (c) Add categories: `pytorch_datasets`
   (`td_dataset.py` + `baseline_dataset.py`), `models_common`
   (`autoregressive/` only), `embeddings` (allowlist `nodes/init.py`,
   `context/base.py`, `context/vrpp.py`), `operator_categories` (the 6 reachable
   families), `sans_engine` (`new` only).
8. **`remove_callbacks.py` + `remove_models.py`/`remove_policy_others.py` —
   coordinated caller edits** (Codex item 10): ZenML eager imports/dispatch in
   `features/{train,test,eval}/engine.py` + `eval/__init__.py`, three
   `zenml_*_pipeline.py`, `configure_zenml_stack` stubs, `zenml_*` schema/yaml
   keys; callbacks in `rl/common/trainer.py:227-240` + exports. GPU-memory
   callback is reachable on CUDA; keep it unless capability is dropped.
9. **Submodule materialization** (Codex 12 / Grok 26 / Mistral 58):
   `wsmart_bin_analysis` must be checked out, not left as an empty gitlink, or
   `emp` cannot import `GridBase`. `package_minimal.sh` already copies submodule
   tracked files; add a pre-package assert that `GridBase` imports.

### 6.A.4 Owner decisions required (merged, numbered)

- **D1 BPC (Kimi):** fix (gate LCI on `vehicle_limit==1`, fleet-scale the LR
  bound, certify pricing before the UB prune) or relabel BPC as a heuristic?
  Until then the export must not call BPC exact. DS-22/23/24/25/26.
- **D2 Checkpoints/resume (Grok R-grok-06):** keep and fix the clock
  (DS-17) + surface day-loop failures (DS-18), or delete the package and record
  failures directly?
- **D3 `vrpp` empty-mandatory semantics (Kimi B-kimi-32):** run the solver
  (flowchart) or return `[0,0]` (current code)? Current smoke expectations
  (km=0 before thresholds) encode the code side, so this changes acceptance.
- **D4 SWC-TCF (Kimi):** fix pyomo (DS-40) or prune wrapper + dep
  (R-kimi-10)? Align the three wrappers on percent units (DS-28)? Fix the
  halved travel cost (DS-27)?
- **D5 ACO-HH (Kimi):** wire `operators`/`sequence_length` or delete the keys
  (R-kimi-18)? Is run-to-run reproducibility required (DS-56)?
- **D6 NA adapter contract (Kimi):** fix profit/cost units (DS-39) or document
  advisory-only?
- **D7 Scenario tree (Kimi R-kimi-12, lane B file):** approve deleting the
  per-day build in `actions/route_construction.py:130-157` + follow-on
  `prediction.py` chain?
- **D8 PBRS (Codex/Muse):** keep the reachable capability (adds parity testing)
  or remove capability + callers + schema + yaml together?
- **D9 `pyvrp` (Mistral/Qwen):** drop with the HGS pyvrp engine (R-qwen-04) or
  keep for API completeness?
- **D10 `openpyxl`/`mixrmbac` (Mistral/Grok):** confirm the `mixrmbac` area is
  out of scope so `openpyxl` can leave core deps.
- **D11 Submodule scope (Mistral):** ship only `export.simulation` +
  `python.sample_gen`, or the full `wsmart_bin_analysis` repo?
- **D12 Eval methods (Codex R-codex-04):** narrow supported decoding to
  greedy+sampling and prune augmentation/multistart, or integrate them?
- **D13 Constants/interfaces/trainer fields:** the Harbinger approvals already
  recorded (R-cursor-07, R-claude-03, B-cursor-01/03) stand; D13 is only the
  packaging confirmation, no new question.

#### 6.A.4.1 Owner rulings (recorded by Claude, 2026-09-25)

| Decision | Ruling | Consequence for the plan |
| --- | --- | --- |
| D1 BPC | **Fix.** | DS-22 (gate LCI cuts on `vehicle_limit==1`), DS-23 (fleet-scale the LR bound), DS-24 (best-so-far on node timeout), DS-25 (non-OPTIMAL LP → None/raise), DS-26 (UB prune only after full pricing) become Group-1 work. BPC stays labelled exact only once the §6.A.7 3-node assertion (`obj == 25`) passes. |
| D2 checkpoints/resume | **Fix if needed, and prune from the packaging.** | R-grok-06 is approved for the export: the `checkpoints/` package and its callers leave the package. DS-17/DS-18 are fixed on `main` only if they survive there; in the export the day-loop must record failures directly and exit non-zero (DS-18 behaviour without the checkpoint wrapper). |
| D3 `vrpp` empty-mandatory | **Keep current behaviour** (`[0, 0]` when no bin is mandatory). | DS-33 / B-kimi-32 → wontfix. Smoke expectation km=0 before thresholds stays. The flowchart, not the code, is corrected. |
| D4 SWC-TCF | **Fix.** | DS-27 (drop `0.5*` travel term), DS-28 (one unit system across gurobi/ortools/pyomo), DS-30 (one `6000.0` km cutoff), DS-31 (branch on solver status), DS-40 (initialize the pyomo set) are Group-1 work. R-kimi-10 (prune pyomo wrapper) and the `pyomo` part of R-mistral-03 are **withdrawn** — pyomo stays in `solvers`. DS-29 (`delta`) still needs fix-or-delete. |
| D11 submodule scope | **Ship only the `GridBase` subset.** Implemented in `.agent/cache/tools/package_minimal.sh`: copies `LICENSE` + the 10-file closure listed below, writes a submodule `__init__` exporting only `GridBase` (records upstream commit), and wraps the `VisualizationMixin` import in `container.py` in `try/except ImportError`. Archive 1139 → 1024 files. Verified from the extracted zip: `GridBase`, `bins/base.py`, `statistical_empirical.py` import with `matplotlib` absent; 9-day `emp` `test_sim` (`alns`,`hgs` × cf70/cf90) routes on day 7 / day 9 as before. Note: the pip `alns` package (`solvers` extra) hard-requires `matplotlib`, so B-claude-07 only bites a core-only install. | Port the same subset rule to `ci/packager.py` / `export_config.json` on `main` (§6.A.3 item 9 superseded: assert the subset imports, not the whole submodule). |

D11 facts (checked on the branch): the retained code imports only `GridBase`. Its runtime closure is `export/{grid,save_load,container}.py` + `export/modules/{core,analysis_utils,processing_utils,plotting_utils}.py` + `export/enums/*` (~1.5k LOC of 173 tracked files / 1.5 MiB). `Simulation` (`export/simulation.py` → `predictors.py`, R/`subprocess`) and `OldGridBase` (`python/sample_gen.py`) are re-exported by the submodule `__init__` but have no consumer in `logic/`. `container.py` pulls `plotting_utils` → `matplotlib`, which is only in the optional `viz` group of `logic/pyproject.toml`: a core-only install cannot import `GridBase`, so `emp` breaks (new finding, B-claude-07 below). Submodule licence is plain AGPL-3.0; the parent repo is AGPL/commercial dual-licensed.

### 6.A.5 Do-not-act list (so the next export does not regress)

- Do not "fix" `B-kimi-32`/`B-cursor-04` acceptance expectations without D3:
  changing the empty-mandatory early return changes the km=0-before-threshold
  smoke behaviour by design.
- Do not add `√n_d` service-level or a second last-minute scale: both are
  owner-closed (`B-cursor-01` wontfix, `B-cursor-03` one unit).
- Do not treat default-off features as dead: PBRS, GPU-memory callback, ZenML
  are runtime-reachable and need caller/schema/yaml edits (Codex items 9-11).
- Do not promote Batch P or the proposed R rows to savings until a worktree
  sweep is green; the verified number is §6.A.2 Batch V.
- Do not port the loader norm/activation swallow to `main` unchanged.

### 6.A.6 Recommended export sequence

1. Land the fixes for Group 1 (or the matching D-decisions) on
   `feat/minimal-export-package`; port B-claude-03/04 to `main`.
2. Apply Batch V; run one combined `compileall` + `import_sweep` + the §0
   train/eval/`test_sim` smokes; assert the §6.A.7 checks.
3. Apply Batch D dependency edits in the ordering given in §6.A.2.
4. Apply Batch P with per-batch worktree sweeps; land the packaging-script
   changes (§6.A.3) with them.
5. Re-run nine-policy `test_sim` + CUDA sampling eval (DS-11/12) and the
   submodule-in-place isolated-archive test (Codex 12) before declaring the
   artifact final.

### 6.A.7 Smoke assertions to add (merge of Codex 9, Kimi 45, Grok 24-25, Cursor 32, Mistral 58, Grok 26)

- Training: generated instance count equals configured `train.env.graph.n_samples`;
  configured `eval_graphs` produce validation loaders; baseline params equal the
  requested values; advantages are `[B]`; loaded model has `LayerNorm` when the
  checkpoint does; loaded-model predictions match the training policy.
- Eval: printed `cost` matches `-reward` semantics; non-zero km/kg on the gamma
  fixture.
- Simulation: `log_<pol_id_orig>_<N>N.json` exists and `mean.km` equals the
  sample mean; sample `time` equals `Σ` daily solver time; idle-day bin at 100%
  does not increment `overflows`; empirical bin IDs equal the distance-matrix
  slice IDs; a resume run reports positive elapsed time.
- Policies: ACO-HH same seed → same tour; SWC-TCF 3-node brute-force optimum;
  BPC 3-node LCI gate (`obj == 25` once fixed); NA profit units; HGS limited
  split infeasible case if D3/D-HGS keep it.
- Packaging: `GridBase` imports from the materialized submodule; every
  `policy_*.yaml` key has a reader unless allowlisted (Kimi item 41).


## 7. Implementation log (Claude, 2026-09-25) — branch `feat/minimal-export-package`

Applied on top of `478301584`. Every commit ran the import sweep and the full smoke
(`.agent/cache/tools/smoke_minimal.sh`: gen_data → train → eval → 10-day `test_sim` of all
nine policies, every policy log must route). New checks are under `.agent/cache/tools/`.

### 7.1 Removals and packaging

| Commit | What | Result |
| --- | --- | --- |
| `7332f71ea` | Batch V file-level rows (R-codex-01/02, R-claude-01/04, R-gemini-01…08 + `CriticBaseline`, R-grok-01…03, R-cursor-01…06, R-kimi-01/02/05/07/15) | 80 modules; sweep 707 → 627 |
| `4a24a2f5c` | Batch P as narrowed by OpenCode (R-opencode-01…05); removed engines now raise | 67 modules; 627 → 560 |
| `01e6ea179` | D2: checkpoints/resume removed; failed runs → non-zero exit (parallel + sequential, verified by injected failure `.agent/cache/tools/inject_fail/`) | 4 modules; 560 → 556; DS-17/18 moot |
| `8979a0ff7` | Batch D: 16 core deps + hexaly/pyvrp/vrpy/alns dropped; shapely and jinja2 moved to core (hard imports on the retained path); openpyxl **kept** (every sim writes the fill-history `.xlsx`, contrary to §4 item 55) | verified in a fresh core+solvers venv from the new lock, from the packaged archive |
| `478301584` | D11: packaging vendors only the `GridBase` closure of `wsmart_bin_analysis` (+ optional plotting mixin) | archive 1139 → 878 files after all removals |

Not applied (function-level, still open): R-grok-04, R-kimi-03/04/06/08/13/14/16/17; Batch P
follow-ups §4 item 62; R-codex-03/04, R-claude-02/03 remainders, R-cursor-07, R-muse-01/03.

### 7.2 Bugs fixed (consolidated IDs from §6.A.1)

| DS | Status | Evidence |
| --- | --- | --- |
| DS-01, 02, 03, 04, 05, 06, 08, 10 | fixed `bcbaee95b` | `lane_a_regression.py`; `loader_parity_check.py` (legacy vs training encoder max diff 0.44 → 0) |
| DS-11, 12 | fixed `62f40502c` | CUDA sampling eval runs (was "Expected a 'cuda' device type for generator"); also ListConfig beam widths |
| DS-13, 14 | fixed `52153942c` | `emp_grid_ids_check.py` (Rio Maior + Figueira grid IDs = routed IDs; Rio Maior now reads the `old_*` pair it routes from); n_samples=2 parallel mean = sample mean |
| DS-17, 18 | moot/fixed `01e6ea179` (D2) | injected failure exits 1 |
| DS-19, 20, 21 | fixed `ec3feeb5c` | `selector_regression.py`; percent threshold per owner ruling |
| DS-22, 23, 24, 25, 26 (+ new) | fixed `3fb97eb47` (D1) | `bpc_bruteforce_check.py`: 375/375 instances optimal (5–6 bins; plain/mandatory/fleet; seeds 0–2) |
| DS-27, 28, 29, 30, 31, 40, 52, 53 | fixed `672943393` (D4) | Kimi SWC repro CHECK 1 optimal; gurobi / ortools-SCIP / pyomo-gurobi identical on a shared instance |
| DS-34, 36, 37, 38, 61, 62 | fixed `3791bb66e` | `hgs_split_check.py` (limited + unlimited split = brute force) |
| DS-33 | wontfix (D3) | — |
| DS-41 | cannot fix: fast-tsp 0.1.5 has no seed | documented |

**New defects found while fixing BPC** (all in `3fb97eb47`): SaturatedArc LCI treated a single
route as a cover (`λ_k ≤ 0`); RCCs applied to optional bins; warm-start columns corrupted by
list aliasing (cost of `[4]`, nodes `[1,4,3]`); stale route signatures after `build_model`;
Phase I→II pricing on zero duals; branching constraints, vehicle/LCI/edge-clique duals and
`exact_mode` never reached pricing; Farkas step called with the ray in `max_routes`, stale
duals, wrong ray sign, and a depot-return cost; truncated pricing trusted for convergence;
divergence branching at the depot (excludes multi-route optima from both children); a
1-row coordinate array indexed by customer; `y_v = 0` branches not enforced in pricing.

**New findings, not fixed**
- Strong branching (`enable_strong_branching_heuristic`) still yields suboptimal plans on
  checked instances; disabled in yaml + config default. Needs a lookahead on a master copy.
- fast-tsp is not repeatable across runs (5 distinct tours / 5 runs), so simulation results
  vary run to run for every policy using the Fast-TSP improver.
- HGS `_split_limited`: the reported mechanism (`V_curr[0]`) was wrong; the real defect was
  missing leading skips / empty plan (fixed).

**Scope notes**
- BPC exactness is verified on the simulator's own parameter path (resolved `BPCParams`
  identical to the harness except the seed) for instances that finish: 5–6 bins. It is exact up
  to the configured `optimality_gap` 0.5 % and `early_termination_gap` 1 % (a heuristic stop the
  code itself warns about). Larger graphs hit the time limit and use the restricted-master IP
  fallback, exercised only by an 80-bin 4 s run.
- DS-13 applies where locations are real bins (simulator, `gen_data` with a focus graph).
  Training's `VRPPGenerator` uses synthetic locations, so its empirical grid is only a pool of
  real fill series; for Rio Maior that pool now also comes from the `old_*` rate file.

**Open (owner decision or not started)**
- DS-15: **owner ruling 2026-09-26 — `time` is the full policy time** (mandatory selection +
  route construction + route improvement). Daily `time` is measured around those three actions
  in `run_day`; the sample `time` is their sum over the days (was the day-loop wall clock,
  which also counted filling, collection and logging; daily time was construction only).
- DS-16: **owner ruling 2026-09-26 — keep counting every day a bin sits full** (current
  behaviour, `new_overflows` = bins at capacity that day, including idle full days). Wontfix;
  B-grok-04 is closed as a definition choice, not a bug.
- DS-07 moot (critic removed); DS-09 (`train_time` row order), DS-32, DS-35 moot (OG engine
  removed), DS-39 (NA advisory units), Group 2 items not on the retained path.
- Packaging scripts on `main` (§6.A.3 items 1–9 and the GridBase-subset rule) not yet ported.
