# Brief — Codex / Chat (lane A: training + eval pipeline; reviewer of all lanes)

Read `.agent/tasks/minimal-export-review-common.md` first. Bus: `.agent/bus/2026-09-25.md`.

## Lane A scope

- `logic/src/pipeline/rl/` — `core/reinforce.py`, `core/losses/`, `common/base/{module,steps,data}.py`,
  `common/baselines/` (10 files; only exponential/rollout/critic are reachable from `train.yaml`),
  `common/{trainer,epoch,pbrs_wrapper,reward_scaler,reward_scaler_batch,route_improvement}.py`.
- `logic/src/pipeline/features/train/` — `engine.py`, `model_factory/{builder,registry}.py`, `zenml_train_pipeline.py`.
- `logic/src/pipeline/features/eval/` — `__init__.py`, `engine.py`, `evaluate.py`, `evaluators/`, `drift_detection.py`, `validation.py`, `zenml_eval_pipeline.py`.
- `logic/src/utils/{model,decoding,tasks,functions}/`.
- `logic/src/configs/rl/__init__.py`, `configs/tasks/{train,eval}.py`, `configs/models/*`, and `logic/configs/tasks/{train,eval}.yaml`.

## Questions this lane must answer

1. Does REINFORCE training on `vrpp` do what the config says: baseline choice (`exponential`, `rollout`, `critic`), entropy weight, gradient clipping, `train_time` multi-day handling, validation reward? Read `reinforce.py` and `common/base/*` against `train.yaml` and list every config key that is read under another name or never read.
2. `features/eval`: `make_dataset` landed in `70e660b03` (report B-claude-03). Check the rest of the path: `_build_dataset_kwargs`, `_eval_multiprocessing`, result file naming/`overwrite`, `get_best` and cost sign conventions (eval prints "Average cost: -3.2" for a profit objective — is the sign right, and does `np.trim_zeros` on sequences drop real depot visits?).
3. `utils/model/loader.py`: the legacy `AttentionModel` (used by eval and by the Neural Agent) is a different class from the training policy (`AttentionModelPolicy`). List every constructor argument the loader passes and check each against the saved `config.yaml`/`args.json`; find the next mismatch after `feed_forward_hidden` (B-claude-04).
4. `pbrs_wrapper.py`, `reward_scaler*.py`, `route_improvement.py`, `rl/core/losses/*`: reachable from the retained config? Removal rows if not.
5. Which of the six `evaluators/` and which baselines are reachable from `eval.yaml`/`train.yaml`? Removal rows for the rest.
6. ZenML pipelines (`zenml_*_pipeline.py`) and `configs/tracking.py` ZenML fields: propose the removal (R-claude-06) with the exact edits `train/engine.py`, `test/engine.py`, `eval/__init__.py` need.

## Review duty

You are the reviewer for lanes B–G. As rows land, check each bug row's repro and each removal row's importer evidence. Put review notes in your own §5 section (`Review of <agent>: R-xx confirmed / B-yy disputed because ...`). A row is not "confirmed" until you or Claude have re-run its evidence.

Post claims and findings on the bus under `### Codex — 2026-09-25 (...)`.
