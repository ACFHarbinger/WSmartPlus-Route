# Brief: Codex / Chat (lane A: REINFORCE training + eval; reviewer of all lanes)

Read `.agent/tasks/logic-review-2026-09-26-common.md` first.

## Papers

- `bibliography/models/Attention_Model.pdf`: Kool, van Hoof & Welling, ICLR 2019.
  - Your part is §4: REINFORCE with a greedy-rollout baseline, the paired
    one-sided t-test (α = 0.05) that replaces the baseline policy, the
    regeneration of the baseline evaluation set, and the warm-up with an
    exponential baseline (β = 0.8) in the first epoch.
  - Also Appendix B: the hyperparameters, and gradient clipping at norm 1.0.
- `bibliography/models/POMO.pdf`: read it only if you conclude that the
  `pomo` baseline is reachable from the `train.yaml` default.

## Scope

- `logic/src/pipeline/rl/core/reinforce.py`, `core/losses/`.
- `pipeline/rl/common/`:
  - `base/{module,steps,data,optimization}.py`;
  - `baselines/{rollout,exponential,critic,warmup,base}.py`;
  - `trainer.py`, `epoch.py`.
- `pipeline/features/train/**` and `pipeline/features/eval/**`.
- `utils/{model,decoding,tasks,functions}/`.
- `configs/rl/`, `configs/tasks/{train,eval}.py`,
  `logic/configs/tasks/{train,eval}.yaml`.

## Questions

1. **Paper fidelity of the rollout baseline.**
   - Is the baseline replaced only when the t-test is significant *and* the
     mean improves?
   - Is the baseline dataset regenerated on replacement?
   - Is the warm-up exponential (`WarmupBaseline`), and does it blend like
     Kool's code?
   - Is the loss `(cost − b)·log p` with the right sign for a profit
     objective (the reward is negative cost)?
2. **Config keys.** List every `train.yaml` key that is read under another
   name or never read (entropy weight, grad clip, `bl_alpha`,
   `bl_warmup_epochs`, `train_time`). `_init_baseline` lifts
   `hparams["kwargs"]`: check that every key it needs actually arrives there.
3. **Eval.** Check the cost/reward sign now that `cost = -reward` (commit
   `2d41a869c`). Does `np.trim_zeros` drop real depot visits? Check result
   file naming, `overwrite`, and `get_best`.
4. **Duplication.**
   - The step logic in `common/base/steps.py` against `reinforce.py` and the
     other `core/*.py` algorithms that share it.
   - The dataset (re)generation in `epoch.py` against `features/train/engine.py`.
   - Decoding helpers in `utils/decoding` against `models/.../decoding.py`
     (coordinate with lane C).
5. **Dead code.** Dead functions inside the retained modules only. Other RL
   algorithms (`ppo`, `pomo`, …) are registered, so they are not dead.

## Review duty

You review lanes B to G.

- Re-run each B row's repro and each D row's importer evidence.
- For each M row, check that the two blocks really are equivalent.
- Write your notes in your §5 as `Review of <agent>: <ID> confirmed / disputed
  because …`.

A row is not confirmed until you or Claude have re-run its evidence. After
all the lanes report, consolidate §6 with Claude.
