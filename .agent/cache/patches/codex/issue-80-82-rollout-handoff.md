# Codex — rollout baseline redesign and greedy invocation refactor

Base: `6d500ed02`. Apply `issue-80-rollout-comparison.patch`, then
`issue-82-rollout-greedy-helper.patch`. These are patches only: no shared source
changes or pushes. Scratch: `/tmp/wsr-codex-rollout-20260928`; integrated review:
`/tmp/wsr-codex-integration-20260928`. The sandbox permits `/tmp`, not the brief's
usual writable cache worktree. Scratch commits are local packaging checkpoints.

## Behavior changes (#80, B-codex-01/02)

- `DataMixin.setup('fit')` configures a separate comparison environment copied
  from the training environment, with its graph, distribution and reward rules.
  The sample count is the training graph's configured `n_samples` (minimum two
  for a paired test; existing fallback ten). Reporting `val_dataset`,
  `val_datasets`, and `eval_envs` remain reporting-only.
- The baseline owns a fixed comparison pool. Both policies roll out on that pool
  and its matching private environment. A candidate must improve mean reward
  and pass the one-sided paired t-test before promotion. Missing setup, invalid
  reward counts, nonfinite rewards and worse candidates never replace weights.
- Accepted promotion prepares a fresh pool first, then installs the frozen
  policy and new pool together. Rejection retains both objects. Generation
  failure leaves the old policy/pool intact. Repeated setup retains the pool;
  use a new baseline for a different training distribution.
- Pool generation uses root seed + promotion generation, isolated from global
  Python/NumPy/Torch streams and the training generator. Checkpoints save the
  seed, count and generation; the same pool is reconstructed on resume before
  the next comparison. Older checkpoints without pool metadata still load.
  Exact regeneration assumes unchanged generator/configuration, as other
  seeded datasets do. The private env moves to the policy device at callback.
- Warmup forwards pool configuration and callbacks. Candidate train/eval mode
  and legacy decoding-strategy fields are restored after evaluation.

This deliberately changes training trajectories and adds one comparison dataset
at fit setup and one after each accepted promotion. No default baseline type,
alpha, warmup length, or reward scale is changed.

## Refactor (#82, M-codex-01)

One private greedy-invocation helper handles the legacy `set_strategy`/tuple API
and the modern `(td, env, strategy='greedy')` API. Dataset reset/padding and batch
copying stay in their original paths. Tests compare real calls through both
paths, including an incomplete final batch and the missing-env contract.

## Verification

- **48 passed** across rollout tests, existing baseline tests, Lightning module
  setup, REINFORCE and time-tracking tests in the isolated lane tree.
- **45 passed** on the integrated other-lane tree. The three time-tracking tests
  are absent there because Mistral deletes that test-only implementation.
- Real CPU `VRPPEnv` plus `AttentionModelPolicy`: seeded three-instance pool,
  greedy frozen-policy rollout with final-batch padding, finite rewards, source
  data unchanged. GPU execution was not available/verified.
- Before applying Codex's patches, the three selected no-comparison/multigraph
  regressions fail. The three greedy API parity tests pass both before and after
  helper extraction. The final lifecycle suite also covers accepted/rejected
  promotions, warmup, RNG isolation, checkpoint load ordering, legacy checkpoint
  loading, failed generation, invalid rewards, and repeated setup.
- Ruff passes on changed production modules and new tests; compileall passes
  over the integrated `logic/src` tree. Both patches apply cleanly on the
  combined reviewed tree. No simulator run: these changes affect RL training,
  not a simulator route constructor.
- Independent reviewer found a repeated-setup reset bug in the initial draft;
  it is fixed with two regression cases in the delivered #80 patch.

Other lanes are **not collectively approved**. See logic-review report §10 and
`code-cleanup-reviewed-sha256.txt` for the exact reviewed patch versions and
remaining revisions. This handoff does not authorize application of their stack.
