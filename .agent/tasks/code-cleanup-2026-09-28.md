# Brief: code clean-up round (#80 C3 robustness, #81 C4 dead code, #82 C5 refactors), 2026-09-28

**Base:** `main` at `6d500ed02`. **Row details:** `.agent/cache/logic_review_2026-09-26.md` (B/D/M rows, with the Codex, Muse,
DeepSeek and OpenCode verdicts in §5). **Plan of record:** `.agent/cache/paper_update_2026-09-26.md` §8.1 (code track C3–C5).
**Bus:** `.agent/bus/2026-09-28.md`.

## Owner rulings for this round

- **Test-only code is deleted together with the tests that only exercise it:** D-kimi-05, D-mistral-09 (the DR-ALNS cluster, plus its
  `train.yaml` `rl.dr_alns` section, `DRALNSConfig` and the `gymnasium` dependency if nothing else uses it), and D-mistral-10.
- **The six unused `logic/gen` scripts are removed** (D-mistral-12). `gen_paper_latex.py`, `report_utils.py` and `gen_dist_matrix.py` stay.
  Drop any dependency that only they used.
- **PG-CLNS switches to the shared operators** (M-qwen-01, fixing B-qwen-03/04). Where the operators should stay identical, a parity test
  is required. Where behavior changes (for example worst removal gains the Ropke & Pisinger randomisation), say so in the handoff.
- **D-grok-01 (`yaml_to_env.py`) stays.** It is disputed: it has live shell callers.
- Already done on `main`, so do not redo them: all of C1 (#78) and C2 (#79). `ensure_registered()` now imports every `policy_*.py`, and
  EGH/LASM import and route.

## Lanes (file ownership is exclusive; ask on the bus before touching another lane's files)

| Agent | Issue items | Files owned |
|---|---|---|
| **Grok** | #80: B-grok-02 (resume clock), B-grok-03 (failed samples: non-zero exit, no zero means), B-grok-04 (`sample_id` popped too early), B-grok-05 (resume key), B-grok-06 (stats-file index), B-grok-07 (display-name parser), plus the NA runner's `model_ls` 2-tuple vs 3-tuple unpack in `states/running.py` (A-gemini-03). #82: M-grok-01, M-grok-02 | `pipeline/simulations/**` except `actions/route_construction.py`; `pipeline/features/test/**` |
| **Gemini** | #80: B-gemini-01 (normalization kwarg), B-gemini-02 (NA empty tour → `[0, 0]`), B-gemini-03 (NA revenue in kg). #81: D-gemini-01 and B-gemini-04/05 (dead projection layers), **keeping existing checkpoints loadable** (filter the removed keys in `utils/model/loader.py`, with a test that loads a checkpoint saved before the change). #82: M-gemini-02, M-gemini-03, and the NA part of M-cursor-01. M-gemini-01 (unify the two AM classes) only with a parity test over the same weights; otherwise write a plan and skip it | `models/**`, `utils/model/**`, `neural_agent/**`, `envs/**` |
| **Codex** | #80: B-codex-01, B-codex-02 (the rollout baseline gets its own comparison dataset and env; it is refreshed only after an accepted promotion; a worse candidate is rejected). #82: M-codex-01. **Also the integration reviewer:** review every lane's patch before Claude applies it (report §10) | `pipeline/rl/**`, `pipeline/features/{train,eval}/**` |
| **Kimi** | #80: B-kimi-34, 36–43 (BPC), B-kimi-50–52 (ACO-HH `except`, elitism ceil vs diagram, stale docstrings), B-kimi-55–59 (SWC-TCF gap, empty-day `[0, 0]`, fleet fallback, route extraction and the highs time limit, typed overrides flattened). #81: D-kimi-01–06. #82: M-kimi-02, M-kimi-03, M-kimi-04; M-kimi-01 (MS-BPC-SP's copies of helpers) only with brute-force parity | `exact_and_decomposition_solvers/**`, `helpers/solvers_and_matheuristics/**`, `hyper_heuristics/ant_colony_optimization_hyper_heuristic/**`, `base/base_routing_policy.py` (for B-kimi-59) |
| **Qwen** | #80: B-qwen-03/04 via M-qwen-01 (owner: switch to the shared operators). #81: D-qwen-01 / D-mistral-05 (`sans_opt.py`). #82: M-qwen-02 (SANS operators), M-qwen-03 (the HMLNS ALNS copy), the SANS part of M-cursor-01 | `meta_heuristics/**` (SANS included), `helpers/operators/**`, `helpers/local_search/**` |
| **Cursor** | #80: B-cursor-03 (BMC/OI docstring examples). #82: M-cursor-02 (VRPP training reward vs simulator profit: one shared function, or a parity test if merging is too risky), M-cursor-03 (scalar vs vectorized selector parity test) | `policies/{mandatory_selection,acceptance_criteria,route_improvement,vector}/**` |
| **Mistral** | #80: B-mistral-03 (SA `iterations_per_temp`: wire it or delete it), B-mistral-04 (unused tracking fields). #81: D-mistral-01–04, 06–08, 11, 09, 10, 12 (per the rulings) and D-cursor-01, plus stale references to deleted code in `AGENTS.md` and the docs (for example the `boolmask` reference in §6.1). #82: M-mistral-01, M-mistral-02 | everything else: `logic/gen/**`, `configs/**`, `logic/configs/**`, `tracking/**`, `utils/**` (except `utils/model`), `pyproject.toml`, docs |

## Rules

1. **Worktree** `git worktree add --detach ~/.cache/wsr-review/<agent>-code 6d500ed02`. Then
   `git submodule update --init logic/src/pipeline/simulations/wsmart_bin_analysis` and symlink `data`. Use the shared read-only venv
   `~/.cache/wsr-main-venv`. **Never edit the shared checkout**, and never touch `assets/papers/**`.
2. **Deliver patches:** `.agent/cache/patches/<agent>/issue-<N>-<topic>.patch` against `6d500ed02`, one per issue, plus a handoff note.
   Every bug fix comes with a regression test that fails before the fix. Every deletion needs the importer, registry and yaml check,
   then a green `compileall` and `import_sweep.py`.
3. **Verification in the handoff:** the affected unit tests; for BPC changes, `.agent/cache/tools/bpc_bruteforce_check.py` (exactness
   must hold); for policy behavior changes, a 10-day riomaior-20 `test_sim` of the affected policies (`sim.cpu_cores=2`). Heavy jobs go
   through `flock ~/.cache/wsr-review/heavy.lock`. **Never run pytest while reading fresh simulator output:** `conftest` deletes untracked
   `assets/output` directories.
4. **Behavior changes** (results differ from before) are listed in the handoff, so the export README and #83 can record them.
5. **Post on today's bus** (`.agent/bus/2026-09-28.md`), not an older file: a claim, then `### <Agent> — 2026-09-28 (code cleanup lane done)`.
