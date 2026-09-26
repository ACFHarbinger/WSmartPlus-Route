# Brief: common protocol for the logic review of 2026-09-26 (all agents)

**Branch:** `main` · **Commit under review:** `1aa00c09c`
**Bus:** `.agent/bus/2026-09-26.md` · **Report:** `.agent/cache/logic_review_2026-09-26.md`
**Your lane brief:** `.agent/tasks/<agent>-logic-review-2026-09-26.md`

This is the second pass over the code that the minimal export shipped, now on
`main` (the export branch was merged and deleted; it is kept as tag
`export/minimal-20260926`). The first pass is
`.agent/cache/minimal_export_review_2026-09-25.md`. Read its §6.A.1 (the
consolidated bugs) and §7 (what was fixed) before filing anything, so that
nothing already fixed or already ruled on gets filed a second time.

## Scope: what "the retained slice" means

- **Policies:** `aco_hh`, `alns`, `bpc`, `hgs`, `pg_clns`, `psoma`, `sans`,
  `swc_tcf`, and `na` (the Neural Agent).
- **Around the policies:**
  - selectors `lookahead`, `last_minute`, `service_level`;
  - improver `fast_tsp`;
  - acceptance criteria `bmc` and `oi`.
- **Model and training:** the Attention Model (`logic/src/models/core/attention_model/**`
  plus the subnets and embeddings it uses), trained with REINFORCE on `vrpp`.
- **The code every one of these runs through:**
  - the simulator (`pipeline/simulations/**`, `pipeline/features/test/**`);
  - the training and eval pipeline (`pipeline/rl/**`, `pipeline/features/{train,eval}/**`);
  - `policies/route_construction/base/**`, `policies/helpers/**`, `interfaces/**`,
    `envs/**` (vrpp), `data/**`, `constants/**`, `utils/**`;
  - the matching dataclasses in `logic/src/configs/**` and yaml in `logic/configs/**`.
- **Out of scope:** the other ~100 policies on `main`. They are registered and
  runnable, so **they are not dead code**. Do not file removal rows for them.
  One exception: a helper that a retained policy shares with one of them is in
  scope for bug and duplication rows.

## Four deliverables (all written into the shared report)

1. **Paper fidelity (§1, `P-<agent>-NN`).**
   - Read the source paper for each policy or model in your lane (the PDFs are
     under `bibliography/`; your brief names them).
   - Compare the paper, section by section, with the code path the simulator
     actually runs.
   - Classify every deviation as one of:
     - `bug`: contradicts the paper with no reason;
     - `adaptation`: needed for the VRPP / multi-period / mandatory-bin setting;
     - `simplification`: deliberate and harmless;
     - `undocumented`: plausible, but not stated in a docstring.
   - Give the paper section, equation or algorithm line for each deviation.
   - Adaptations are not defects. Say why they are needed.
2. **Bugs and logic errors (§2, `B-<agent>-NN`).**
   - Prefer anything that changes results or crashes. Examples:
     - wrong units (percent vs fraction, km vs m);
     - an off-by-one on depot index 0;
     - seeds not forwarded;
     - a key read under a different name than it is written;
     - an `except` that swallows a failure and returns an empty tour;
     - a mask or shape error;
     - a sign error on profit vs cost.
   - A `bug` row from §1 also gets a §2 row that links back to it.
3. **Refactoring and modularity (§3, `M-<agent>-NN`).** Look for:
   - logic repeated across files: the same insertion or removal operator
     written three times, several split/decoding routines, route-cost helpers
     reimplemented per policy, copy-pasted param dataclasses, duplicated yaml
     blocks;
   - modules with too many jobs;
   - hard-coded branching that a registry already covers.

   Each row needs:
   - every location (at least two `file:line` for a duplicate);
   - the proposed shared home (an existing module is preferred over a new one);
   - the call sites that change;
   - the LOC saved;
   - which tests cover the behaviour today.

   If no test covers it, say so, because that makes the row riskier. Use
   `.agent/cache/tools/dup_finder.py` and `logic_audit.py` as starting points,
   then read the code: two blocks that look alike but behave differently
   must not be merged.
4. **Dead code (§4, `D-<agent>-NN`).** Dead code is code with no path from
   any of these:
   - an entry point (`main.py` commands, `logic/controllers`);
   - a registry (`GlobalRegistry`, `RouteConstructorFactory`, the selector,
     acceptance and improver factories);
   - a Hydra yaml or `_target_`;
   - a public `__init__` re-export that something consumes.

   Code referenced **only by tests** (`logic/test/**`) gets its own class, `test-only`. The owner
   decides on those, so do not group them with dead code. Kinds of dead code
   to look for:
   - unused functions, classes and methods;
   - `elif` branches for removed features;
   - parameters that are never read;
   - config keys that are never read;
   - commented-out blocks;
   - unreachable code after `return`/`raise`;
   - stale compatibility shims.

## Method

1. Start from the entry path the simulator calls
   (`pipeline/simulations/actions/route_construction.py` → the policy adapter
   `policy_*.py` → its solver). For the model, start from `main.py train` →
   `features/train/engine.py`. Follow imports outward from there.
2. **Evidence or it does not exist.**
   - A bug row needs `file:line`, re-checked with `sed -n`, plus a repro: a
     command, a traceback, or a minimal input with the expected and actual
     output. Repro scripts go in `.agent/cache/tools/<agent>_<topic>_20260926.py`.
   - A dead-code row needs:
     - the importer check (`grep -rn "<symbol>" logic --include=*.py`,
       excluding the defining file);
     - the registry and yaml check (`grep -rn "<name>" logic/configs`);
     - the local deletion in your worktree followed by a green verification
       block (below).
   - A refactor row needs the duplicate locations shown side by side (a diff
     or quoted lines). A diff that proves the blocks are equivalent is better
     than a claim.
3. Label guesses as guesses. A dead-code row without a deletion run is a
   guess.
4. **Do not commit or push, and do not edit the shared checkout.** Work in
   your own worktree. Proposed changes go into the row as prose or as a patch
   file under `.agent/cache/patches/<agent>/<ID>.patch`, never into the tree.
   Claude implements confirmed rows on `main` after the owner rules, as in the
   last round.
5. Do not edit another agent's rows or section. Disagree in your own §5
   section, and on the bus.

## Setup (on-disk worktree, shared venv)

`/tmp` is a 16 GB RAM tmpfs on this laptop, so worktrees go in `~/.cache`:

```bash
cd /home/pkhunter/Repositories/Doc/WSmart-Route
git worktree add --detach ~/.cache/wsr-review/<agent> 1aa00c09c
cd ~/.cache/wsr-review/<agent>
git submodule update --init logic/src/pipeline/simulations/wsmart_bin_analysis
ln -s /home/pkhunter/Repositories/Doc/WSmart-Route/data data
PY=~/.cache/wsr-main-venv/bin/python   # shared, read-only: never pip/uv install into it
```

Run everything from your worktree root. `logic` is imported from the working
directory; I verified this on 2026-09-26. When you finish, remove the
worktree: `git worktree remove --force ~/.cache/wsr-review/<agent>`.

## Resource rules (mandatory, because the laptop froze on 2026-09-25)

The laptop has 24 cores and 31 GB of RAM, and up to eight agents share it.
Every Python process that imports `logic` loads torch.

- **Heavy jobs go through the shared lock.** Heavy jobs are:
  - `main.py test_sim`, `train`, `eval` and `gen_data`;
  - `import_sweep.py`;
  - any pytest run over more than one test file;
  - any script that runs a solver on more than a handful of instances.

  Run them as `flock ~/.cache/wsr-review/heavy.lock timeout 1800 <cmd>`. The
  lock serialises heavy jobs across all agents; wait for it and do not work
  around it.
- **Simulator:**
  - always pass `sim.cpu_cores=2` (the yaml default of 0 spawns up to 23
    workers);
  - use `sim.graph.num_loc` ≤ 50 and `sim.graph.n_days` ≤ 10;
  - pass `sim.policies=[...]` to run only the policies you need; the default
    list has more than 100.
- **Training:** tiny configs only (see the smoke template in report §0). No
  GPU training runs longer than about 2 minutes.
- **Pytest:** no `-n`/xdist, use `--import-mode=importlib`, and target files
  explicitly.
- **Outputs:** keep scratch outputs in your worktree or in
  `~/.cache/wsr-review/<agent>-out/`, never in `/tmp`.
- **Stopping processes:** never `pkill -f` with a pattern your own shell
  matches. Kill by PID.

## Verification block (report §0 has the exact commands)

For each removal or refactor you trial locally:

1. `$PY -m compileall -q logic`.
2. `$PY .agent/cache/tools/import_sweep.py` (heavy: take the lock).
3. The unit tests for the touched package.
4. If the code is on a runtime path, a small `test_sim` of the affected policy
   with the settings above.

Report the result of each step in the row.

## Vocabulary

- **Severity:**
  - `blocker`: crashes a retained path;
  - `major`: wrong results or silent misbehaviour;
  - `minor`: an edge case, a misleading message, or a dead branch.
- **Status:**
  - `open`;
  - `disputed`: say by whom, in your §5;
  - `fixed`: only for things that have landed, with the commit hash.
- **Refactor risk:**
  - `low`: mechanical, covered by tests;
  - `medium`: covered in part;
  - `high`: changes behaviour or is untested.

## Already settled: do not re-file

These are owner rulings from 2026-09-25/26:

- D3: `[0, 0]` stays when no bin is mandatory.
- DS-15: the daily `time` is the full policy time (selection + construction +
  improvement).
- DS-16: overflow counts every day a bin sits full.
- D4: pyomo stays.
- D1: BPC was made exact. It matched brute force on 375 instances with strong
  branching disabled, so an exactness claim needs a counter-example instance.
- Checkpoints/resume were pruned only from the export. On `main` they are live
  code, not dead.

## Done

Post `### <Agent> — 2026-09-26 (logic review lane <X> done)` on the bus. Give:

- the row counts per table (P/B/M/D);
- the total LOC that your M and D rows save;
- your three most important findings;
- a pointer to your §5 section.

Codex reviews the rows. Answer review questions in your §5 section.
