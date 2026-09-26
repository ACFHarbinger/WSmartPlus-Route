# Brief — common protocol for the minimal-export review (all agents)

**Branch:** `feat/minimal-export-package` · **Commit under review:** `70e660b03`
**Bus:** `.agent/bus/2026-09-25.md` · **Report:** `.agent/cache/minimal_export_review_2026-09-25.md`
**Your lane brief:** `.agent/tasks/<agent>-minimal-export-review.md`

## Goal

Two deliverables, both written into the shared report:

1. **Bugs / errors / logic mistakes** in your lane of the retained `logic/` tree
   (§2 table, one row per finding, `B-<agent>-NN`).
2. **Further removals** that keep the retained policies/models working (§3
   table, one row per item, `R-<agent>-NN`). Files, classes, functions, config
   keys, yaml files, dependencies — anything the packaging scripts should drop
   next time.

Plus your own `## <N>. <Agent> — lane <X> — 2026-09-25` section in §5: files
read, IDs you added, disagreements, questions for the owner.

## Method

1. Read report §0 (functional contract) and §1 (inventory). Read your lane's
   entry points first, then follow imports outward. For policies, start from
   the `policy_*.py` adapter and the `execute()` path the simulator calls
   (`logic/src/pipeline/simulations/actions/route_construction.py`).
2. For bugs, prefer things that change results or crash: shape/device
   mismatches, wrong units (percent vs fraction, km vs m), off-by-one on
   depot index 0, seeds not forwarded, config keys read under a different
   name than written, exception handlers that swallow failures and return
   empty tours, dead `elif` branches for removed features that now mask
   errors. Run the code where you can — a traceback beats a hypothesis.
3. For removals, the test is reachability from the retained paths, not
   "looks unused": `grep -rn "<symbol>" logic --include=*.py` (exclude the
   defining file), check `__init__.py` re-exports and registries, then remove
   the item in your worktree and run the verification block from report §0
   (at least `compileall` + `import_sweep.py`; a smoke run when the item is
   on a runtime path). Report the LOC saved.
4. Every row carries evidence. Bug rows: `file:line` + repro (command,
   traceback, minimal input). Removal rows: importer check summary +
   verification run + the packaging hook (`prune_codebase.py` flag,
   `export_config.json` key, `remove_*.py` script, or "manual").
5. Do not commit or push. Do not edit another agent's rows or section. Do not
   touch `docs/moon/*`. Local worktree only:
   `git worktree add /tmp/<agent>-wsr feat/minimal-export-package && ln -s
   /home/pkhunter/Repositories/Doc/WSmart-Route/data /tmp/<agent>-wsr/data`.
6. When a candidate removal would require a small code change to stay
   functional (for example collapsing five dataset classes into one), write
   the change as a proposal in the row's "Why it is safe" column, not as a
   diff in the tree.

## Severity and status vocabulary

- Severity: `blocker` (crashes a retained path), `major` (wrong results or
  silent misbehaviour), `minor` (edge case, misleading message, dead branch).
- Status: `open`, `disputed` (say by whom, in your §5), `fixed` (only for
  things already landed on the branch, with the commit hash).

## Done

Post `### <Agent> — 2026-09-25 (lane <X> done)` on the bus with: number of
bug rows, number of removal rows with their total LOC, the three findings you
consider most important, and a pointer to your §5 section. Codex will review;
answer review questions in your §5 section, not by editing rows in place.
