# Brief: Mistral (lane G: whole-tree dead code, duplication map and config consistency)

Read `.agent/tasks/logic-review-2026-09-26-common.md` first. This lane has no
paper. It is cross-cutting, and the other lanes use it as their evidence base.

## Scope

All of `logic/` on `main`, including the policies outside the retained slice.
Here those policies count as *roots*, not as removal candidates.

## Deliverables

1. **The reachability map (§4, early).**
   - Rerun `.agent/cache/tools/reachability.py` on `main`. It was written for
     the export branch; adapt it to count every registered policy, model,
     selector, improver and acceptance criterion as a root, plus every Hydra
     `_target_` and every yaml under `logic/configs`.
   - Cross-check it with `vulture logic --min-confidence 80` (install it into
     your own `~/.cache/wsr-review/mistral-tools` venv, not into the shared
     one).
   - Publish the list of unreachable modules and symbols early, as a bus
     post, so the other lanes can use it.
   - Split the list into `dead` and `test-only`.
2. **The duplication map (§3).**
   - Run `.agent/cache/tools/dup_finder.py logic 12` and `logic_audit.py
     logic/src`.
   - Cluster the duplicate blocks by topic (operators, route cost, config
     lifting, logging, tensor utilities), and assign each cluster to its
     lane's owner on the bus.
   - File M rows yourself only for clusters outside lanes A to F: `utils/**`,
     `configs/**`, `tracking/**`.
3. **Config consistency.**
   - Dataclass fields in `logic/src/configs/**` that no code reads.
   - Yaml keys that no dataclass declares.
   - Policy yaml files that restate their dataclass defaults word for word
     (deleting those is a candidate M row).
   - A duplicated `defaults:` list across `logic/configs/tasks/*.yaml`.
4. **Dependencies.** Packages in `logic/pyproject.toml` / the root
   `pyproject.toml` with no remaining import on `main`, each with the grep
   evidence.

## Method notes

- Dynamic imports exist: string registries, `importlib` in factories, Hydra
  `_target_`, and lazy `__getattr__` in `logic/src/data/__init__.py` and
  `data/processor/setup.py`. Every "unreachable" claim must survive a grep for
  the module's dotted path *and* its registry key.
- Verify each D row by deleting the item in your worktree and running the
  verification block. The import sweep is heavy, so take the lock, and batch
  several deletions per sweep.
