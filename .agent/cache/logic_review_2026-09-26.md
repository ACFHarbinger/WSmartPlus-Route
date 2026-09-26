# Shared report: logic review of 2026-09-26 (paper fidelity, bugs, refactoring, dead code)

**Branch:** `main` · **Commit under review:** `1aa00c09c`
**Kickoff:** `.agent/bus/2026-09-26.md` ("logic review kickoff")
**Protocol:** `.agent/tasks/logic-review-2026-09-26-common.md`
**Lane briefs:** `.agent/tasks/<agent>-logic-review-2026-09-26.md`
**Previous round:** `.agent/cache/minimal_export_review_2026-09-25.md`. Read §6.A.1 and §7 there
before filing anything.

Rules:

- The report is append-only.
- Each agent adds rows to the shared tables and writes its own §5 section.
- Do not edit another agent's rows.
- A row without evidence is a guess and must be labelled as one.

## 0. Scope, entry points and verification commands

**The retained slice.** It is defined in the common brief. In short:

- the policies `aco_hh alns bpc hgs pg_clns psoma sans swc_tcf na`;
- the selectors `lookahead last_minute service_level`, the improver `fast_tsp`,
  and the acceptance criteria `bmc oi`;
- the Attention Model with REINFORCE on `vrpp`;
- the simulator, training and eval code they run through.

The other registered policies are live code (roots for reachability), not
removal candidates.

**Entry points.**

- `main.py` commands `train`, `eval`, `test_sim`, `gen_data` (and the others
  that `main.py` dispatches).
- `logic/controllers`.
- Registries: `GlobalRegistry`, `RouteConstructorFactory`, and the selector,
  improver and acceptance factories.
- Every Hydra yaml and `_target_` under `logic/configs`.

**Verification block.** Run it from your worktree root, with
`PY=~/.cache/wsr-main-venv/bin/python` and `L="flock ~/.cache/wsr-review/heavy.lock timeout 1800"`:

```bash
$PY -m compileall -q logic
$L $PY .agent/cache/tools/import_sweep.py                    # heavy
$L $PY -m pytest --import-mode=importlib -q logic/test/<touched package> -x   # heavy if >1 file

# Small simulator smoke of the affected policies (heavy):
$L $PY main.py test_sim sim.graph.area=riomaior sim.graph.num_loc=20 sim.graph.n_days=10 \
  'sim.graph.dm_filepath="gmaps_distmat_plastic[riomaior].csv"' sim.graph.focus_graph=graphs_20V_1N_plastic.json \
  sim.graph.load_dataset=null sim.data_distribution=emp sim.cpu_cores=2 \
  'sim.policies=[{alns: ${p.alns.alns}},{psoma: ${p.psoma.psoma}}]'
# Logs land under assets/output/10days/ in your worktree; check that km > 0 on some day for each policy.

# Tiny training smoke (heavy). The template is the "train" step of .agent/cache/tools/smoke_minimal.sh
# (embed_dim=32, n_layers=1, n_samples=64, batch 16, one epoch). Adjust any override main rejects,
# and record the adjustment in your §5.
```

The whole `main` test suite (1343 passed, 3 skipped on 2026-09-26) takes
several minutes. Do not run it. Run only the tests that cover what you touch.

## 1. Paper fidelity (append rows)

Class values:

- `bug`: contradicts the paper with no reason;
- `adaptation`: needed for the VRPP / multi-period / mandatory-bin setting;
- `simplification`: deliberate and harmless;
- `undocumented`: plausible, but not stated in a docstring.

| ID | Policy/model | Paper § / eq. / alg. line | Code `file:line` | Deviation | Class | Severity | Status |
|---|---|---|---|---|---|---|---|

## 2. Bugs, errors and logic mistakes (append rows)

| ID | Sev. | `file:line` | What is wrong | Repro / evidence | Proposed fix | Status |
|---|---|---|---|---|---|---|

## 3. Refactoring, modularity and duplication (append rows)

| ID | Topic | Locations (≥2 `file:line` for duplicates) | What repeats / what is tangled | Proposed shared home and call-site changes | LOC saved | Tests covering today | Risk | Status |
|---|---|---|---|---|---|---|---|---|

## 4. Dead code (append rows)

Kind values: `dead` (no path from any entry point, registry or yaml) and
`test-only` (referenced only by tests).

| ID | Kind | Symbol / file | Importer + registry + yaml evidence | Local deletion + verification result | LOC | Status |
|---|---|---|---|---|---|---|

## 5. Per-agent sections

Each agent adds a section headed `## 5.<lane>. <Agent> — lane <X> — 2026-09-26`.
It lists:

- the files and papers the agent read;
- the IDs it filed;
- its disagreements;
- its questions for the owner.

Codex adds its review notes under its own section.

## 6. Consolidation (Claude + Codex, after all the lanes report)

## 7. Owner decisions

## 8. Implementation log (Claude, after the owner rules)
