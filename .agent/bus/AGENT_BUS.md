# AGENT_BUS — index

Shared, append-only log for the multi-agent work on WSmart+ Route. One file
per day, so it never grows into a single unreadable blob.

**Post new entries to today's file:** `.agent/bus/<YYYY-MM-DD>.md`
(create it if today has none yet, and add a row to the table below).

Entry heading convention:

```
### <Agent> — YYYY-MM-DD (topic)
```

Write findings, not status. "Regenerated the 90d Pareto figure" is a status;
"the 90d Pareto front contains no ALNS point because ALNS was never run at
90 days" is a finding. Contradict other agents freely — the bus is where
disagreements get settled in the open, with evidence.

**Reading history:** the current day lives under `.agent/bus/`; older days
move to `.agent/archive/bus/` unchanged (never rewritten or summarised).

| Day | Location |
|---|---|
| 2026-09-26 (current) | `.agent/bus/2026-09-26.md` |
| 2026-09-25 | `.agent/bus/2026-09-25.md` |
| 2026-08-28 | `.agent/bus/2026-08-28.md` |
| 2026-08-27 | `.agent/bus/2026-08-27.md` |
| 2026-08-26 | `.agent/bus/2026-08-26.md` |
| 2026-08-25 | `.agent/bus/2026-08-25.md` |

When 2026-09-25 stops being "today", move it to `.agent/archive/bus/` and
start a fresh dated file. 08-25 and 08-26 are held back from the archive for
now: the website re-skin thread on 08-26 is still live and agents are still
appending to it.

## Roles for the current effort (2026-09-26: logic review on `main`)

Report of record: `.agent/cache/logic_review_2026-09-26.md`. Briefs:
`.agent/tasks/logic-review-2026-09-26-common.md` + `.agent/tasks/<agent>-logic-review-2026-09-26.md`.
Commit under review: `1aa00c09c` on `main`. Report only; Claude implements after the owner rules.

| Agent | Lane | Scope |
|---|---|---|
| **Claude** | lead | Bus, report skeleton, consolidation (§6) with Codex, implementation (§8). |
| **Codex (Chat)** | A + review | REINFORCE/baselines vs Kool et al. §4, train/eval features, `utils/{model,decoding,tasks,functions}`; reviews every lane's rows. |
| **Grok** | B | Simulator day loop, actions/states, parallel runner, result writers, `main.py`, controllers. |
| **Agy (Gemini)** | C | Attention Model vs Kool et al. §3, subnets/embeddings, `utils/model/loader.py`, Neural Agent. |
| **Kimi** | D | BPC, SWC-TCF, ACO-HH vs their papers; `helpers/solvers_and_matheuristics`. |
| **Qwen** | E | ALNS, HGS, PG-CLNS (vs HVPL), PSOMA, SANS vs their papers; operator duplication map. |
| **Cursor** | F | Selectors, `fast_tsp`, `bmc`/`oi`, policy base classes/factory/registry, `interfaces`, `envs`, `data`, `constants`. |
| **Mistral** | G | Whole-tree reachability/dead code, duplication clusters, config consistency, dependencies. |

## Roles for the previous effort (2026-09-25: minimal-export review, historical)

Report of record: `.agent/cache/minimal_export_review_2026-09-25.md`. Briefs:
`.agent/tasks/minimal-export-review-common.md` + `.agent/tasks/<agent>-minimal-export-review.md`.
Branch: `feat/minimal-export-package`.

| Agent | Lane | Scope |
|---|---|---|
| **Claude** | lead | Bus, report skeleton, consolidation (§6) with Codex. |
| **Codex (Chat)** | A + review | RL/training/eval pipeline, `utils/{model,decoding,tasks,functions}`, train/eval configs; reviews every lane's rows. |
| **Grok** | B | Simulator, orchestrator, result writers (`tracking/logging`), controllers, `main.py`. |
| **Agy (Gemini)** | C | Neural model stack (`models/**`), model configs, `utils/model/loader.py` (with Codex). |
| **Kimi** | D | BPC, SWC-TCF, ACO-HH, Neural Agent, policy base, `helpers/solvers_and_matheuristics`. |
| **Qwen** | E | ALNS, HGS, PG-CLNS, PSOMA, SANS, `helpers/{operators,local_search}` reachability. |
| **Cursor** | F | Selection/improvement/acceptance, `vector`, `interfaces`, `envs`, `data/**`, `constants`. |
| **Mistral** | G | Whole-tree reachability, yaml↔dataclass consistency, dependency pruning, packaging-script edits (report §4). |

## Roles for the paper effort (2026-08-25 → 2026-08-28, historical)

| Agent | Role |
|---|---|
| **Claude** | Team lead / task manager. Owns issue tracking, this bus, and whichever foundational piece unblocks the others on a given task. |
| **Codex (Chat)** | Co-lead / reviewer. Verifies claims against source data; reviews all agents' diffs before they are considered done. Currently: #58 (CTOP RL-envs review), then the general bug/lint pass over `logic/src/` (#61). |
| **Agy (Gemini)** | Currently: #59 (wire the test simulator to CTOP's time budget), then the policies-vs-bibliography cross-check pass (#62). Previously: website visual design and identity for `docs/website/`. |
| **Grok** | Joined 2026-08-27, in place of Opencode. Currently: #60 (CTOP Hydra config tree), then the models-vs-bibliography cross-check pass (#63). |

Historical note: Opencode held the website-interactive/3D-visualisation lane
through the paper work (see `.agent/tasks/opencode-website-interactive.md`);
Grok has taken its place in the rotation going forward.

## Ground rules

1. **Every number in the paper traces to a file.** If a claim cannot be
   derived from `docs/private/global/simulation/simulation_summary*.csv` or a raw
   log under `assets/output/`, it does not go in the paper. Report the gap
   instead of filling it with plausible prose.
2. **Update `docs/moon/CHANGELOG.md` and `docs/moon/ROADMAP.md`** as you go,
   in the same commit as the work. Historical entries are never rewritten.
3. **Commit on `main`.** The feature branches of both efforts (`feat/paper-results-and-website`, `feat/minimal-export-package`) were merged into `main` and deleted on 2026-09-26; the export branch is kept as tag `export/minimal-20260926`. Never
   push to `main` (see `AGENTS.md` §5.3).
4. **Stay in your lane.** Two agents editing `docs/website/src/App.tsx` at
   once will conflict. Claim files in the bus before touching shared ones.
