# Brief: open issues round (#90, #41, #57, #61), 2026-09-29

**Base:** `main` at `dfb049e7e`. **Bus:** `.agent/bus/2026-09-29.md` (post only there). **Reviewer:** Codex reviews every patch
before Claude integrates it. #83 (reruns) stays blocked on the owner's run records and is not part of this round.

## Non-negotiable safety rules (the owner's data was destroyed once on 2026-09-28)

1. **Never edit the shared checkout** (`/home/pkhunter/Repositories/Doc/WSmart-Route`), and never touch `assets/papers/**` or `data/`.
   Work only in `git worktree add --detach ~/.cache/wsr-review/<agent>-open dfb049e7e`.
2. **The `data` link:** after `git worktree add`, run `ln -s /home/pkhunter/Repositories/Doc/WSmart-Route/data data`. `/data` is in
   `.gitignore`, so `git status` must never show it. **A patch must never contain `data`, a symlink (mode 120000), or a path outside your lane.**
   Before delivering, run
   `grep -E "^diff --git a/data|120000" <patch>`. It must print nothing.
3. **Stage explicit paths** (`git add -- logic docs ...`), never `git add -A` over the whole tree.
4. **Resources:** heavy jobs go through `flock ~/.cache/wsr-review/heavy.lock`. Pass `sim.cpu_cores=2`. Only one heavy job at a time.
   Never run pytest while reading fresh `assets/output` results, because `conftest` deletes untracked output directories.
5. **Deliver** `.agent/cache/patches/<agent>/issue-<N>-<topic>.patch` plus a handoff with SHA-256. Every behavior change comes with a
   test that fails before it. Every refactor comes with a parity test against the previous behavior; record outputs from the base commit
   when the old code is deleted.

## Lanes

| Agent | Issue | Work |
|---|---|---|
| **Kimi** | #90 | M-kimi-01 (MS-BPC-SP's copies of the shared helpers), only with brute-force parity on MS-BPC-SP instances; M-kimi-03 (the SWC-TCF model built once for all backends), with a cross-backend parity test. |
| **Kimi** | #41 (solver side) | Why SWC-TCF stops collecting mid-horizon at Figueira da Foz N=350 / Gamma-3 (LA and SL2). Rerun those cells with Grok's instrumentation. Report whether it is infeasible forced sets, time limits without an incumbent, the arc cutoff, or selection returning nothing. Propose a fix. **Report first, then patch.** |
| **Grok** | #41 (instrumentation) | Log the solver status and the mandatory set every day, even when the tour is empty (today the logger erases `mandatory_nodes` on empty tours; roadmap §I.3). Deliver the patch early so Kimi can run with it. |
| **Grok** | #90 | Validate an external stats-file waste sample's row count up front (it needs `n_days + 1` rows) with a clear error. Include a test. |
| **Gemini** | #90 | M-gemini-01 (unify the legacy `AttentionModel` with `AttentionModelPolicy`), with weight-level output parity and old checkpoints still loading. M-gemini-03 (the VRPP init/context embedders), with parity. If parity can't be shown, deliver the plan and skip. |
| **Qwen** | #90 | M-qwen-02 (SANS operators onto `helpers/operators`) and the SANS part of M-cursor-01 (SANS onto `BaseRoutingPolicy._run_solver`), with parity for unchanged operators and disclosure of intended changes. |
| **Mistral** | #57 | Author `.gitlab-ci.yml` equivalent to the four GitHub workflows. Validate it locally (`gitlab-ci-local` or a lint via the GitLab schema), and check the Gitea/Forgejo mirrors for `uses:` actions that aren't available. There is no real runner, so say exactly what was and wasn't verified. Lane: `.gitlab*`, `.gitea/`, `.forgejo/`, `.github/`. |
| **Cursor** | #61 | Run `.agent/skills/systematic-bug-hunt.md` as written, starting the tracker `docs/errors/ROADMAP.md`. Focus on components the 2026-09-26 logic review did not cover (`logic/controllers`, `data/**` pipeline, `envs`, `utils/**`). Fix clear bugs with tests; report anything that needs a design decision. Lane: those paths plus `docs/errors/`. |
| **Codex** | review | Review every patch. Check the safety rules first: `data`, symlinks, lane paths. |

Post `### <Agent> — 2026-09-29 (open-issues lane done)` with the patch hashes when done.
