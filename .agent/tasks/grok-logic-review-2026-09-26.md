# Brief: Grok (lane B: simulator and the path every policy runs through)

Read `.agent/tasks/logic-review-2026-09-26-common.md` first.

## Papers

There is no algorithm paper for this lane. The simulation framework paper is
the submodule `assets/papers/Simulation-Framework-for-the-MPVRP-with-Profits-in-Smart-Waste-Collection`.
Initialise it if it is readable; it defines profit, overflow and km/kg. Check
that each metric's definition matches the code, and record the owner rulings
DS-15 and DS-16 as settled.

## Scope

- `logic/src/pipeline/simulations/**`:
  - `actions/`, `states/`, `day_context.py`;
  - `bins/`, the network and distance-matrix loaders;
  - `repository`, `checkpoints/`.
- `pipeline/features/test/**`: the orchestrator and the parallel runner.
- `pipeline/callbacks/simulation/**`.
- `tracking/logging` (the result writers).
- `utils/{infrastructure,data,input,configs}`, `logic/controllers`, `main.py`.

## Questions

1. **The day loop** (`day_context.run_day` plus the actions). Check:
   - fill, then selection, then construction, then improvement, then
     collection, then logging, in that order;
   - that the policy time covers steps 1 to 3 only (DS-15, already landed);
   - that the collected kg and the overflow use the bin state from *before*
     collection;
   - that `must_go` and `[0, 0]` are handled consistently across all nine
     policies.
2. **The config path.** Follow `_flatten_config` / `_EXPANDED_KEYS` in
   `actions/base.py` and the re-link in `states/initializing.py` (B-claude-01
   and B-claude-05 are fixed; look for the next variant that resolves to the
   wrong block). Include the name-only policy entries and the dict-style yaml
   (psoma, sans).
3. **Parallel runs.**
   - Is a seed forwarded per sample/policy?
   - Do the workers share mutable state?
   - Is a worker failure surfaced, or swallowed as a zero-km log?
   - Is the progress/lock handling around log writes race-free?
4. **Duplication.**
   - The log/summary writers that restate the same metric computation.
   - Repeated distance-matrix/coordinate loading.
   - The per-action boilerplate in `actions/*.py`.
   - Any metric computed both in `finishing.py` and in the callbacks.
5. **Dead code.**
   - `elif` branches for problems or features no yaml selects any more;
   - unused state classes;
   - `checkpoints/` helpers that nothing calls (the checkpoint feature itself
     is live on `main`);
   - leftover argparse-era helpers in `utils/configs`.
