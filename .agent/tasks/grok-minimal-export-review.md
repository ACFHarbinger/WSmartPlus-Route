# Brief — Grok (lane B: simulator, orchestration, result writers)

Read `.agent/tasks/minimal-export-review-common.md` first. Bus: `.agent/bus/2026-09-25.md`.

## Lane B scope

- `logic/src/pipeline/simulations/` — `simulator.py`, `day_context.py`, `states/*`, `actions/*`, `bins/*`, `repository/*`, `checkpoints/*`, `failure_analyzer.py`.
- `logic/src/pipeline/features/test/` — `engine.py`, `config.py` (policy expander), `orchestrator/*`, `validation.py`, `zenml_sim_pipeline.py`.
- `logic/src/tracking/` — the shim `__init__.py` and `logging/{log_utils,logger_writer,pylogger}.py`, `logging/modules/{analysis,storage,gui,metrics,failure_emit}.py`.
- `logic/src/utils/{infrastructure,data,input,configs}/`, `logic/controllers/*`, `main.py`, `__main__.py`.

## Questions this lane must answer

1. **Result correctness.** Follow one simulated day end to end (`run_day` → Fill → MandatorySelection → RouteConstruction → RouteImprovement → Collect → Log) and check the numbers written to `log_*.json` and the `.jsonl` stream: km from `get_route_cost` on the *global* matrix, kg collected vs bin fill units (percent vs kg: `bins/base.py`, `actions/collection.py`), overflow counting, `time`, and the `mean`/`std` aggregation across samples. Any unit or index-0/depot mistake is a `major`.
2. **Naming.** B-claude-06: `resolve_policy_display_name`/`to_slug`/`get_full_policy_name` in `day_context.py` vs the ids built by `features/test/config.py::expand_policy_configs`. Propose one rule.
3. **Parallel path.** `orchestrator/parallel_runner.py` + `simulator.py::init_single_sim_worker` with `sim.cpu_cores>1`: does it still work now that the dashboard (`monitor.py` returns `display=None`) and tracking (`wst.init_worker` is a no-op) are gone? Run a 2-core sim.
4. **Checkpoints.** `checkpoints/{manager,hooks,persistence}.py` with `sim.checkpoint_days>0` and `sim.resume=true`: does resume restore bins and day counters? Removal row if the owner does not need it (say so explicitly).
5. **Failure analysis and GUI feed.** `failure_analyzer.py` + `modules/failure_emit.py` and `modules/gui.py` (`send_daily_output_to_gui`, `popup.html`, jinja2): which parts only fed the removed Studio app? R-claude-07 is yours.
6. `repository/filesystem.py`: hard-coded file names per area (`Rio_Maior_Sensores_..._104.csv`, `coordinates104.csv`, `out_info[figdafoz].csv`, ...). Which branches are reachable for `riomaior`/`figueiradafoz` with the data files that actually exist in `data/simulator`? Dead branches → removal rows; wrong file names → bug rows.
7. `utils/input/`, `utils/data/loader.py`, `utils/infrastructure/setup_*.py`: reachability from `test_sim`/`gen_data`.

Post claims and findings on the bus under `### Grok — 2026-09-25 (...)`.
