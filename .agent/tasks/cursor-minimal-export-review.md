# Brief — Cursor (lane F: selection/improvement/acceptance, interfaces, envs, data, constants)

Read `.agent/tasks/minimal-export-review-common.md` first. Bus: `.agent/bus/2026-09-25.md`.

## Lane F scope

- `logic/src/policies/mandatory_selection/` (`selection_{lookahead,last_minute,service_level}.py`, `base/*`), `policies/route_improvement/` (`fast_tsp.py`, `base/*`, `common/*`), `policies/acceptance_criteria/` (`boltzmann_metropolis_criterion.py`, `only_improving.py`, `base/*`), `policies/vector/` (`selection/*`, `__init__.py`).
- `logic/src/interfaces/` (19 files), `logic/src/envs/` (`base/*`, `generators/*`, `routing/vrpp.py`, `tasks/{base,vrpp}.py`, `problems.py`), `logic/src/constants/*`.
- `logic/src/data/` — `datasets/{pytorch,simulation}/*`, `distributions/*`, `network/*`, `generators/*`, `processor/*`, `time/`.

## Questions this lane must answer

1. **Selectors, scalar vs vectorized.** `selection_service_level.py` vs `vector/selection/service_level.py`: the paper review (bus 2026-08-28, Codex) found the service-level term implemented as `z·σ·n_d` where the paper says `z·σ·√n_d` — confirm on this branch, and check that the scalar (simulator) and vectorized (training/NA) implementations of `lookahead`, `last_minute`, `service_level` agree on units (fill percent vs fraction), thresholds (`threshold: 70` vs `0.7`) and horizon semantics. Disagreements are `major`.
2. **Fast-TSP improver.** `fast_tsp.py`: the paper review says `seed` is accepted but not forwarded and `time_limit` is a saved budget of 30 s — confirm, and check how it handles a tour that is `[0]`/`[0, 0]`, duplicate depot entries, and mandatory nodes (must not drop them).
3. **Acceptance criteria.** `boltzmann_metropolis_criterion.py`/`only_improving.py`: `accept()` signature actually used by ALNS/PSOMA/HGS (`accept(current, new, f_best, it, max_it)`), `setup()`/`step()` cooling, `exp(Δ/T)` overflow guards.
4. **Interfaces.** Which of the 19 files are implemented by retained code? (`IEnv`, `IModel`, `IRouteConstructor`, `IRouteImprovement`, `IMandatorySelectionStrategy`, `IAcceptanceCriterion`, `IBinContainer`, `ITraversable`, `context/*`, `distance_metric`, `tensor_dict_like`). Removal rows for orphan protocols; note `interfaces/context/problem_context.py` imports the simulator lazily (import-cycle guard added on the branch).
5. **Envs.** `envs/base/{base,batch,improvement,ops}.py`, `generators/base.py`, `routing/vrpp.py`, `tasks/vrpp.py`: is `improvement.py` (ImprovementEnvBase) reachable? VRPP reward/cost conventions (`COST_KM`, `REVENUE_KG` in `constants/tasks.py`) vs the simulator's `R`/`C`; `get_costs` depot handling; `VRPPGenerator` `max_waste=1.0` vs simulator percent fills (this is why `make_dataset` normalises by `max_waste` — confirm the training side matches).
6. **Distributions (R-claude-02).** Owner wants only `empirical` + `gamma`. Map every consumer: `data/generators/waste.py` (`unif`, `beta`, `dist`, `const`, `emp`, `gamma*`), `bins/base.py` (`sample_dist == "emp" or "gamma" in ...`), `pipeline/rl/common/base/data.py`, `envs/generators/vrpp.py`, `constants/data.py::GAMMA_PRESETS`. Decide what `unif` in the smoke tests should become. Give the exact edits for `distributions/__init__.py` (`DISTRIBUTION_REGISTRY`, `__all__`) and `waste.py`.
7. **Datasets (R-claude-03).** One class for `datasets/pytorch` and one for `datasets/simulation`. Inventory who constructs each of `BaselineDataset`, `ExtraKeyDataset`, `TensorDictDatasetFastGeneration`, `FastTdDataset`, `GeneratorDataset`, `TensorDictDataset` (training: `rl/common/base/data.py`, `pipeline/features/train/*`) and `GenerativeDataset`, `NumpyPickleDataset`, `NumpyDictDataset`, `PandasCsvDataset`, `PandasExcelDataset`, `SimulationDataset` (`bins/base.py`, `repository/dataset.py`, `envs/tasks/base.py::make_dataset`). Propose the survivor of each side and the call-site changes, as text.
8. **Network / processor / generators.** `network/{euclidean,file}.py` + `haversine_distance` in `network/__init__.py` (still exported — used?); `processor/*` (1,565 LOC: which functions does `gen_data`/`test_sim` reach?); `generators/{builders,datasets,validators,waste}.py` dataset types other than `test_simulator`/`train` (`train_time`?).
9. `constants/*`: values referenced by nothing (`MAX_LENGTHS`, `LOSS_KEYS`, `METRICS` vs `SIM_METRICS`, `user_interface.py`, `system.py`).

Post claims and findings on the bus under `### Cursor — 2026-09-25 (...)`.
