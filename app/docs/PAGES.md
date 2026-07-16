# Studio Pages

Navigation is grouped in the sidebar into five sections. Each page corresponds
to an `AppMode` value (deep-linkable via `#m=<mode>`).

## Monitor

| Page | Mode | Description |
| ---- | ---- | ----------- |
| Simulation Digital Twin | `simulation` | Live multi-day simulation monitoring from `.jsonl` day logs: geospatial deck.gl routes, KPI strip, day scrubber; the Rust sim watcher streams new days in real time |
| Training Monitor | `training` | Training-run dashboards: metric sparklines, loss curves, training health (gradient norms, LR), checkpoint browser |
| Process Monitor | `process_monitor` | All Python subprocesses spawned from the Studio: status pills, structured progress, log tails, cancellation |

## Analysis

| Page | Mode | Description |
| ---- | ---- | ----------- |
| Simulation Summary | `simulation_summary` | Aggregate KPIs of a finished simulation run, policy telemetry panels and trends |
| Benchmark Analysis | `benchmark` | Cross-policy benchmark: Pareto panels, distribution/graph/portfolio heatmaps, parallel coordinates, efficiency ranking |
| City Comparison | `city_comparison` | Side-by-side per-city comparison of policies and network statistics |
| OLAP Explorer | `olap_explorer` | DuckDB-Wasm data cube: SQL panel, pivot tables, auto-charting of query results |
| Data Explorer | `data_explorer` | Dataset/NPZ inspection: distribution stats, histograms/KDE, tensor slices |
| Experiment Tracker | `experiment_tracker` | MLflow runs and metric histories |
| Algorithm Registry | `algorithms` | Registered policies/solvers with metadata and comparison views |
| HPO Tracker | `hpo_tracker` | Optuna studies: trial tables, importance, history/parallel plots |

## Launch

| Page | Mode | Description |
| ---- | ---- | ----------- |
| Simulation Launcher | `sim_launcher` | Configure and launch `main.py test_sim` runs (policies, cities, horizon, …) |
| Training & HPO Hub | `training_hub` | Launch training (`train_lightning`, meta-RL) and HPO sweeps with live panels |
| Data Generation | `data_gen` | Launch dataset generation with persisted forms and completion toasts |
| Evaluation Runner | `eval_runner` | Launch checkpoint evaluation and inspect result cards/KPIs |
| Report Studio | `report_studio` | The §H analysis & presentation engine: generate the simulation/dataset reports, the PPTX deck, HTML/PDF exports — natively in-app (default) or via the archived Python scripts (Legacy toggle); includes scrolling report preview and paginated deck preview |

## Files

| Page | Mode | Description |
| ---- | ---- | ----------- |
| Output Browser | `output_browser` | Browse `assets/output` run trees, open artefacts, hand paths off to analysis pages |
| Config Editor | `config_editor` | Monaco YAML editor for Hydra configs with validation |
| System Tools | `system_tools` | File-system tools (update/delete/cryptography) and the program test-suite runner |

## App

| Page | Mode | Description |
| ---- | ---- | ----------- |
| Settings | `settings` | Project root, theme, Python probing, updater, session profiles |

## Global UX

- **Command palette** (Ctrl+K) — fuzzy navigation and commands
- **Keyboard shortcuts** (see `?` / Ctrl+Shift+/) — single-key page jumps (p, m, q, …), Ctrl+, settings, Ctrl+R launch
- **Global filter bar** — policy / sample / run-label / city brushes and the log-scale toggle, propagated across analytics views and encoded in the URL hash
- **Onboarding dialog & guided tour** — first-run project-root setup
- **Toasts** (sonner) for background completions, with artefact handoff buttons
