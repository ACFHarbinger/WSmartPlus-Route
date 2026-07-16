# Studio Architecture

## Process model

The Studio is a Tauri 2.0 application:

- **Webview frontend** — React 19 + TypeScript + Vite + Tailwind, ECharts for
  2-D charts, deck.gl/MapLibre for geospatial views, Sigma.js/graphology for
  topology, React Three Fiber for 3-D, DuckDB-Wasm for in-browser OLAP,
  Monaco for editors, Zustand for state.
- **Rust core** (`src-tauri/`) — filesystem access, NPZ/CSV/Arrow loaders,
  Python subprocess management (spawning `main.py` entry points), MLflow /
  Optuna / ZenML readers, the simulation log watcher, bundle import/export,
  and the updater.

The frontend never touches the filesystem or network directly; everything goes
through `invoke()` commands or Tauri plugins (dialog, shell, store, notification,
updater).

## Frontend source layout (`src/`)

```
src/
├── App.tsx              # mode-based page switch, global keyboard shortcuts
├── main.tsx             # React root
├── components/
│   ├── analysis/
│   │   ├── benchmark/   # heatmaps, Pareto panels, portfolio parallel/ranking
│   │   ├── topology/    # Sigma/Cosmograph topology + attention views
│   │   ├── telemetry/   # policy telemetry panels, strategy legend
│   │   ├── routes/      # route visualiser, failure analysis overlays
│   │   ├── explorer/    # SQL query + pivot table panels
│   │   └── training/    # loss landscape 3D, ML introspection, training health
│   ├── common/          # export buttons, path/run-label chips & toolbars
│   ├── editors/         # YAML (Monaco) editor
│   ├── gen/             # deck + report previews for Report Studio
│   ├── layout/          # sidebar, top bar, command palette, tour, filter bar
│   ├── maps/            # deck.gl route map
│   ├── monitor/
│   │   ├── live/        # launcher/train-HPO live panels, sparklines, progress
│   │   ├── eval/        # evaluation result/KPI cards
│   │   └── process/     # process id footer, log tail
│   └── ui/              # primitives (KpiCard, StatusPill)
├── constants/           # Python CLI command definitions
├── gen/                 # native §H report/deck generation engine (see GEN_PIPELINE.md)
├── hooks/
│   ├── app/             # useDuckDbInit, useHashSync, useStartupTiming, useThemeSync
│   ├── brush/           # run-label brush hooks (tables, logs, portfolio, processes)
│   ├── files/           # file drop, recent-file handoff, .wsroute import
│   └── process/         # useProcessMonitor, useSpawnProcess, useSimWatcher
├── pages/               # one component per Studio mode (analysis/app/files/launch/monitor)
├── store/               # Zustand stores (app, layout, filters, process, sim, …)
├── types/               # shared TypeScript types
└── utils/
    ├── app/             # page prefetch, startup timing
    ├── benchmark/       # eval results, Pareto, portfolio, telemetry, comparisons
    ├── charts/          # ECharts helpers: export, highlight, log-scale, symlog, colors, theme
    ├── duckdb/          # DuckDB-Wasm client, SQL templates, Arrow pipeline, pivots
    ├── graph/           # attention/topology graph builders, KMeans, loss landscape
    ├── map/             # map positions, route viz, failure overlays, vehicle tours
    ├── process/         # launcher process helpers, log parsing, progress markers
    ├── runs/            # output run paths, run logs, recent handoff
    ├── sim/             # day-log parsing, sim metadata, failure extraction
    └── training/        # checkpoints, training metrics/health, run paths
```

### Navigation & deep-links

There is no router: `store/app.ts` holds an `AppMode` string and `App.tsx`
renders the matching page (lazy-loaded). `hooks/app/useHashSync.ts` mirrors the
mode plus the global filters into the URL hash (`#m=<mode>&p=<policy>&l=1…`) so
views are bookmarkable, and restores them on startup / hash change.

### State stores (`src/store/`)

| Store | Purpose |
| ----- | ------- |
| `app` | mode, project root, theme, persisted app-level settings |
| `layout` | sidebar/onboarding/tour/command-palette UI state |
| `filters` | global policy/sample/run-label/city brushes + log-scale toggle |
| `process` | spawned Python process registry (stdout ring buffers, status) |
| `sim` | simulation watcher state (day logs streamed from Rust) |
| `launchers`, `launchTrigger` | persisted launcher forms and cross-page launch triggers |
| `duckdb` | DuckDB-Wasm session state |
| `recentFiles`, `sessionProfiles` | recent artefact handoff, saved session profiles |

Stores that must survive restarts persist via `zustand/persist` to
`localStorage`.

## Rust command surface (`src-tauri/src/commands/`)

Grouped by concern (each is a `#[tauri::command]` invoked from TS):

- **Filesystem**: `list_dir`, `list_files_recursive`, `list_output_dirs`,
  `path_exists`, `read_text_file` / `write_text_file`, `read_binary_file` /
  `write_binary_file`
- **Data loading**: `load_csv_file`, `load_npz_flat`, `load_npz_vectors`,
  `inspect_npz_archive`, `probe_npy_mmap`, `load_tensor_slice`,
  `preview_dataset_stats`
- **Arrow / DuckDB pipeline**: `csv_to_arrow_ipc`, `simulation_log_to_arrow_ipc`,
  `tensor_slice_to_arrow_ipc`, `export_csv_to_parquet`, `export_table_parquet`,
  `benchmark_arrow_pipeline`
- **Simulation & logs**: `load_simulation_log`, `start_sim_watcher` /
  `stop_sim_watcher`, `load_sim_failure_log`, `list_sim_policies`,
  `load_policy_viz_log`, `load_policy_telemetry_trends`,
  `load_policy_trajectory_trends`, `load_attention_viz_log`
- **Training / experiment tracking**: `load_training_metrics`,
  `load_training_health_log`, `list_training_runs`, `list_mlflow_runs`,
  `list_mlflow_metric_keys`, `load_mlflow_metric_history`,
  `list_optuna_studies`, `load_optuna_study`, `export_optuna_reports`,
  `list_zenml_pipeline_runs`, `load_zenml_run_steps`
- **Process management**: `spawn_python_process`, `cancel_process`,
  `list_processes`, `probe_python`, `dump_hydra_config`,
  `validate_project_root`
- **Bundles & app lifecycle**: `create_wsroute_bundle`, `extract_wsroute_bundle`,
  `inspect_wsroute_bundle`, `get_app_version`, `check_for_updates`,
  `install_app_update`

### IPC conventions

- Long-running Python work is **spawned**, not awaited: `spawn_python_process`
  emits `process-output` / `process-exit` events consumed by
  `hooks/process/useProcessMonitor`, which feeds the process store.
- Bulk tabular data crosses the boundary as **Arrow IPC bytes** and is
  registered directly with DuckDB-Wasm (§G.6) instead of JSON.
- Python subprocesses emit structured markers on stdout (`PROGRESS:{json}`,
  `GUI_…` log emitters) that `utils/process` parses for progress bars and
  artefact path handoff.
