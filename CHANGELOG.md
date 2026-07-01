# Changelog

All notable changes to WSmart-Route are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### WSmart-Route Studio — Tauri App (`app/`)

Initial scaffold and core implementation of the WSmart-Route Studio desktop app,
a Tauri 2.0 + React 19 replacement for the PySide6 GUI and the Streamlit dashboard.
Implements §G.0, §G.9–§G.12, §G.15 from `docs/moon/ROADMAP.md`.

**Rust backend (`app/src-tauri/`)**
- `src-tauri/src/lib.rs` — plugin registration (notification, store, dialog, shell) and all command handlers
- `src-tauri/src/commands/sim_watcher.rs` — real-time `GUI_DAY_LOG_START:` log line watcher; polls every 200 ms, emits `sim:day_update` Tauri events; replaces Streamlit's `time.sleep()` + `st.rerun()` polling loop
- `src-tauri/src/commands/data.rs` — `load_simulation_log`, `load_csv_file` (returns `CsvFile` with headers+rows), `list_output_dirs` (returns `OutputDir` with metadata), `list_training_runs`, `load_training_metrics`
- `src-tauri/src/commands/process.rs` — `spawn_python_process` (stdout/stderr streamed as `process:stdout` events), `cancel_process` (tokio watch channel), `list_processes`; global `PROCESS_REGISTRY`
- `src-tauri/Cargo.toml` — tauri 2.0, tauri-plugin-{notification,store,dialog,shell}, serde, tokio (full), csv, anyhow
- `src-tauri/tauri.conf.json` — window 1600×1000, min 1200×700
- `src-tauri/capabilities/default.json` — Tauri 2.0 capability grants for all plugins

**React frontend (`app/src/`)**
- `types/index.ts` — `DayLogEntry`, `SimDayData`, `TrainingRun`, `TrainingMetricsRow`, `ProcessEntry`, `ProcessStatus`, `StdoutLine`, `StatusUpdate`, `AppMode`, `NavSection`, `NavItem`
- `store/app.ts` — Zustand with persist: `mode`, `theme` (syncs `dark` class), `projectRoot`
- `store/sim.ts` — `entries`, `selectedPolicy/Sample/Day`, `watchPath`, `isWatching`; `addEntry` deduplicates by `(policy, sample_id, day)`; exports `uniquePolicies`, `uniqueSamples`, `filterEntries`
- `store/process.ts` — `processes` map; `appendLog` caps at 2000 lines per process
- `hooks/useSimWatcher.ts` — subscribes to `sim:day_update`, calls `start_sim_watcher`/`stop_sim_watcher`
- `hooks/useProcessMonitor.ts` — subscribes to `process:stdout` and `process:status` events
- `components/layout/Layout.tsx`, `Sidebar.tsx`, `TopBar.tsx` — 3-section nav (Monitor / Analysis / Launch), running-process count badge, theme toggle
- `components/ui/KpiCard.tsx` — label, value, unit, delta with trend icons, `lowerIsBetter` prop
- `components/ui/StatusPill.tsx` — animated pulse badge for process status
- `index.css` — Tailwind base + component layer (`card`, `kpi-card`, `btn-primary`, `btn-ghost`, `input-base`, `select-base`, `log-line`, `kpi-delta-pos/neg`)

**Pages**
- `pages/SimulationMonitor.tsx` — real-time digital twin; file picker for log, Rust watcher, KPI dashboard (primary + secondary), day-slider, ECharts timeseries; ports `logic/src/ui/pages/simulation/`
- `pages/TrainingMonitor.tsx` — training run discovery, metrics.csv loading, ECharts loss/reward curves; ports `logic/src/ui/pages/training.py`
- `pages/SimulationSummary.tsx` — per-policy aggregate KPIs and bar charts from completed logs
- `pages/BenchmarkAnalysis.tsx` — multi-run, multi-policy comparison with overlaid bar charts
- `pages/DataExplorer.tsx` — paginated CSV table viewer (50 rows/page)
- `pages/ExperimentTracker.tsx` — output directory browser with creation time and size
- `pages/AlgorithmComparison.tsx` — radar chart + per-metric bars comparing all policies in loaded log
- `pages/HPOTracker.tsx` — training run final-reward bar chart; Optuna embedding planned §G.18
- `pages/ProcessMonitor.tsx` — live table of all spawned processes, inline log viewer, cancel button
- `pages/SimulationLauncher.tsx` — Hydra override textarea → `spawn_python_process main.py test_sim`
- `pages/TrainingHub.tsx` — mode selector (train/hpo/eval) + Hydra overrides → `spawn_python_process main.py <mode>`
- `pages/DataGeneration.tsx` — script picker + extra args → `spawn_python_process`

**Config**
- `package.json` — React 19, Tauri 2, ECharts, Zustand 5, react-router-dom 7, sonner, lucide-react
- `vite.config.ts` — Tauri build settings, `VITE_` + `TAURI_` env prefix
- `tsconfig.json` — strict TypeScript, `@/*` path alias
- `tailwind.config.ts` — custom `canvas-*` and `accent-*` color palette, `darkMode: "class"`
- `index.html` — `<html class="dark">` shell

#### ROADMAP

- `docs/moon/ROADMAP.md` — rewrote §D (GUI/UX) for Tauri/React architecture; added §G (WSmart-Route Studio, 16 phases); added §G.16 (Simulation Digital Twin), §G.17 (Training Monitor), §G.18 (Experiment & HPO Tracker)

---

## [0.5.0] — 2026-06-XX

### Added
- Figueira da Foz 350-bin dataset with plastic-bin results using Empirical distribution and Classical Local Search route improver
- City comparison simulation analysis (`global/`)

### Changed
- Analysis CSVs moved to `global/` directory; markdown references updated

---

## [0.4.0] — earlier

*(Earlier history not yet documented in this changelog)*

---

[Unreleased]: https://github.com/ACFHarbinger/WSmart-Route/compare/HEAD...HEAD
