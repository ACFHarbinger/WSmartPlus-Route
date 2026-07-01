# Changelog

All notable changes to WSmart-Route are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### WSmart-Route Studio — Tauri App (`app/`) — third pass

Third implementation pass: full-featured Simulation Launcher and Training Hub forms, tabular
Process Monitor with live duration, Settings page (§G.19) with project root / Python path
persistence, first-run onboarding banner, and extended `tools/app/justfile`.

**Rust backend**
- `process::spawn_python_process`: new `python_executable: Option<String>` parameter; empty string treated as `None`, falling back to `which_python`
- `process::which_python`: now takes `working_dir` parameter; checks `<workingDir>/.venv/bin/python` (uv-managed venv) and `<workingDir>/.venv/Scripts/python.exe` (Windows) before system PATH

**React frontend**
- `pages/SimulationLauncher.tsx` — rewritten: 8-policy multi-select checkboxes; area / num_loc / n_samples / cpu_cores / seed inputs; distribution radio (Normal/Gamma/Empirical); Advanced Overrides collapsible; `useMemo` command preview; Hydra args exactly mirror `just controller::test-sim`
- `pages/TrainingHub.tsx` — rewritten: mode selector (Train / HPO Sweep / Evaluate); problem/model/encoder selects; mode-specific param groups (epochs/batch_size for train; method/trials/workers for HPO; checkpoint picker / dataset picker / strategy / val_size for eval); WandB toggle; command preview
- `pages/ProcessMonitor.tsx` — rewritten: tabular `ProcessRow` components with `StatusPill`, process ID, command, PID, live duration (`useLiveDuration` 1s tick), exit code; expand/collapse inline log with auto-scroll toggle; stderr lines coloured warning
- `pages/Settings.tsx` — new: Project Root (text input + directory picker), Python Executable (override `which_python`), Appearance (dark/light radio), About section; dirty-state detection; Save / Discard buttons
- `store/app.ts` — `pythonPath` field + `setPythonPath` action added; persisted via `partialize`
- `types/index.ts` — `"settings"` added to `AppMode` union
- `hooks/useSpawnProcess.ts` — reads `pythonPath` from app store; passes `pythonExecutable: pythonPath || null` to `spawn_python_process`
- `components/layout/Sidebar.tsx` — "App" section added with Settings entry; `FolderOpen` icon for output_browser; `Settings` icon for settings entry
- `components/layout/TopBar.tsx` — first-run warning banner: shown when `projectRoot` is empty and mode ≠ `"settings"`; "Open Settings" quick-link
- `App.tsx` — Settings page import and router case added

#### Build tooling

- `tools/app/justfile` — extended with `check-rust` (`cargo check`), `fmt-rust` (`cargo fmt`), `preview` (build + serve), `update` (`npm update`) recipes

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.9 additional items checked (full form); §G.10 additional items checked (full form, all three modes); §G.15 additional items checked (tabular layout, live duration); §G.19 added (Settings & First-Run Onboarding); Effort × Impact matrix updated

---

#### WSmart-Route Studio — Tauri App (`app/`) — second pass

Second implementation pass: completes all page stubs, wires process lifecycle events, adds
Config Editor (§G.13) and Output Browser (§G.14), and introduces `tools/app/justfile`.

**Rust backend additions**
- `data::read_text_file` — reads any text file (YAML, JSON, plain text) as a `String`; used by ConfigEditor and OutputBrowser
- `data::list_dir` — lists files and subdirectories in a path; returns `DirEntry` with `name`, `path`, `is_dir`, `size_bytes`, `extension`
- `process::ProcessSpawned` event — emitted immediately when a process is spawned (before any stdout); frontend registers the process in the store automatically via `useProcessMonitor`
- `process::which_python` — now resolves `<workingDir>/.venv/bin/python` first (uv-managed project venv), then `.venv/Scripts/python.exe` (Windows), then system PATH

**React frontend additions**
- `hooks/useSpawnProcess.ts` — wraps `spawn_python_process` invoke with loading state and `sonner` toasts; used by all three launcher pages
- `hooks/useProcessMonitor.ts` — now subscribes to `process:spawn` (new) in addition to `process:stdout` and `process:status`; process is registered in the store on spawn, not on first stdout line
- `pages/ConfigEditor.tsx` — Raw / Table / Diff view modes for any YAML/TOML config file; flat YAML parser; "Copy Overrides" button via `navigator.clipboard`; Diff view highlights changed keys between two files (e.g. `pruned_config.yaml` from two runs)
- `pages/OutputBrowser.tsx` — three-pane layout: run list (`list_output_dirs`), file tree (`list_dir`, lazy-loads subdirs), file viewer (CSV table up to 200 rows; raw text for YAML/JSON/log); arbitrary directory picker via Tauri dialog
- `components/layout/Sidebar.tsx` — added "Files" section with Output Browser and Config Editor entries
- `types/index.ts` — added `ProcessSpawned`, `DirEntry`, `OutputDir` interfaces
- Updated `SimulationLauncher`, `TrainingHub`, `DataGeneration` to use `useSpawnProcess` (removes direct `invoke` calls and manual state management)

#### Build tooling

- `tools/app/justfile` — new just module with `install`, `dev`, `tauri-dev`, `build`, `check`, `clean-js`, `clean-rust`, `clean` recipes
- Root `justfile` — added `mod app 'tools/app'` and shorthands: `just studio` (→ `app::tauri-dev`), `just studio-build` (→ `app::build`), `just studio-install` (→ `app::install`)
- `tools/helper/justfile` — updated help text to list `app` module and `just studio` shorthand

#### ROADMAP

- `docs/moon/ROADMAP.md` — marked §G.0, §G.9–§G.15 as 🚧 In Progress with completed items checked; remaining items clearly separated

---

#### WSmart-Route Studio — Tauri App (`app/`) — initial scaffold

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
