# Changelog

All notable changes to WSmart-Route are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### WSmart-Route Studio — Tauri App (`app/`) — eighth pass

Eighth implementation pass: LR schedule sparkline in Training Monitor (§G.17); completion
navigation in Training Hub (§G.10); eval results grid with CSV export in Evaluation Runner (§G.12).

**React frontend**
- `pages/monitor/TrainingMonitor.tsx` — LR schedule sparkline (§G.17):
  - Refactored `GradNormSparkline` and new `LrSparkline` to share a `MetricSparkline` base component (avoids duplication; same ECharts config parameterised by `label`, `data`, `color`)
  - `LrSparkline` plots `lr` vs `step` in amber (`#fbbf24`); shown per selected run below the grad-norm sparkline
- `pages/launch/TrainingHub.tsx` — completion navigation (§G.10):
  - "Output Browser →" button appears in live progress header when `runStatus === "completed"`; navigates to `output_browser` mode so users can inspect checkpoints immediately after training
- `pages/launch/EvaluationRunner.tsx` — results grid (§G.12):
  - `EvalResult` interface; `EVAL_RESULT_KEYS` sentinel list (`cost`, `gap`, `tour_cost`, `obj`, `time`, `policy`, `checkpoint`)
  - `processToCheckpoint` ref: maps process ID → checkpoint filename; populated at launch, used by the global `process:stdout` listener to attribute result rows
  - `ResultsGrid` component: dynamic columns from first result; numeric values formatted to 4 dp; updates live as rows arrive; replaces static placeholder card
  - "Export CSV" button in `ResultsGrid`: builds CSV string from all result rows, triggers `<a>` download via `Blob` + `URL.createObjectURL`

**ROADMAP**
- `docs/moon/ROADMAP.md` — §G.17 LR sparkline checked; §G.10 completion navigation checked; §G.12 results grid + CSV export checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventh pass

Seventh implementation pass: TrainingHub live progress chart (§G.10); OutputBrowser run
metadata panel + Sim Summary handoff (§G.14); Settings import/export JSON (§G.19); global
keyboard shortcuts (§G.7); pages directory reorganised into five subdirectories.

**React frontend**
- `pages/launch/TrainingHub.tsx` — live progress panel (§G.10):
  - `parseMetricLine`: tries JSON parse first; falls back to `key=value` scanning; detects rows with `train_loss`, `val_loss`, `reward`, `grad_norm`, `epoch`, or `step` keys
  - `LiveChart` component: ECharts canvas with train_loss (solid indigo), val_loss (dashed green), reward (dotted amber, right y-axis); shown once ≥ 2 metric rows received
  - Live snapshot row: epoch / train_loss / val_loss / reward / ‖∇‖ inline below chart
  - "Process Monitor" navigation button; `CheckCircle`/`XCircle` status header on completion
- `pages/files/OutputBrowser.tsx` — enhancements (§G.14):
  - Run metadata card: on `selectRun` auto-loads `pruned_config.yaml` / `config.yaml`; parses flat key-value pairs filtered by `META_KEYS` (task, seed, envs, area, policies, …); shown below the file tree as a compact two-column card
  - "Open in Sim Summary" button: shown for `.jsonl` files after loading; sets `store.pendingLogPath` + navigates to `simulation_summary`
- `pages/analysis/SimulationSummary.tsx` — consumes `pendingLogPath` via `useEffect` on mount; calls `loadLog` (extracted from button handler) and clears the store field
- `pages/app/Settings.tsx` — Backup & Restore card (§G.19):
  - "Export Settings": opens `save` dialog, serialises `{projectRoot, pythonPath, theme}` to JSON via `write_text_file`
  - "Import Settings": opens file picker, parses JSON, populates draft fields for review before saving
- `store/app.ts` — `pendingLogPath: string | null` + `setPendingLogPath` action (ephemeral, not persisted)
- `App.tsx` — global keyboard shortcuts (§G.7):
  - `Ctrl+,` → `settings`; `Ctrl+Shift+P` → `process_monitor`
  - Digit `1`–`8` (when no input focused): quick-switch to simulation / simulation_summary / training / benchmark / sim_launcher / training_hub / process_monitor / settings

**Project structure**
- `app/src/pages/` reorganised into five subdirectories mirroring sidebar sections:
  - `monitor/` — SimulationMonitor, TrainingMonitor, ProcessMonitor
  - `analysis/` — SimulationSummary, BenchmarkAnalysis, DataExplorer, ExperimentTracker, AlgorithmComparison, HPOTracker
  - `launch/` — SimulationLauncher, TrainingHub, DataGeneration, EvaluationRunner
  - `files/` — ConfigEditor, OutputBrowser
  - `app/` — Settings
- All intra-page imports updated from `../` to `../../`; `App.tsx` import paths updated to `pages/<subdir>/`

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.10 live training progress checked; §G.14 metadata panel and Open in Sim Summary checked; §G.19 import/export checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixth pass

Sixth implementation pass: SimulationLauncher gains a live-status panel (§G.9); ConfigEditor gains a
Save button (§G.13); ProcessMonitor gains structured JSON log rendering and persistent history (§G.15);
SimulationSummary is rewritten with a ranking table, per-day trajectory chart, and four metric charts.

**Rust backend**
- `commands/data.rs` — `write_text_file(path, content)`: writes (or overwrites) any text file; creates parent directories; used by ConfigEditor Save button; registered in `lib.rs`

**React frontend**
- `pages/SimulationLauncher.tsx` — live-status panel (§G.9):
  - After launch, subscribes to `process:stdout` Tauri events filtered by the spawned process ID
  - Parses `GUI_DAY_LOG_START:` markers (same protocol as `sim_watcher.rs`) to extract `DayLogEntry` JSON
  - Displays a per-policy card grid with latest day / profit / km / overflows in real time
  - Status header: animated `Activity` icon while running; `CheckCircle`/`XCircle` on completion
  - "View Summary →" button navigates to `simulation_summary` mode; "Process Monitor" button to `process_monitor`
- `pages/ConfigEditor.tsx` — Save button (§G.13):
  - Calls `write_text_file` Tauri command with the currently open path and textarea content
  - Tracks dirty state via `savedContentRef` (updates on open + save); button label shows `Save*` when unsaved edits exist; disabled when no changes
  - `Save` icon from lucide-react; spinner shown during write
- `pages/SimulationSummary.tsx` — full rewrite:
  - `RankingTable` component: sortable by any of 4 metrics (profit / km / overflows / kg); click column header to sort ascending/descending; shows mean ± std in `font-mono`; coloured policy dot + rank number
  - `TrajectoryChart` component: single ECharts line chart overlaying all policies across simulation days (mean per day, averaged across samples); metric selector tabs (Overflows / Profit / Distance / Waste); 8-colour palette
  - `MetricBarChart` component: per-metric bar chart with std dev exposed in tooltip hover
  - `aggregateByPolicyAndDay` helper for trajectory data: groups entries by `(policy, day)`, averages across samples
  - `std()` helper function
- `pages/ProcessMonitor.tsx` — improvements (§G.15):
  - `LogLine` component: attempts `JSON.parse` on each log line; if the result has `level`/`levelname`/`severity` and `msg`/`message`/`text` fields, renders timestamp prefix + colour-coded level badge (danger/warning/muted/default) + message body; falls back to raw string otherwise
  - Per-row `Trash2` "Remove" button for completed processes
  - "Clear completed (N)" header button calls `clearCompleted` store action
- `store/process.ts` — persistence and bulk-clear (§G.15):
  - Wrapped `create` in `persist` middleware; `partialize` strips `logLines` and retains only the last 50 non-running processes; stored under key `"wsmart-studio-processes"`
  - `clearCompleted()` action: removes all entries with `status !== "running"` from the map

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.9 live-status item checked; §G.13 `write_text_file` and Save button checked; §G.15 structured log parsing, remove/clear buttons, and history persistence checked; §G.16 Simulation Summary rewrite noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifth pass

Fifth implementation pass: SimulationMonitor gains day-scrubber controls, a bin-fill strip chart,
and a tour sequence table (§G.16); TrainingMonitor gains multi-run overlay chart, hyperparameter
panel, gradient norm sparkline, and checkpoint browser with one-click Eval Runner handoff (§G.17).

**React frontend**
- `pages/SimulationMonitor.tsx` — rewritten:
  - Day scrubber: `◀`/`▶` step buttons flanking the range input; "Following" badge (green pulse, shown when `selectedDay` is null and watcher active); "Latest ↓" button releases back to auto-follow
  - `BinFillStrip` component: top-25 bins sorted by fill %, 0-100% horizontal bars (green <80%, amber ≥80%, red ≥100%), mandatory (!) and collected (✓) badges; show/hide toggle
  - `TourTable` component: stop #, bin ID, fill %, collected, mandatory columns; reads `tour_indices` preferentially; capped at 60 rows; show/hide toggle
- `pages/TrainingMonitor.tsx` — rewritten:
  - `MultiRunChart`: single ECharts canvas overlaying all selected runs; 8-colour palette; solid train loss, dashed val loss, dotted reward (right y-axis); scrollable legend
  - `GradNormSparkline`: compact `grad_norm` chart per run
  - `HparamsPanel`: collapsible; reads `hparams.yaml` via `read_text_file`; flat YAML parser; 8-row preview with "Show all" expand
  - `CheckpointBrowser`: `list_dir` on `<run>/checkpoints/`; "Load in Eval Runner →" button sets `pendingCheckpoint` in app store and navigates to Eval Runner
  - `RunPanel`: groups grad norm + hparams + checkpoints per run below the shared overlay chart
- `pages/EvaluationRunner.tsx` — reads `pendingCheckpoint` on mount via `useEffect`; pre-populates first checkpoint entry and clears the store field
- `store/app.ts` — `pendingCheckpoint: string | null` + `setPendingCheckpoint` action (not persisted)

#### Build tooling

- `tools/app/justfile` — added `bundle` (list installer output), `logs-dir` (print platform data dir), `reset-data` (delete Tauri Store files)
- Root `justfile` — added `studio-reset` shorthand (→ `app::reset-data`)

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.16 items checked (bin-fill, tour table, day scrubber, secondary KPI toggle); §G.17 items checked (multi-run overlay, grad norm, hparams panel, checkpoint browser)

---

#### WSmart-Route Studio — Tauri App (`app/`) — fourth pass

Fourth implementation pass: Evaluation Runner page (§G.12), full DataGeneration form (§G.11),
Settings validation with Rust backend probes, `tools/app/justfile` Clippy/outdated recipes,
and `studio-check`/`studio-clippy` root shorthands.

**Rust backend**
- `commands/system.rs` — new module with two commands:
  - `validate_project_root(path)`: checks path exists, is a directory, and contains `main.py`
  - `probe_python(python_path)`: runs `<path> --version`, handles Python 2 (stderr) and 3 (stdout), returns version string
- `lib.rs` — registers `validate_project_root` and `probe_python`; imports `system` module

**React frontend**
- `pages/EvaluationRunner.tsx` — new: dynamic checkpoint list (add/remove/file-picker), dataset path picker, problem/strategy/device/val_size selects, multi-checkpoint launch (one process per checkpoint, tagged by filename), Advanced Overrides, command preview, results placeholder (§G.12)
- `pages/DataGeneration.tsx` — rewritten: problem selector, distribution checkboxes (Gamma-3/Empirical), dataset type selector, overwrite toggle, graph form (area/num_loc/n_samples/n_days), Advanced Overrides, command preview; Hydra args mirror `gen_data.yaml` (§G.11)
- `pages/Settings.tsx` — validation wiring: `onBlur` and pre-save calls to `validate_project_root` and `probe_python`; inline `CheckCircle`/`XCircle` badges; save blocked on validation errors
- `types/index.ts` — `"eval_runner"` added to `AppMode` union
- `components/layout/Sidebar.tsx` — `"Evaluation Runner"` entry added to Launch section; `ClipboardList` icon
- `components/layout/TopBar.tsx` — `"Evaluation Runner"` title added to TITLES map
- `App.tsx` — `EvaluationRunner` import and router case added

#### Build tooling

- `tools/app/justfile` — added `clippy` (`cargo clippy -- -D warnings`) and `outdated` (`npm outdated`) recipes
- Root `justfile` — added `studio-check` (→ `app::check`) and `studio-clippy` (→ `app::clippy`) shorthands

#### ROADMAP

- `docs/moon/ROADMAP.md` — §G.11 additional items checked (full form); §G.12 marked 🚧 In Progress with completed items; §G.19 additional items checked (validation commands)

---

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
