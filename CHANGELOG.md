# Changelog

All notable changes to WSmart-Route are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-sixth pass (§G.3 / §G.6)

Thirty-sixth pass adds **multi-vehicle route rendering**, **DuckDB SQL explorer**,
**graph-split benchmark heatmaps**, and **§G.2 drill-down error bars**.

**React frontend**
- `utils/vehicleTours.ts` — split depot-delimited tours into per-vehicle segments (ColorBrewer palette)
- `DeckRouteMap` — distinct PathLayer/TripsLayer per vehicle; legend chips per vehicle
- `SimulationMonitor` `RouteMapChart` — multi-vehicle colored route lines on Cartesian map
- `components/analysis/SqlQueryPanel.tsx` — Monaco SQL editor + templates + sortable result grid + CSV export
- `utils/duckdbTemplates.ts` — robustness, variance, Pareto candidate query templates
- `DataExplorer` — DuckDB SQL panel when CSV ingested into Wasm worker
- `BenchmarkAnalysis` — graph-facet heatmaps (RM-100 / RM-170 / FFZ-350) with overflows/kg/km toggle
- `policyHierarchy.ts` — drill-down profit std + Empirical↔Gamma spread for error-bar whiskers
- `SimulationSummary` — drill-down bars show distribution variance when error bars enabled

**ROADMAP**
- §G.3.2 multi-vehicle rendering checked (partial — per-vehicle stop colors deferred)
- §G.1.3 graph-split heatmaps, §G.2 drill-down error bars checked (partial)
- §G.6 DuckDB query editor, templates, result grid + CSV export checked (partial)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-fifth pass (§G.1 / §G.2)

Thirty-fifth pass adds **§G.2 hierarchical drill-down** on Simulation Summary and
closes more **§G.1** multi-run / distribution-split items on Benchmark Analysis.

**React frontend**
- `utils/policyHierarchy.ts` — build sunburst/treemap tree (city → strategy → constructor); breadcrumb path helpers
- `utils/paretoPanels.ts` — classify runs into Gamma-3/FTSP · Empirical/FTSP · Gamma-3/CLS · Empirical/CLS panels
- `utils/simMetadata.ts` — shared `strategyColor`, `citySymbol`, `cityScaleLabel` helpers
- `SimulationSummary` — `PolicyHierarchyPanel` sunburst/treemap toggle; drill-down bar chart + breadcrumb trail
- `SimulationSummary` — `DistributionFacetHeatmaps` splits heatmaps when multiple distributions present
- `SimulationSummary` — zero-overflow corridor slider cross-filters parallel coords + all brushed panels
- `SimulationSummary` — parallel polylines colored by selection strategy
- `BenchmarkAnalysis` — 4-panel Pareto grid from loaded runs; City Comparison log-scale bar chart (§G.1.6)

**ROADMAP**
- §G.2 sunburst/treemap, drill-down bars, breadcrumb checked (partial — DuckDB filter deferred)
- §G.1.2 four-panel Pareto, §G.1.3 distribution heatmap split, §G.1.4 strategy colors + overflow corridor,
  §G.1.6 city comparison log scale checked (partial)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-fourth pass (§G.1)

Thirty-fourth pass implements **§G.1 Statistical Overview Dashboard** cross-filter
brushing, grouped KPI charts, parallel coordinates, and richer policy metadata tooltips.

**React frontend**
- `utils/simMetadata.ts` — parse log paths and policy labels into city/scale/distribution/strategy metadata
- `utils/chartHighlight.ts` — `isHighlighted`, `barOpacity`, `toggleBrush` for dashboard cross-filtering
- `SimulationSummary` — `ConfigMetaBanner` run-config strip; `PolicyBrushBar` chip cross-filter
- `GroupedMetricBarChart` — overflows by selection strategy; kg/km by constructor (mean ± std)
- `PolicyParallelChart` — ECharts parallel coordinates (profit · kg/km · overflows · km)
- `PolicyHeatmapChart` — metric mode toggle (all / overflows / kg/km); brush dimming
- `PolicyParetoChart` — strategy color + city/scale marker shape encoding; brush dimming
- `EfficiencyRankingChart` / `MetricBarChart` — bar opacity by brush; click-to-filter; rich tooltips
- §G.1.6 — auto log-scale duplicate row for profit and overflows when global log toggle is off

**ROADMAP**
- §G.1 grouped KPI bars, interactive brushing, parallel coords (partial), heatmap metric toggle,
  Pareto color/shape encoding, rich tooltips, secondary log-scale views checked (partial)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-third pass (Phase 0)

Thirty-third pass completes **§G.0 Phase 0** foundation items deferred by later-phase
work: Arrow IPC serialization, DuckDB-Wasm worker, and end-to-end latency probe.

**Rust backend (`app/src-tauri/`)**
- `commands/arrow.rs` — CSV + simulation JSONL → Arrow IPC file; `read_binary_file` for zero-copy handoff
- Simulation Arrow schema: policy, sample_id, day, profit, km, overflows, kg, kg_per_km, cost, ncol, kg_lost
- `benchmark_arrow_pipeline` command for Rust-side timing

**React frontend**
- `@duckdb/duckdb-wasm` + `apache-arrow` dependencies; `duckdb` vendor chunk in Vite
- `duckdbClient.ts` — DuckDB-Wasm worker singleton; `insertArrowFromIPCStream` table registration
- `arrowPipeline.ts` — CSV/log → Rust → Arrow → DuckDB orchestration with 500 ms budget
- `useDuckDbInit` — spawns worker on app mount; startup timing milestone `duckdbReady`
- `Settings` — Phase 0 pipeline panel + "Run Arrow Pipeline Benchmark" button
- `DataExplorer` — auto-ingests opened CSV into DuckDB; shows row count + latency
- `SimulationMonitor` — auto-ingests opened simulation log into DuckDB

**ROADMAP**
- §G.0 Arrow IPC + DuckDB-Wasm worker + latency benchmark checked (Phase 0 complete)

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-second pass

Thirty-second implementation pass: policy heatmap (§G.1); deck.gl 3D pitch;
AlgorithmComparison log scale; filtered CSV export.

**React frontend**
- `SimulationSummary` — policy × metric heatmap with normalised scores and PNG export
- `DeckRouteMap` — 3D pitch toggle (0°/45°); controlled pan/zoom view state
- `AlgorithmComparison` — log-scale toggle on per-metric bar charts
- `DataExplorer` — CSV export respects active filter/sort (exports visible subset)

**ROADMAP**
- §G.1 policy configuration heatmap checked (partial — multi-config/multi-city deferred)
- §G.3.1 deck.gl 3D pitch toggle checked (partial — OrbitView deferred)
- §G.1 AlgorithmComparison log-scale toggle noted
- §G.6 Data Explorer filtered CSV export noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirty-first pass

Thirty-first implementation pass: graph preset auto-detect (§G.3.1); symlog
overflows; Data Explorer filter; deck.gl fill-scaled nodes; benchmark log scale.

**React frontend**
- `utils/graphCoords.ts` — `guessGraphPreset()` infers RM-100/RM-170/FFZ-350 from log path or bin count
- `utils/symlog.ts` — symmetric log transform for near-zero overflow values
- `SimulationMonitor` — auto-selects graph preset on log load; shows "auto-detected" badge
- `SimulationSummary` — symlog overflows bar when log scale on; efficiency ranking error bars
- `DataExplorer` — row filter search across all columns with match count
- `DeckRouteMap` — tour-stop scatter radius scales with fill level
- `BenchmarkAnalysis` — log-scale toggle on multi-run comparison bar charts

**ROADMAP**
- §G.3.1 graph preset auto-detect from log path/bin count checked (partial)
- §G.1 symlog overflows bar + efficiency ranking error bars checked (partial)
- §G.6 Data Explorer row filter checked (partial)
- §G.16 deck.gl node radius ∝ fill level checked (partial — profit deferred)
- §G.1 BenchmarkAnalysis log-scale toggle noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirtieth pass

Thirtieth implementation pass: graph JSON coordinate loader (§G.3.1); Pareto
log-scale; BenchmarkAnalysis efficiency rank; Evaluation Runner charts (§G.12).

**React frontend**
- `utils/graphCoords.ts` — load RM-100/RM-170/FFZ-350 coordinates from graph JSON + area CSV via project root
- `SimulationMonitor` — graph preset selector + "Load graph coords" enriches logs for deck.gl tile map
- `SimulationSummary` — log-scale toggle applies to Pareto scatter y-axis (overflows)
- `BenchmarkAnalysis` — horizontal efficiency ranking chart (kg/km) with PNG export
- `EvaluationRunner` — inline cost/gap/time bar charts with PNG export on results grid
- `App.tsx` — maplibre-gl + @deck.gl/react included in startup prefetch batch

**ROADMAP**
- §G.3.1 graph JSON coordinate loader checked (partial — auto-detect from log metadata deferred)
- §G.1 Pareto log-scale y-axis checked (partial — true symlog deferred)
- §G.12 Evaluation Runner inline charts + PNG export checked
- §G.7 maplibre/deck.gl vendor prefetch noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-ninth pass

Twenty-ninth implementation pass: Pareto + efficiency charts (§G.1); Data
Explorer column sort; BenchmarkAnalysis kg/km metric.

**React frontend**
- `SimulationSummary` — horizontal efficiency ranking chart (kg/km); profit vs overflows Pareto scatter with dashed frontier
- `utils/pareto.ts` — Pareto front + step-line helpers for policy comparison
- `BenchmarkAnalysis` — `kg/km` added to simulation comparison metrics
- `DataExplorer` — sortable column headers (asc/desc toggle on click)

**ROADMAP**
- §G.1 Pareto front + horizontal kg/km ranking checked (partial — multi-config deferred)
- §G.6 Data Explorer column sort checked (partial)
- §G.7 BenchmarkAnalysis kg/km export line updated

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-eighth pass

Twenty-eighth implementation pass: launcher/monitor PNG exports (§G.7);
Simulation Summary radar + error bars (§G.1); Data Explorer CSV export.

**React frontend**
- `TrainingMonitor` — PNG export on multi-run overlay chart and grad-norm / LR sparklines
- `TrainingHub` — PNG export on live training chart and grad-norm / entropy sparklines
- `DataGeneration` — PNG export on dataset demand histogram preview
- `SimulationSummary` — policy radar chart; error-bar whiskers toggle on bar charts (linear scale)
- `DataExplorer` — Export CSV button for loaded table data
- `App.tsx` — `DeckRouteMap` chunk included in startup prefetch batch

**ROADMAP**
- §G.1 policy radar + error bars checked (partial — symlog/Pareto deferred)
- §G.7 TrainingMonitor / TrainingHub / DataGeneration PNG export checked
- §G.7 Data Explorer CSV export checked
- §G.10 / §G.11 / §G.17 launcher & monitor PNG export noted

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-seventh pass

Twenty-seventh implementation pass: SimulationSummary chart PNG export (§G.7);
deck.gl tile map PNG capture (§G.16); startup 2s budget indicator (§G.7).

**React frontend**
- `SimulationSummary` — PNG export on per-day trajectory chart and all four policy bar charts
- `DeckRouteMap` — PNG export button captures WebGL canvas via `exportCanvasPng`
- `chartExport.ts` — `exportCanvasPng` helper for deck.gl / canvas screenshots
- `Settings` — prefetch timing shows pass/fail against 2s load budget
- `useStartupTiming` — `withinBudget` flag derived from prefetch milestone
- `App.tsx` — echarts vendor chunk included in startup prefetch batch

**ROADMAP**
- §G.7 SimulationSummary + deck.gl PNG export + 2s budget probe checked (partial — hardware benchmark deferred)
- §G.3.1 ScatterplotLayer fill-coded nodes checked (partial)
- §G.16 deck.gl PNG export checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-sixth pass

Twenty-sixth implementation pass: global file drop (§G.8); extended startup
timing (§G.7); chart PNG exports; guided tour spotlights (§G.19).

**React frontend**
- `hooks/useGlobalFileDrop.ts` — app-wide `.wsroute` extract + `.jsonl` log open from OS file drop
- `utils/startupTiming.ts` — shared startup milestone marks; prefetch-complete timing in Settings
- `GuidedTour` — `data-tour` spotlight rings highlight sidebar, palette, and nav targets per step
- `OnboardingDialog` — auto-offers guided tour after first project-root configuration
- `BenchmarkAnalysis` — PNG export on simulation and eval comparison bar charts
- `AlgorithmComparison` — PNG export on per-metric bar charts (radar already supported)
- `App.tsx` — `Ctrl+Shift+/` opens guided tour; Escape dismisses tour overlay

**ROADMAP**
- §G.3.1 deck.gl + MapLibre integration checked (was implemented, now documented)
- §G.7 prefetch timing probe + BenchmarkAnalysis PNG export checked (partial — <2s target deferred)
- §G.8 global file drop checked
- §G.19 guided tour spotlight + auto-offer checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-fifth pass

Twenty-fifth implementation pass: `.wsroute` drag-drop (§G.14); map compare
deep link (§G.16); ECharts side-by-side split; guided tour (§G.19).

**React frontend**
- `hooks/useFileDrop.ts` — Tauri window `onDragDropEvent` listener for OS file drops
- `OutputBrowser` — drag-drop `.wsroute` onto file viewer; dashed overlay + manifest inspect
- `store/app.ts` — `pendingMapCompare` ephemeral state for Algorithm Comparison → map navigation
- `AlgorithmComparison` — "Compare on Map" sets policy filters + split layout when 2 policies
- `SimulationMonitor` — consumes `pendingMapCompare`; ECharts Cartesian side-by-side when split + 2 policies
- `GuidedTour` — 5-step studio walkthrough; TopBar compass, command palette, Settings entry
- `store/layout.ts` — `guidedTourOpen` / `guidedTourStep` / `guidedTourDismissed` persistence

**ROADMAP**
- §G.14 `.wsroute` drag-drop checked
- §G.16 ECharts side-by-side + map deep link checked
- §G.19 guided tour checked (partial — spotlight deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-fourth pass

Twenty-fourth implementation pass: side-by-side route map compare (§G.16);
Algorithm Comparison map link + PNG export; update check command; startup timing probe.

**Rust backend**
- `system::check_for_updates` — fetches optional `WSMART_UPDATE_URL` JSON manifest; compares `version` field to `CARGO_PKG_VERSION`
- `reqwest` dependency (rustls) for async update manifest fetch

**React frontend**
- `SimulationMonitor` — overlay/split layout toggle on deck.gl tile map when exactly 2 policies visible; split renders dual labelled `DeckRouteMap` panels
- `AlgorithmComparison` — "Compare on Map" navigates to Simulation Monitor; radar chart PNG export via `exportChartPng`
- `hooks/useStartupTiming.ts` — module-load → first-mount timing probe surfaced in Settings About
- `Settings` — startup timing display; "Check for Updates" button wired to `check_for_updates`

**ROADMAP**
- §G.16 side-by-side route compare checked (partial — ECharts Cartesian deferred)
- §G.3.3 algorithm comparison side-by-side map checked (partial)
- §G.7 startup timing probe + AlgorithmComparison PNG export checked (partial — <2s load target deferred)
- §G.8 `check_for_updates` checked (partial — Tauri updater plugin deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-third pass

Twenty-third implementation pass: multi-policy route map overlay (§G.16);
log-scale bar charts (§G.1 partial); app version command; recent-run navigation.

**Rust backend**
- `system::get_app_version` — returns `CARGO_PKG_VERSION` for Settings About panel

**React frontend**
- `DeckRouteMap` — refactored for multi-policy `routes[]` overlay with per-policy colour paths and legend
- `SimulationMonitor` — map policy visibility chips; overlays all policies for the selected day on tile map
- `SimulationSummary` — log-scale toggle on policy ranking bar charts (values clamped to 0.001 for log axis)
- `store/app.ts` — `pendingRunPath` for command-palette recent-run deep link
- `OutputBrowser` — auto-selects run when opened via `pendingRunPath`
- `Settings` — version loaded from Rust; notes auto-update requires release endpoint

**ROADMAP**
- §G.16 multi-policy map overlay + toggle visibility checked (partial — side-by-side deferred)
- §G.1 log-scale toggle on bar charts checked (partial — symlog/Pareto deferred)
- §G.8 `get_app_version` checked (partial — updater plugin deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-second pass

Twenty-second implementation pass: TripsLayer route trail animation and depot
marker (§G.16); recent-files quick open (§G.7 / §G.14); startup route prefetch.

**React frontend**
- `components/maps/DeckRouteMap.tsx` — `TripsLayer` trail animation during day playback; gold depot `ScatterplotLayer`; dimmed idle bins
- `store/recentFiles.ts` — persisted recent logs, runs, and CSVs (max 12)
- `CommandPalette` — Recent section for quick reopen; logs navigate to Simulation Summary
- `SimulationMonitor`, `SimulationSummary`, `OutputBrowser`, `DataExplorer` — track opened files/runs
- `App.tsx` — prefetch simulation, summary, process monitor, and output browser on startup

**Dependencies**
- `@deck.gl/geo-layers` — TripsLayer for animated route trails

**ROADMAP**
- §G.16 TripsLayer animation + depot marker checked (partial — multi-vehicle deferred)
- §G.7 recent files + startup prefetch checked (partial — <2s load target deferred)
- §G.14 recent run/file tracking checked (partial)

---

#### WSmart-Route Studio — Tauri App (`app/`) — twenty-first pass

Twenty-first implementation pass: first-run onboarding wizard (§G.19);
simulation day playback controls (§G.16); Tauri bundler configuration (§G.8);
sidebar page prefetch and `.wsroute` import via command palette (§G.7).

**React frontend**
- `components/layout/OnboardingDialog.tsx` — welcome modal when `projectRoot` is unset; directory picker + `validate_project_root`; dismissible with persistence
- `pages/monitor/SimulationMonitor.tsx` — play/pause day playback with 1×/2×/4× speed multiplier on the day scrubber
- `utils/pagePrefetch.ts` — warms lazy route chunks on sidebar `mouseEnter`
- `hooks/useWsrouteImport.ts` — pick bundle → extract → navigate to Simulation Summary
- `constants/commands.ts` — "Import .wsroute Bundle" command palette action
- `package.json` — `tauri:dev`, `tauri:build`, `tauri:build:linux` scripts

**Tauri bundler**
- `tauri.conf.json` — explicit `deb`/`appimage`/`msi`/`dmg` targets; short/long description; Linux deb section; Windows NSIS install mode

**ROADMAP**
- §G.19 first-run onboarding wizard checked (partial — guided tour deferred)
- §G.16 day playback controls checked (partial — TripsLayer animation deferred)
- §G.8 Tauri bundler config + build scripts checked (partial — signing/auto-update deferred)
- §G.7 sidebar prefetch + palette bundle import checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — twentieth pass

Twentieth implementation pass: command palette (§G.7); Parquet table export;
bundle round-trip integration test; Vite manual chunk splitting for faster
initial load.

**Rust backend**
- `data::export_csv_to_parquet` — converts an on-disk CSV to Parquet via pandas/pyarrow subprocess
- `data::export_table_parquet` — writes in-memory tabular data to Parquet (temp CSV → convert)
- `wsroute_bundle_round_trip_preserves_jsonl` unit test — create → extract → verify `.jsonl` content

**React frontend**
- `components/layout/CommandPalette.tsx` — fuzzy-search overlay for all 17 views + theme/shortcuts actions; `Ctrl+K` or TopBar search button
- `constants/commands.ts` — shared palette command registry
- `utils/tableExport.ts` — `downloadParquetFromCsv()` and `downloadParquetTable()` helpers
- `DataExplorer`, `OutputBrowser`, `SimulationSummary` — Parquet export buttons alongside CSV
- `vite.config.ts` — `manualChunks` for echarts, maplibre, deck.gl, monaco vendor bundles

**ROADMAP**
- §G.7 command palette and Parquet export checked; manual chunk splitting noted (partial — <2s load target deferred)
- §G.8 bundle round-trip integration test checked (partial — Tauri bundler/updater deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — nineteenth pass

Nineteenth implementation pass: `.wsroute` bundle create/extract (§G.8); deck.gl
tile route map (§G.16); keyboard shortcuts help overlay; lazy-loaded pages and
SVG chart export (§G.7).

**Rust backend**
- `data::create_wsroute_bundle(source_dir, output_path)` — walks a run directory, zips eligible artefacts plus `manifest.json`
- `data::extract_wsroute_bundle(path, dest_dir)` — decompresses a `.wsroute` zip; returns first `.jsonl` log path for Simulation Summary

**React frontend**
- `components/maps/DeckRouteMap.tsx` — deck.gl `PathLayer` + `ScatterplotLayer` over MapLibre dark basemap; lazy-loaded from SimulationMonitor
- `components/layout/KeyboardShortcutsHelp.tsx` — modal overlay listing all global shortcuts; opened via `?` or TopBar button
- `App.tsx` — all page components lazy-loaded behind `Suspense`; `?` opens shortcuts help; `Escape` dismisses
- `pages/files/OutputBrowser.tsx` — "Export as .wsroute" on selected run; "Extract & Open" on bundle files
- `pages/monitor/SimulationMonitor.tsx` — ECharts / deck.gl route map toggle; SVG export on Cartesian map
- `utils/chartExport.ts` — `exportChartSvg()` for ECharts SVG download
- `vite.config.ts` — build target bumped to `es2022` for deck.gl BigInt literals

**Dependencies**
- `@deck.gl/core`, `@deck.gl/layers`, `@deck.gl/react`, `maplibre-gl`, `react-map-gl`

**ROADMAP**
- §G.8 bundle create/extract commands + Output Browser UI checked (partial — Tauri bundler/updater deferred)
- §G.16 deck.gl `PathLayer` tile route map checked
- §G.7 lazy-loaded pages, shortcuts help overlay, SVG export checked (partial — Parquet/command palette deferred)

### Fixed

#### WSmart-Route Studio — Tauri build

- Removed unused `protocol-asset` feature from `Cargo.toml` (mismatched Tauri allowlist)
- Corrected capability permissions to `core:*` identifiers for Tauri 2 ACL
- Added placeholder RGBA app icons required by `generate_context!()`

### Added

#### WSmart-Route Studio — Tauri App (`app/`) — eighteenth pass

Eighteenth implementation pass: analytical workflow navigation strip and
collapsible sidebar (§G.7); `P`/`M` keyboard shortcuts; `GlobalFilterBar`
propagated to Benchmark Analysis; MLflow dashboard iframe embed (§G.18);
`.wsroute` bundle export script and inspector (§G.8 partial).

**Python**
- `logic/gen/export_for_studio.py` — packages run output artefacts (CSV, JSON/JSONL, YAML, NPZ, PKL, PT, Parquet) into a `.wsroute` zip with `manifest.json`

**Rust backend**
- `data::inspect_wsroute_bundle(path)` — lists zip contents and parses bundle manifest
- `zip` crate dependency for bundle inspection

**React frontend**
- `components/layout/WorkflowNav.tsx` — Overview → Drill-Down → Geospatial → Registry → ML → HPO → Launch strip (§G.7)
- `components/layout/GlobalFilterBar.tsx` — shared policy/sample filter controls
- `store/layout.ts` — `sidebarOpen` state with persistence; TopBar toggle + mobile overlay backdrop
- `App.tsx` — `P` → Process Monitor, `M` → Simulation Digital Twin
- `pages/analysis/BenchmarkAnalysis.tsx` — global filter propagation + comparison CSV export
- `pages/analysis/ExperimentTracker.tsx` — MLflow Runs/Dashboard tabs; iframe embed + open-in-browser
- `pages/files/OutputBrowser.tsx` — `.wsroute` bundle manifest viewer
- `SimulationSummary`, `AlgorithmComparison` — `GlobalFilterBar` integration

**ROADMAP**
- §G.7 workflow nav, P/M shortcuts, sidebar collapse (partial), global filters to Benchmark checked
- §G.18 MLflow iframe embed fallback checked
- §G.8 export script + bundle inspector checked (partial — full import deferred)

---

#### WSmart-Route Studio — Tauri App (`app/`) — seventeenth pass

Seventeenth implementation pass: Monaco YAML editor in Config Editor (§G.13);
ZenML pipeline run browser with step-duration Gantt chart (§G.18); table CSV
export utility; global filters in Simulation Summary; HPO chart PNG export;
responsive layout container (§G.7).

**Rust backend**
- `commands/zenml.rs` — `list_zenml_pipeline_runs`, `load_zenml_run_steps`: Python subprocess queries ZenML via `Client.list_pipeline_runs` and `get_pipeline_run`

**React frontend**
- `components/editors/YamlEditor.tsx` — Monaco YAML editor (lazy-loaded) with dark/light theme sync; replaces raw textarea in ConfigEditor
- `pages/analysis/ZenMLPipelineView.tsx` — pipeline run table, step-duration horizontal bar chart (Gantt-style), CSV/PNG export
- `pages/analysis/ExperimentTracker.tsx` — embeds ZenML section; MLflow runs CSV export
- `pages/analysis/SimulationSummary.tsx` — respects `useGlobalFiltersStore`; ranking table CSV export; active filter badge
- `pages/analysis/HPOTracker.tsx` — PNG export buttons on all four ECharts panels
- `utils/tableExport.ts` — reusable `downloadCsv()` for table data export
- `components/layout/Layout.tsx` — max-width container (`1920px`) and responsive padding
- `types/index.ts` — `ZenmlPipelineRun`, `ZenmlPipelineStep` interfaces
- `package.json` — `@monaco-editor/react` dependency

**ROADMAP**
- §G.13 Monaco Editor integration checked
- §G.18 ZenML pipeline view checked
- §G.7 table CSV export (partial), responsive layout (partial), theme toggle noted done

---

#### WSmart-Route Studio — Tauri App (`app/`) — sixteenth pass

Sixteenth implementation pass: MLflow run browser and metric comparison (§G.18);
ECharts route map preview (§G.16 partial); global filter store, URL hash
bookmarking, chart PNG export, and `Ctrl+R` launch shortcut (§G.7).

**Rust backend**
- `commands/mlflow.rs` — `list_mlflow_runs`, `list_mlflow_metric_keys`, `load_mlflow_metric_history`: Python subprocess queries local/remote MLflow tracking via `mlflow.search_runs` and `MlflowClient`

**React frontend**
- `pages/analysis/ExperimentTracker.tsx` — MLflow run table with multi-select; metric comparison ECharts chart with normalize toggle; params panel; output dirs retained
- `pages/monitor/SimulationMonitor.tsx` — `RouteMapChart` ECharts scatter + path using `all_bin_coords` + `tour_indices`; fill-level colour coding; PNG export on charts
- `store/filters.ts` — `useGlobalFiltersStore` (policy + sampleId) propagates across SimulationMonitor and AlgorithmComparison
- `store/launchTrigger.ts` — nonce-based launch triggers for `Ctrl+R` on launcher pages
- `hooks/useHashSync.ts` — serializes `mode` + filters to URL hash for deep-linking
- `utils/chartExport.ts` — reusable `exportChartPng()` via ECharts `getDataURL()`
- `App.tsx` — `Ctrl+R` launches on active launcher page; `useHashSync()` on mount
- Launcher pages (`SimulationLauncher`, `TrainingHub`, `DataGeneration`, `EvaluationRunner`) — subscribe to launch trigger nonces
- `types/index.ts` — `MlflowRun`, `MlflowMetricPoint` interfaces

**ROADMAP**
- §G.18 MLflow run table and metric comparison chart checked
- §G.16 ECharts route map preview checked (deck.gl tile basemap still open)
- §G.7 global filters, URL hash bookmarking, chart PNG export, `Ctrl+R` checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fifteenth pass

Fifteenth implementation pass: session profiles in Output Browser (§G.14);
sensor CSV data source and dataset preview panel (§G.11); Config Editor Form
mode (§G.13); Optuna cross-study comparison (§G.18).

**Rust backend**
- `data::preview_dataset_stats(path, project_root, python_executable)`: Python subprocess inspects `.pkl`/`.pt` datasets; returns `DatasetPreviewStats` (instances, nodes, demand μ±σ, histogram, file size)

**React frontend**
- `store/sessionProfiles.ts` — `useSessionProfilesStore` (persist, max 20 profiles); captures/restores all three launcher Zustand stores via `captureLauncherSnapshot()` / `applyLauncherSnapshot()`
- `pages/files/OutputBrowser.tsx` — Session Profiles sidebar (§G.14): name input + Save button; load/delete profile list
- `pages/launch/DataGeneration.tsx` — sensor source (§G.11): third `dataSource` radio; CSV file picker; Hydra `data.source=sensor` + `data.sensor_file=<path>`
- `pages/launch/DataGeneration.tsx` — Instance Preview panel (§G.11): "Preview .pkl/.pt" button; KPI cards + ECharts demand histogram via `preview_dataset_stats`
- `pages/files/ConfigEditor.tsx` — Form mode (§G.13): fourth view toggle; typed widgets (checkbox/number/text) inferred from value; edits sync back to Raw YAML via `rowsToYaml()`
- `pages/analysis/HPOTracker.tsx` — cross-study comparison (§G.18): "Compare with" study dropdown; overlaid best-so-far line chart; side-by-side best-value KPI cards
- `store/launchers.ts` — `sensorCsvPath` field in `useDataGenStore`
- `types/index.ts` — `DatasetPreviewStats` interface

**ROADMAP**
- §G.14 session profiles checked
- §G.11 sensor source and preview panel checked
- §G.13 Form mode checked (partial — flat YAML, no OmegaConf schema introspection)
- §G.18 cross-study comparison checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — fourteenth pass

Fourteenth implementation pass: Output Browser compare-runs multi-select (§G.14);
structured directory tree with hydra/ auto-expand (§G.14); Config Editor Apply to
Launcher (§G.13); Data Generation TSPLIB source option (§G.11).

**React frontend**
- `pages/files/OutputBrowser.tsx` — compare runs (§G.14) + structured tree (§G.14):
  - Per-run checkbox multi-select; "Compare N Runs →" button when ≥2 selected
  - `findRunJsonl()` scans top-level and `hydra/` for `.jsonl` logs
  - `setPendingBenchmarkLogs` + navigate to `benchmark` mode
  - Auto-expand `hydra/` on run selection; `sortEntries()` prioritises config and log files
  - Highlight `pruned_config.yaml` / `.jsonl` entries in the file tree
- `pages/analysis/BenchmarkAnalysis.tsx` — consumes `pendingBenchmarkLogs` on mount; loads multiple simulation logs for side-by-side comparison
- `pages/files/ConfigEditor.tsx` — Apply to Launcher (§G.13):
  - Target selector (Simulation Launcher / Training Hub / Data Generation)
  - `applyConfigToLauncher()` maps flat YAML keys to Zustand store patches; navigates to target page
- `utils/configToLauncher.ts` — key-mapping utility for sim/train/data-gen Hydra fields + unmapped keys → `extraOverrides`
- `pages/launch/DataGeneration.tsx` — TSPLIB source option (§G.11):
  - `dataSource` radio: synthetic vs TSPLIB; `.vrp`/`.tsp` file picker via Tauri dialog
  - Hydra overrides `data.source=tsplib` + `data.tsplib_instance=<path>`; graph form hidden for TSPLIB mode
- `store/launchers.ts` — `dataSource` + `tsplibPath` persisted in `useDataGenStore`
- `store/app.ts` — `pendingBenchmarkLogs: BenchmarkLogRef[] | null` ephemeral handoff field

**ROADMAP**
- §G.14 compare runs and structured directory tree checked
- §G.13 Apply to Launcher checked
- §G.11 TSPLIB source option checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — thirteenth pass

Thirteenth implementation pass: dynamic policy registry in SimulationLauncher (§G.9);
Eval Runner → Benchmark Analysis handoff (§G.12); resolved Hydra config dump in ConfigEditor
(§G.13); Optuna study browser in HPOTracker (§G.18); Tauri OS notifications and Ctrl+. cancel
(§D.8/§G.7).

**Rust backend**
- `commands/policies.rs` — `list_sim_policies(project_root)`: parses `logic/configs/tasks/test_sim.yaml` for `/policies@p.{id}:` entries; returns sorted `SimPolicyEntry` list; falls back to 8 default policies when file is missing
- `commands/hpo.rs` — Optuna integration via Python subprocess:
  - `list_optuna_studies(storage_url, project_root, python_executable)`: enumerates studies with trial counts and best values
  - `load_optuna_study(storage_url, study_name, project_root, python_executable)`: returns trials, FANOVA importances, best value, and best params as JSON
- `commands/system.rs` — `dump_hydra_config(task, project_root, python_executable)`: runs `python main.py <task> --cfg job` and returns resolved YAML
- `commands/process.rs` — `resolve_python()` extracted as public helper shared by spawn, HPO, and Hydra commands

**React frontend**
- `pages/launch/SimulationLauncher.tsx` — dynamic policy registry (§G.9):
  - `availablePolicies` state loaded via `list_sim_policies` on `projectRoot` change
  - Scrollable checkbox grid (89 policies from `test_sim.yaml`); reload button with `RefreshCw` spinner
  - Stale selections pruned when registry reloads; count badge in header
- `pages/launch/EvaluationRunner.tsx` — "Open in Analytics →" button in `ResultsGrid` (§G.12):
  - Serialises result rows to `pendingEvalResults` in app store; navigates to `benchmark` mode
- `pages/analysis/BenchmarkAnalysis.tsx` — eval results panel (§G.12):
  - `EvalResultsPanel` component: 3-column bar charts (cost / gap / time) + summary table
  - Consumes `pendingEvalResults` on mount via `useEffect`; dismissible independently of simulation runs
- `pages/files/ConfigEditor.tsx` — resolved Hydra config loader (§G.13):
  - Task selector (test_sim / train / hpo / eval / gen_data) + "Load via --cfg job" button
  - Calls `dump_hydra_config`; populates Raw view without requiring a file on disk
- `pages/analysis/HPOTracker.tsx` — Optuna study browser rewrite (§G.18):
  - Storage URL input with SQLite file picker; study dropdown with trial counts
  - ECharts: optimisation history scatter + best-so-far line; FANOVA parameter importance bars; parallel coordinates
  - KPI cards (trials / completed / best value / param count); "Copy best params" as Hydra overrides
- `hooks/useProcessMonitor.ts` — OS notifications (§D.8) + cancel shortcut (§D.7):
  - `maybeSendOsNotification()`: requests permission and fires native notification when `document.hidden` on completed/failed
  - Global `Ctrl+.` listener cancels first running process via `cancel_process`
- `store/app.ts` — `pendingEvalResults: EvalAnalyticsRow[] | null` + `setPendingEvalResults` (ephemeral)
- `types/index.ts` — `SimPolicyEntry`, `EvalAnalyticsRow`, `OptunaStudySummary`, `OptunaTrial`, `OptunaStudyData`
- `App.tsx` — additional keyboard shortcuts: `G` → simulation monitor, `Q` → HPO tracker

**ROADMAP**
- §G.9 policy registry loading checked
- §G.12 Open in Analytics checked
- §G.13 Load resolved Hydra config checked
- §G.18 Optuna study browser (partial — history, importance, parallel coords, copy best params) checked
- §D.8 OS notifications checked; §D.7 Ctrl+. cancel checked

---

#### Analysis script & report — Pareto-front policy catalogue

- `logic/gen/gen_simulation_analysis.py` — new `build_pareto_front_table(df)` function:
  - Computes the Pareto front (min overflows, max kg/km) independently for each `(dist, improver)` panel
  - Groups front members by unique `(selection variant, constructor, improver)` key; merges `cf`/`sl_var` into a human-readable label (`LM (CF70)`, `SL (SL1)`, …)
  - Outputs a markdown table with columns: Selection | Constructor | Improver | Overflows | kg/km | Pareto-Front Scenarios
  - Scenarios column lists every `Region-N / Distribution` combination where that configuration reached the front; sorted descending by scenario count
  - Wired into `generate_markdown` at the end of section 2 (Analytics Comparison — Pareto View)
- `public/simulation_analysis.md` — "Pareto-Front Policy Catalogue" table inserted at the end of §2 (22 rows; BPC + ACO_HH + PG-CLNS dominate the front across all panels)

#### WSmart-Route Studio — Tauri App (`app/`) — twelfth pass

Twelfth implementation pass: live training mode in TrainingMonitor (§G.17); Lightning column
normalization in TrainingMonitor and TrainingHub (§G.17 parity); §G.16 Streamlit parity confirmed.

**React frontend**
- `pages/monitor/TrainingMonitor.tsx` — live training mode (§G.17) + column normalization:
  - `LIVE_KEY = "__live__"` constant: virtual run key for the live process entry in `metricsMap`
  - `normalizeMetricRow(raw)`: maps Lightning CSV column aliases to canonical `TrainingMetricsRow` keys — `train/rl_loss` / `train/il_loss` → `train_loss`; `val/cost` / `val_cost` → `val_loss`; `lr-*` prefix variants → `lr`; applied at both CSV load time and live stdout parse time
  - `parseMetricLine(line)` extended with `/`-containing key patterns (`\w[\w/]*`) to handle Lightning's `/`-separated metric names in key=value format
  - `METRIC_SIGNAL_KEYS` extended with Lightning variants: `train/rl_loss`, `train/il_loss`, `val/cost`, `val_cost`
  - `activeTrainId`: `useMemo` over `useProcessStore` — first `train_*` process with `status === "running"`
  - Live stdout `useEffect`: when `activeTrainId` is set, initializes `metricsMap[LIVE_KEY] = []` and attaches a `process:stdout` listener that calls `parseMetricLine` and appends parsed rows; cleans up on `activeTrainId` change
  - Auto-select `useEffect`: prepends `LIVE_KEY` to `selected` when `activeTrainId` appears; removes it when process exits
  - `runsMetrics` memo: live entry inserted first with `name: "Live Training"`
  - Live entry in run selector: `Radio` icon with `animate-pulse`; update count shown; checkbox to toggle manually
  - Live `RunPanel`-style block: green pulsing dot header + `GradNormSparkline` + `LrSparkline` for the live row set
  - CSV loading now applies `normalizeMetricRow` via `rows.map(normalizeMetricRow)` in `loadMetrics`
- `pages/launch/TrainingHub.tsx` — column normalization sync:
  - `METRIC_SIGNAL_KEYS` extended with Lightning column variants (same set as `TrainingMonitor.tsx`)
  - `normalizeMetricRow()` added (identical implementation); applied inside `parseMetricLine` for both JSON and key=value code paths
  - key=value regex updated to `(\w[\w/]*)` to capture `/`-separated metric names

**ROADMAP**
- §G.16 Streamlit parity check confirmed and checked
- §G.17 live training mode checked
- §G.17 column normalization checked
- §G.17 Streamlit parity check checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — eleventh pass

Eleventh implementation pass: session persistence for all three launcher forms (§G.9/G.10/G.11);
auto-navigate countdown in SimulationLauncher (§G.9); grad_norm + entropy sparklines in
TrainingHub (§G.10).

**React frontend**
- `store/launchers.ts` — new file; three persisted Zustand stores using a single `patch` action:
  - `useSimLauncherStore` (`wsroute-sim-launcher`): `selectedPolicies`, `area`, `numLoc`, `samples`, `nCores`, `seed`, `distribution`, `extraOverrides`
  - `useTrainHubStore` (`wsroute-train-hub`): all train/hpo/eval form fields
  - `useDataGenStore` (`wsroute-data-gen`): `problem`, `distributions`, `datasetType`, `seed`, `overwrite`, `area`, `numLoc`, `nSamples`, `nDays`, `extraOverrides`
- `pages/launch/SimulationLauncher.tsx` — session persistence (§G.9) + auto-navigate (§G.9):
  - Local `useState` for all form fields replaced with `useSimLauncherStore`
  - `navCountdown: number | null` state; first `useEffect` sets it to 5 when `simStatus === "completed"`; second `useEffect` decrements every second via `setTimeout` and calls `setMode("simulation_summary")` on 0
  - Countdown label `"(auto in Xs — cancel)"` shown beside "View Summary →" button; cancel clears countdown
- `pages/launch/TrainingHub.tsx` — session persistence (§G.10) + sparklines (§G.10):
  - Local `useState` for all form fields replaced with `useTrainHubStore`
  - `MiniSparkline` component: compact 70 px ECharts `line` chart; area fill at `color + "22"` opacity; returns `null` when all data values are null (metric not emitted by the run)
  - Grad norm sparkline (red `#f87171`) + entropy sparkline (purple `#a78bfa`) rendered as a 2-column grid below `LiveChart` when ≥2 metric updates have been received
- `pages/launch/DataGeneration.tsx` — session persistence (§G.11):
  - Local `useState` for all form fields replaced with `useDataGenStore`
  - `toggleDist` rewritten to avoid functional updater (incompatible with store `patch` signature)

**ROADMAP**
- §G.9 auto-navigate and session persistence checked
- §G.10 grad_norm + entropy sparklines and session persistence checked
- §G.11 session persistence checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — tenth pass

Tenth implementation pass: process toast notifications (§G.15); per-process progress bar (§G.15);
policy multi-select overlay on KPI timeseries (§G.16).

**React frontend**
- `hooks/useProcessMonitor.ts` — toast notifications (§G.15):
  - `import { toast } from "sonner"` added
  - `StatusUpdate` listener fires `toast.success` (4 s) / `toast.error` (6 s) / `toast.info` (3 s) on terminal status transitions; human-readable label extracted via `id.split("_")[0]`
- `pages/monitor/ProcessMonitor.tsx` — progress bar (§G.15):
  - `PROGRESS_MARKER = "PROGRESS:"` constant + `ProgressInfo` interface added
  - `getLatestProgress(logLines)` scans last 30 log lines for `PROGRESS:{json}` markers; returns `{ value, total?, label? }`; accepts both `value` and `current` keys
  - Progress bar rendered in `ProcessRow` between header row and log viewer when process is running and progress data is present; deterministic `width: pct%` bar when `total` is known, indeterminate pulsing bar otherwise
- `pages/monitor/SimulationMonitor.tsx` — policy multi-select overlay (§G.16):
  - `POLICY_COLORS` 8-colour palette (`#6366f1`, `#34d399`, `#f87171`, …) defined at module level
  - `MetricTimeseries` refactored: replaces `entries` + implicit single series with `policySeries: { policy; entries; color }[]`; builds one ECharts line series per policy; shows legend when >1 series; top grid margin increases to 20 when legend is visible; area fill only when single series
  - `chartPolicies: string[]` state + `activeChartPolicies` memo (defaults to all policies when `chartPolicies` is empty)
  - `toggleChartPolicy(p)` callback: XOR toggle; prevents deselecting all (resets to full set)
  - `policySeries` memo: maps each `activeChartPolicy` to filtered entries + assigned color
  - Chip-toggle row rendered below header controls when ≥2 policies present; chip border/text/background tinted with policy color; inactive chips at 35% opacity

**ROADMAP**
- §G.15 progress bar checked; cancel button confirmed already wired (no code change); toast notifications checked
- §G.16 policy/sample multi-select checked

---

#### WSmart-Route Studio — Tauri App (`app/`) — ninth pass

Ninth implementation pass: DataGeneration live progress panel (§G.11); OutputBrowser simulation
KPI summary card on run selection (§G.14).

**React frontend**
- `pages/launch/DataGeneration.tsx` — live progress panel (§G.11):
  - `liveProcessId`, `runStatus`, `logTail` state added
  - `useEffect([liveProcessId])`: subscribes to `process:stdout` (appends last 20 non-empty lines to `logTail`) and `process:status` (updates `runStatus`)
  - `launch` now generates a stable process ID and clears state before spawn
  - Live panel renders below the Launch button: `Activity`/`CheckCircle`/`XCircle` status icon; scrollable pre-block with last 20 stdout lines; "Process Monitor" navigation button
- `pages/files/OutputBrowser.tsx` — simulation KPI summary (§G.14):
  - `runKpi` state: `Array<{ policy, overflows, kgkm, profit }> | null`
  - `selectRun` now scans top-level entries for the first `.jsonl` ≤ 20 MB; reads via `read_text_file`; parses each line as `DayLogEntry`; aggregates per-policy means; sorted ascending by overflows
  - KPI card rendered below the config metadata card: 3-column micro-table (Policy | Overflows | kg/km); overflows colour-coded (green = 0, amber = low, red > 20)

**ROADMAP**
- §G.11 live progress checked; §G.14 simulation result summary checked

---

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
