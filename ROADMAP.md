# WSmart-Route — Analysis App Roadmap

This roadmap captures the full implementation plan for the **Advanced Visual Analytics and System Architecture for Combinatorial Optimization and Machine Learning** described in `markdown/Combinatorial Optimization Analysis App.md`. Each phase builds on the previous and represents a shippable milestone.

---

## Phase 0 — Foundation & Tooling

**Goal:** Establish the project scaffold, dev environment, and data pipeline so all subsequent phases have a stable base.

- [ ] Bootstrap Tauri 2.0 project (`src-tauri/` + React/TypeScript frontend)
- [ ] Configure Tailwind CSS and component library (shadcn/ui or similar)
- [ ] Set up Rust workspace with required crates: `arrow`, `parquet`, `ndarray-npy`, `serde`
- [ ] Define Arrow IPC schema for simulation log rows (city, N, dist, improver, strategy, constructor, overflows, kgkm, km, profit, kg, reward, ncol, kg_lost)
- [ ] Implement Rust backend command: load `public/global/simulation/simulation_summary.csv` → Arrow IPC stream
- [ ] Implement Rust backend command: load `public/global/datasets/` CSVs → Arrow IPC stream
- [ ] Wire frontend to receive Arrow buffers via Tauri `invoke` (bypassing JSON serialization)
- [ ] Spawn DuckDB-Wasm in a Web Worker; ingest simulation Arrow stream on app load
- [ ] Verify end-to-end latency: CSV → Rust → Arrow → DuckDB-Wasm in < 500 ms

---

## Phase 1 — Statistical Overview Dashboard (ECharts 2D)

**Goal:** Reproduce and extend the existing static `simulation_analysis.md` charts as interactive ECharts panels inside the app.

### 1.1 KPI Summary Bar / Box Charts
- [ ] Mean ± std overflows per constructor, grouped by mandatory-selection strategy
- [ ] Mean ± std kg/km per constructor, grouped by city/scale
- [ ] Interactive brushing: selecting a bar cross-filters all panels on the dashboard

### 1.2 Overflow vs Efficiency Scatter (Pareto Front)
- [ ] 4-panel layout: Gamma-3/FTSP · Empirical/FTSP · Gamma-3/CLS · Empirical/CLS
- [ ] Color encoding: LA · LM-CF70 · LM-CF90 · SL-SL1 · SL-SL2
- [ ] Marker shape: RM-100 circle · RM-170 square · FFZ-350 diamond
- [ ] Computed Pareto front drawn as white dashed step line
- [ ] Log-scale toggle (symlog X) for the same view
- [ ] Hover tooltip: all config values + KPI values

### 1.3 Policy Configuration Heatmaps
- [ ] Heatmap split by distribution (Gamma-3 vs Empirical)
- [ ] Heatmap split by graph (RM-100 vs RM-170 vs FFZ-350)
- [ ] Cell value = mean overflows or mean kg/km (toggle)
- [ ] Color gradient from dark (worst) to bright (best)

### 1.4 Parallel Coordinates (Hyper-Dimensional Policy Explorer)
- [ ] Axes: city · N · dist · improver · strategy · constructor · overflows · kgkm · km · profit
- [ ] Each of the 480 simulation logs rendered as a polyline
- [ ] Brushing on any axis instantly filters all other panels (via DuckDB-Wasm SQL)
- [ ] Highlight corridor: drag brush on overflows ≤ threshold to identify zero-overflow configs
- [ ] Color polylines by mandatory-selection strategy

### 1.5 Constructor Ranking Chart
- [ ] Horizontal bar chart, bars growing left-to-right (bottom-up ordering)
- [ ] Rank by mean kg/km across all configurations
- [ ] Error bars showing std deviation
- [ ] Dark background, consistent color scheme

### 1.6 Secondary Log-Scale Views
- [ ] Auto-generate log-scale version below each chart that benefits from it (overflow counts, profit ranges)
- [ ] City Comparison section uses log scale only (not outlier removal) to preserve extreme values

---

## Phase 2 — Hierarchical Drill-Down (Sunburst / Treemap)

**Goal:** Enable macro → micro navigation from algorithm family level down to individual config variants.

- [ ] Top-level Sunburst chart: inner ring = city/scale · middle ring = selection strategy · outer ring = constructor
- [ ] Angular span mapped to accumulated profit; color gradient = kg/km efficiency
- [ ] Click on any segment fires DuckDB-Wasm filter query
- [ ] Drill-down transition: Sunburst morphs into horizontal bar chart (mean ± variance per variant)
- [ ] Error bars on drill-down bars representing variance across Empirical vs Gamma-3 distributions
- [ ] Breadcrumb trail showing current filter path; click to navigate back up
- [ ] Treemap alternative view: area = profit, color = overflows (toggle with Sunburst)

---

## Phase 3 — Geospatial Routing Visualization (deck.gl)

**Goal:** Animate the physical routes constructed by each algorithm over the real-world city graphs.

### 3.1 Base Map Layer
- [ ] Integrate deck.gl with MapLibre GL (OpenStreetMap tiles)
- [ ] Load node coordinates for Rio Maior (N=100, N=170) and Figueira da Foz (N=350) from graph JSON files
- [ ] Render nodes as ScatterplotLayer: radius ∝ profit value, color = fill level / overflow status
- [ ] Render depot as distinct marker
- [ ] Pan/zoom/tilt with 3D perspective

### 3.2 Route Animation (TripsLayer)
- [ ] Parse per-day route files from simulation output into timestamped coordinate arrays
- [ ] Feed routes into deck.gl TripsLayer with vehicle-color-coded trails
- [ ] Timeline slider: step through 30-day simulation day by day
- [ ] Playback controls: play / pause / speed multiplier
- [ ] Multi-vehicle rendering with distinct color coding per vehicle

### 3.3 Algorithm Comparison Mode
- [ ] Side-by-side view: BPC routes vs ACO_HH routes on same map
- [ ] Toggle visibility per policy
- [ ] Overlay profit nodes that were skipped (dimmed) vs visited (bright)

### 3.4 Non-Geographic Cartesian Mode (OrbitView)
- [ ] Switch between geographic (Mercator) and abstract Cartesian coordinate system
- [ ] OrbitView camera: orbit, pan, zoom a 3D point cloud
- [ ] Used for normalized/synthetic datasets where coordinates are not GPS

---

## Phase 4 — Topological Graph Analytics (Sigma.js / Cosmograph)

**Goal:** Visualize the raw optimization graph structure, pheromone trails, and node-edge weights.

- [ ] Load distance matrix from `assets/` as a weighted edge list
- [ ] Render graph using Sigma.js (WebGL): node radius ∝ profit, edge thickness ∝ inverse distance
- [ ] Force-directed layout (ForceAtlas2) via Graphology
- [ ] ACO pheromone trail visualization: edge opacity/color intensity ∝ accumulated pheromone weight after each iteration
- [ ] Cross-filter from DuckDB-Wasm: brushing a profit range highlights matching nodes
- [ ] Dynamic re-layout when filter applied: clusters emerge based on algorithm prioritization
- [ ] Cosmograph alternative for large dense graphs (N=350)
- [ ] Timeline slider synced with route animation to show pheromone evolution over iterations

---

## Phase 5 — Machine Learning Introspection Dashboard

**Goal:** Expose the internals of trained neural CO models (Attention Models, Routing Transformers).

### 5.1 TensorDict Data Pipeline
- [ ] Rust backend: load `.npy`/`.npz` TensorDict files via `ndarray-npy` crate
- [ ] Memory-map large tensor files (avoid full RAM load)
- [ ] Stream specific tensor slices to frontend over Arrow IPC on demand

### 5.2 3D Loss Landscape Visualization (React Three Fiber)
- [ ] Python utility script: compute loss surface grid using Li et al. filter-normalized random directions
- [ ] Export 2D grid of loss values as `.npz`
- [ ] React Three Fiber: render grid as `InstancedMesh` continuous 3D topography
- [ ] Color gradient: low loss = deep blue, high loss = bright red
- [ ] Camera: orbit, zoom, perspective controls
- [ ] Overlay 2D ECharts contour map adjacent to the 3D canvas (CSS positioned)
- [ ] Project exact-solver solutions (BPC optimum) as a marker on the landscape
- [ ] Identify sharp vs flat minima; annotate with generalization notes (Gamma-3 vs Empirical)

### 5.3 Attention Weight Visualization (Sigma.js overlay)
- [ ] Load attention weight matrices from TensorDict for a selected simulation step
- [ ] Render as bipartite graph on top of node coordinates: edge opacity ∝ attention weight magnitude
- [ ] Query/Key/Value color coding per attention head
- [ ] Timeline slider: step through sequential decoding steps
- [ ] Sparse Routing Transformer mode: show only top-k attention connections (spherical k-means clusters)
- [ ] Compare attention patterns of model trained on Empirical vs Gamma-3 distributions
- [ ] Side-by-side vs overlay toggle

---

## Phase 6 — OLAP Data Cube Explorer

**Goal:** Give the researcher a free-form SQL/pivot interface backed by DuckDB-Wasm for custom analysis queries.

- [ ] DuckDB-Wasm query editor with syntax highlighting (Monaco or CodeMirror)
- [ ] Pre-built query templates: robustness profile, variance analysis, Pareto efficiency frontier
- [ ] Result grid with sortable columns and export to CSV
- [ ] Auto-chart: map query result columns to ECharts chart type suggestions
- [ ] Pivot table UI: drag dimensions/measures onto row/column/value wells
- [ ] Cross-filtering from pivot table updates all Phase 1–2 charts bidirectionally

---

## Phase 7 — Integrated Workflow & UX Polish

**Goal:** Connect all phases into a single cohesive analytical narrative flow.

- [ ] App-level navigation: Overview → Drill-Down → Geospatial → Graph → ML → Query
- [ ] Global filter state management (Zustand or Jotai): any filter applied in one view propagates to all others
- [ ] Bookmarkable analysis states (serialize filter + view to URL hash)
- [ ] Dark theme throughout (consistent with existing chart style: `#1a1a2e` background)
- [ ] Keyboard shortcuts: `G` = geospatial, `P` = parallel coords, `M` = ML dashboard, `Q` = query
- [ ] Responsive layout for different screen sizes (primary target: 2560×1440 research workstation)
- [ ] Performance: app loads and renders all baseline charts in < 2 s on target hardware
- [ ] Export: any chart exportable as PNG/SVG; any table exportable as CSV/Parquet

---

## Phase 8 — Data Export & Packaging

**Goal:** Make the app distributable and extend the Python pipeline to output app-compatible data.

- [ ] Python export script: `logic/scripts/export_for_app.py` — packages simulation CSV + graph JSONs + TensorDict NPZs into a single `.wsroute` bundle (zipped Parquet + Arrow IPC)
- [ ] Rust backend: open `.wsroute` bundle directly (no separate file management)
- [ ] Tauri bundler: produce signed `.deb`/`.AppImage` (Linux), `.dmg` (macOS), `.msi` (Windows) distributables
- [ ] Auto-update via Tauri updater plugin
- [ ] Integration test: round-trip export → import → verify all 480 simulation rows load correctly

---

## Dependency Map

```
Phase 0  →  Phase 1, Phase 3, Phase 4, Phase 5
Phase 1  →  Phase 2, Phase 6
Phase 2  →  Phase 7
Phase 3  →  Phase 4
Phase 4  →  Phase 5 (pheromone + attention share Sigma.js)
Phase 5  →  Phase 7
Phase 6  →  Phase 7
Phase 7  →  Phase 8
```

## Technology Stack Summary

| Concern | Technology |
|---|---|
| Desktop shell | Tauri 2.0 (Rust backend + native WebView) |
| Frontend framework | React 19 + TypeScript |
| Styling | Tailwind CSS |
| Data serialization | Apache Arrow IPC (zero-copy Rust ↔ JS) |
| In-browser OLAP | DuckDB-Wasm (Web Worker) |
| 2D charts | Apache ECharts |
| Geospatial rendering | deck.gl (WebGL, TripsLayer, OrbitView) |
| Graph visualization | Sigma.js v4 + Graphology / Cosmograph |
| 3D ML visualization | React Three Fiber (Three.js) |
| Tensor I/O (Rust) | ndarray-npy crate |
| State management | Zustand |
