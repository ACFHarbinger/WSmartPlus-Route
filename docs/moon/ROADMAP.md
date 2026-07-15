# WSmart+ Route — Moon Roadmap

> **Version**: 1.1  
> **Last Updated**: July 2026  
> **Status**: Living Document — updated each sprint  
> **Scope**: Logic layer (`logic/src/`), GUI layer (migrating from `gui/src/` PySide6 → Tauri), CI/CD, documentation

This document captures medium-to-long-horizon improvements for the WSmart+ Route framework across seven dimensions: Analytics & Interpretability, Architecture, Documentation, GUI/UX, New Features, Performance, and the WSmart-Route Studio Tauri application. Each item follows a **Pain → Options → Recommendation** structure with effort/impact tags.

Tags: `[Quick Win]` ≤ 1 day · `[Research]` involves novel work · `[Blocked]` depends on another item

---

## Anchor Index

| Section                                                              | Topic                                                                  |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [§A — Analytics & Interpretability](#a--analytics--interpretability) | Telemetry, attention maps, policy dashboards, HPO analytics            |
| [§B — Architecture](#b--architecture)                                | Test coverage, plugin system, logging, type safety, interfaces         |
| [§C — Documentation](#c--documentation)                              | API docs, architecture diagrams, Jupyter notebooks, CI docs pipeline   |
| [§D — GUI / UX](#d--gui--ux)                                         | Route visualization, training progress, themes, session persistence    |
| [§E — New Features](#e--new-features)                                | Multi-problem benchmarking, REST API, LLM integration, export formats  |
| [§F — Performance](#f--performance)                                  | Batched inference, GPU memory, test suite speed, simulation throughput |
| [§G — WSmart-Route Studio](#g--wsmart-route-studio)                  | Tauri 2.0 app: analytics, geospatial, ML introspection, launcher UIs  |

---

## A — Analytics & Interpretability

### §A.1 — Interactive Route Solution Visualizer

**Pain**: Solutions (tours, collected bins, costs) are currently logged as JSON arrays. There is no visual overlay of routes on a spatial canvas, making debugging decoder outputs and comparing policies against each other extremely tedious.

**Options**

- **A** — Add an ECharts panel inside the Studio analysis view: render depot, nodes, edges, and colour-code routes per vehicle using an ECharts `custom` series or a lightweight 2D canvas renderer. Low friction, consistent with the rest of the Studio tech stack.
- **B** — Export solutions to GeoJSON and open them in a browser via Folium/Leaflet, decoupled from the application.
- **C** — Use Plotly Dash as a standalone web dashboard, running in a background process launched from `main.py`.
- **D** — Integrate `rerun.io` for a time-scrubbing 2-D trajectory viewer (works well with simulation day-by-day replay).
- **E** — Use the deck.gl `PathLayer` + `ScatterplotLayer` inside the Studio's geospatial view (§G Phase 3) — the same WebGL renderer used for geospatial routing, repurposed for abstract Cartesian coordinates with the OrbitView camera.

**Recommendation**: **Option A first** (lowest cost, consistent with Studio stack), then **Option E** as the production-quality widget — deck.gl scales better and integrates with the Studio's geospatial phase. Option D is interesting for multi-day simulation replay but requires an external runtime.

**Effort × Impact**: Medium effort / High impact

**Delivered (§A.1 Option A — hundred-fourteenth pass)**

- [x] ``RouteViz`` React component — shared ECharts spatial panel with star depot, demand-sized tour nodes, per-vehicle coloured edges, and optional failure overlay (overflow / skipped high-fill highlights)
- [x] ``routeViz.ts`` — ``buildRouteVizOption`` utility; reuses ``resolveBinPositions`` + ``splitVehicleTourIndices``
- [x] Simulation Monitor — refactored inline ``RouteMapChart`` to ``RouteViz`` (ECharts mode)
- [x] Simulation Summary — day scrubber + multi-policy route comparison grid in analysis view
- [x] PNG/SVG export via ``ChartExportButtons`` (§G.7)

**Status**: §A.1 Option A complete — Option E (deck.gl PathLayer) already delivered via ``DeckRouteMap`` (§G.3 / §G.16); Options B/C/D deferred.

---

### §A.2 — Attention Map Visualization for Neural Decoders

**Pain**: The AM, TAM, DDAM, and MoE decoders compute multi-head attention over node embeddings, but these attention weights are never exported or displayed. Without visibility into what the model attends to, diagnosing routing errors or comparing trained heads is guesswork.

**Options**

- **A** — Hook `nn.MultiheadAttention` outputs with forward hooks; buffer the last batch's attention tensors in a ring-buffer on the model object. Visualize as a heatmap in the Studio ML introspection phase (§G Phase 5).
- **B** — Integrate `BertViz`-style row-column attention visualizer adapted for graph problems (node × node matrix).
- **C** — Log attention weights to WandB / TensorBoard as image summaries during evaluation; no GUI integration needed.
- **D** — Export attention weights to `.npz` per inference call and build a separate offline viewer script.

**Recommendation**: **Option C** for fast iteration (zero GUI work), then **Option A** for the Studio integration once Option C has validated that the data is interpretable. Option B is academic-grade but requires a browser runtime.

**Effort × Impact**: Low effort (Option C) → Medium effort (Option A) / High impact

**Delivered (§A.2 Option C — hundred-thirteenth pass)**

- [x] ``logic/src/tracking/logging/visualization/heatmaps.py`` — runtime attention capture via ``add_attention_hooks``, PNG rendering, WandB ``wandb.Image`` + TensorBoard ``add_image`` logging
- [x] ``AttentionHeatmapCallback`` — validation-epoch hook; respects ``tracking.log_attention``, ``tracking.log_attention_heatmaps``, and ``viz_every_n_epochs``
- [x] ``WSTrainer`` — auto-registers callback when tracking flags enabled
- [x] Eval engine — ``maybe_log_eval_attention_heatmaps()`` after ``evaluate_policy`` when ``tracking.log_attention*`` set
- [x] Unit tests in ``logic/test/unit/tracking/test_attention_heatmaps.py``

**Delivered (§A.2 Option A — hundred-seventeenth pass)**

- [x] ``AttentionRingBuffer`` — fixed-capacity ring-buffer for encoder attention snapshots (layer, head, decode step, normalised matrix)
- [x] ``install_attention_ring_buffer`` / ``ensure_attention_buffer`` — persistent forward hooks on encoder MHA layers
- [x] ``attention_emit.py`` — ``ATTENTION_VIZ_START:`` stdout + ``attention_viz.jsonl`` append when ``tracking.log_attention`` enabled
- [x] ``maybe_log_eval_attention_heatmaps`` — integrates ring-buffer capture + Studio emission after eval/validation
- [x] Rust ``parse_attention_viz_line`` + ``load_attention_viz_log`` command
- [x] Studio ``RuntimeAttentionPanel`` — ECharts heatmap with snapshot/layer/head selectors on Training Monitor + ML Introspection Attention tab
- [x] Unit tests in ``logic/test/unit/tracking/test_attention_buffer.py``

**Delivered (§A.2 Option A — hundred-thirtieth pass)**

- [x] ``collectAttentionVizFromLogLines`` — shared ``ATTENTION_VIZ_START:`` parser for process stdout
- [x] Training Hub — ``RuntimeAttentionPanel`` during live train/hpo runs; stdout ingest alongside metrics (§G.10 / §A.2)
- [x] Process Monitor — ``RuntimeAttentionPanel`` for selected ``train_`` / ``hpo_`` processes (§G.15 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-first pass)**

- [x] ``findActiveLiveTrainProcessId`` / ``findActiveHpoProcessId`` — shared train/HPO process detection for live analytics
- [x] Training Monitor — live stdout ingest for ``hpo_*`` processes; ``Live HPO`` label when HPO active (§G.17 / §A.2)
- [x] HPO Tracker — ``RuntimeAttentionPanel`` during live ``hpo_*`` runs; ``Process Monitor →`` navigation shortcut (§G.18 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-second pass)**

- [x] Experiment Tracker — ``RuntimeAttentionPanel`` during live ``hpo_*`` runs; ``HPO Tracker →`` + ``Process Monitor →`` shortcuts (§G.18 / §A.2)
- [x] Training Monitor / Process Monitor / HPO Tracker — cross-page navigation shortcuts for live train/HPO workflows (§G.15 / §G.17 / §G.18 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-third pass)**

- [x] Experiment Tracker — ``Training Monitor →`` shortcut during live ``hpo_*`` runs (§G.18 / §A.2)
- [x] Training Monitor / HPO Tracker / Process Monitor / Training Hub — ``Experiment Tracker →`` shortcut when live HPO active (§G.10 / §G.15 / §G.17 / §G.18 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-fourth pass)**

- [x] Training Monitor / Process Monitor / HPO Tracker / Experiment Tracker — ``Training Hub →`` shortcut during live train/HPO workflows (§G.10 / §G.15 / §G.17 / §G.18 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-fifth pass)**

- [x] ``TrainHpoNavMesh`` — shared cross-page train/HPO navigation component; replaces duplicated shortcut buttons on Training Hub, Training Monitor, Process Monitor, HPO Tracker, and Experiment Tracker (§G.7 / §A.2 / §A.4)

**Status**: §A.2 Options A+C complete — Option B (BertViz) deferred.

---

### §A.3 — Policy Telemetry Dashboard (Extension of `PolicyVizMixin`)

**Pain**: `logic/src/tracking/viz_mixin.py` already records per-iteration metrics (cost, feasibility, elapsed time) into a fixed-capacity ring-buffer via `_viz_record()`. However, this data is only accessible programmatically through `get_viz_data()` and is never surfaced to the user during or after a run.

**Options**

- **A** — Wire `get_viz_data()` output into the Studio's analytics view: after a simulation run, populate an ECharts bar chart with per-policy metrics (cost trajectories, improvement curves). Synergises with §G Phase 1. `[Quick Win]`
- **B** — Emit ring-buffer snapshots over a WebSocket / Tauri event channel to a React panel refreshed at 2 Hz while the simulation runs. Synergises with §G Phase 15 (Real-Time Process Monitor).
- **C** — Persist ring-buffer dumps to a SQLite database (`assets/telemetry.db`) and query them across runs for cross-policy trending.
- **D** — Push telemetry to Prometheus and visualize in Grafana (overkill for single-machine runs).

**Recommendation**: **Option A** immediately (hours of work), **Option C** for multi-run analytics once the database schema is stable.

**Effort × Impact**: Very Low effort (Option A) / High impact

**Delivered (§A.3 Option A — hundred-ninth pass)**

- [x] ``POLICY_VIZ_START:`` stdout + JSONL log marker from ``policy_viz_emit.py`` after route construction / improvement when ``PolicyVizMixin.get_viz_data()`` is non-empty
- [x] Rust ``parse_policy_viz_line`` + ``load_policy_viz_log`` + ``sim:policy_viz_update`` watcher events
- [x] Studio ``PolicyTelemetryPanel`` — ECharts cost trajectories, operator histograms, and algorithm-specific charts (ALNS/HGS/ACO/ILS/selector/generic) on Simulation Monitor
- [x] Live ingest via ``process:stdout`` parser + historical load on log open; PNG/SVG export via ``ChartExportButtons`` (§G.7)

**Delivered (§A.3 Option B — hundred-nineteenth pass)**

- [x] ``PolicyVizStreamSession`` — daemon thread emits growing ring-buffer snapshots every 0.5 s (2 Hz) during route construction / improvement
- [x] Route actions wrap ``adapter.execute`` and ``processor.process`` in stream sessions; final snapshot on context exit
- [x] Studio sim store upserts policy-viz entries by policy/sample/day/type (replaces stale snapshots during live runs)
- [x] ``PolicyTelemetryPanel`` — 2 Hz throttled ECharts refresh + **Live · 2 Hz** badge when file-watcher or ``test_sim`` process is active
- [x] Live ingest via ``process:stdout`` (§G.15) + ``sim:policy_viz_update`` file-watcher events
- [x] Unit tests in ``logic/test/unit/tracking/test_policy_viz_emit.py``

**Delivered (§A.3 Option C — hundred-twentieth pass)**

- [x] ``policy_telemetry_db.py`` — SQLite store at ``assets/telemetry.db`` with ``simulation_runs`` + ``policy_viz_snapshots`` tables
- [x] ``persist_policy_viz_snapshot`` — upserts terminal ring-buffer per run × policy × sample × day on each ``POLICY_VIZ_START:`` emit
- [x] ``query_policy_telemetry_trends`` — cross-run rows with ``final_metric``, ``step_count``, and algorithm family filter
- [x] Rust ``load_policy_telemetry_trends`` command (Python subprocess bridge)
- [x] Studio ``PolicyTelemetryTrendsPanel`` — cross-run comparison bar chart, steps chart, and history table on Simulation Monitor
- [x] Unit tests in ``logic/test/unit/tracking/test_policy_telemetry_db.py``

**Delivered (§A.3 Option C — hundred-twenty-first pass)**

- [x] ``query_policy_trajectory_series`` — extracts improvement curves (``best_cost`` / ``global_best_cost`` / etc.) from persisted ``data_json`` ring-buffers
- [x] Rust ``load_policy_trajectory_trends`` command — Python subprocess bridge for trajectory payloads
- [x] Studio ``PolicyTelemetryTrendsPanel`` — cross-run improvement trajectory line chart with policy filter + optional EMA smoothing; PNG export via ``ChartExportButtons`` (§G.7)
- [x] Unit tests for trajectory query roundtrip and policy-type filtering

**Delivered (§A.3 Option C — hundred-twenty-second pass)**

- [x] ``buildTrendTrajectoryOption`` — trajectory x-axis uses unioned solver step indices (iteration / generation) from persisted ring-buffers instead of array index
- [x] ``PolicyTelemetryTrendsPanel`` — history table CSV export via ``exportPolicyTelemetryTrendsCsv``; row click brushes global policy / ``run_label`` filter (§G.6 / §G.7)
- [x] Simulation Monitor — passes ``initialPolicy`` to pre-filter trajectory dropdown from active policy selection
- [x] Benchmark Analysis — ``PolicyTelemetryTrendsPanel`` for portfolio cross-run solver telemetry (§G.1 / §A.3)

**Delivered (§A.3 Option C — hundred-twenty-third pass)**

- [x] ``filterTrendRows`` / ``filterTrajectorySeries`` — global policy / ``run_label`` brush filters comparison, steps, and trajectory chart data (§G.6 / §G.7)
- [x] ``PolicyTelemetryTrendsPanel`` — chart click brushes global policy / run; active-brush badge + clear control; trajectory CSV via ``exportPolicyTrajectoryCsv``
- [x] ``query_policy_trajectory_series`` — includes ``run_label`` on each trajectory payload for run-key brush parity
- [x] Simulation Summary — ``PolicyTelemetryTrendsPanel`` with ``initialPolicy`` from active chart brush (§G.1 / §A.3)

**Delivered (§A.3 Option C — hundred-twenty-fourth pass)**

- [x] ``buildTrendComparisonOption`` / ``buildTrendStepsOption`` / ``buildTrendTrajectoryOption`` — brush dimming via ``TrendBrushFilter`` + ``chartHighlight`` opacity (non-selected series stay visible at 25%)
- [x] ``PolicyTelemetryTrendsPanel`` — history table uses ``filteredRows``; empty-state when brush excludes all rows; charts dim from full dataset (not hard-filtered)
- [x] Algorithm Comparison — ``PolicyTelemetryTrendsPanel`` with ``initialPolicy`` from global brush (§G.1 / §A.3)
- [x] City Comparison — ``PolicyTelemetryTrendsPanel`` with ``initialPolicy`` from global brush (§G.1.6 / §A.3)
- [x] Benchmark Analysis — ``initialPolicy`` brush sync on ``PolicyTelemetryTrendsPanel`` (parity with Simulation Summary)

**Delivered (§A.3 Option C — hundred-twenty-fifth pass)**

- [x] ``query_policy_telemetry_trends`` / ``query_policy_trajectory_series`` — optional ``run_label`` SQL filter for server-side portfolio scoping
- [x] Rust ``load_policy_telemetry_trends`` / ``load_policy_trajectory_trends`` — ``run_label`` bridge arg; panel passes active global brush to Python queries
- [x] ``PolicyTelemetryTrendsPanel`` — ``initialRunLabel`` prop syncs global run brush; steps chart click indexes ``displayStepRows`` (fixes brush click parity)
- [x] Simulation Summary / Benchmark Analysis / City Comparison / Algorithm Comparison — ``initialRunLabel`` from portfolio single-run brush (§G.1 / §G.6)
- [x] OLAP Explorer — ``PolicyTelemetryTrendsPanel`` with policy + run_label brush sync (§G.6 / §A.3)
- [x] Unit tests for ``run_label`` filter roundtrip in ``logic/test/unit/tracking/test_policy_telemetry_db.py``

**Delivered (§A.3 Option C — hundred-twenty-sixth pass)**

- [x] ``runLabelFromPath`` — shared ``Path.stem`` helper for SQLite ``run_label`` keys from log paths
- [x] ``PolicyTelemetryTrendsPanel`` — trajectory chart click indexes ``allSeries`` (fixes brush click when chart shows dimmed full dataset)
- [x] Simulation Monitor — ``initialRunLabel`` from active log path stem; cross-run trends scoped to open simulation (§G.15 / §A.3)
- [x] Data Explorer — ``PolicyTelemetryTrendsPanel`` with policy + run_label brush sync (§G.16 / §A.3)

**Delivered (§A.3 Option C — hundred-twenty-seventh pass)**

- [x] Output Browser — ``PolicyTelemetryTrendsPanel`` scoped to selected run via ``runJsonlPath`` stem + global brush sync (§G.14 / §A.3)
- [x] Output Browser — KPI summary policy rows click-to-brush global policy filter (parity with trends panel table)

**Delivered (§A.3 Option C — hundred-twenty-eighth pass)**

- [x] Output Browser — auto ``run_label`` brush on run select via ``setRunLabel`` + run list ring highlight when brush active (§G.14 / §A.3)
- [x] ``extractJsonlPathFromLogLines`` — scan process stdout for ``.jsonl`` paths to derive SQLite ``run_label`` keys
- [x] ``collectPolicyVizFromLogLines`` / ``uniquePolicyVizPolicies`` — parse per-process ``POLICY_VIZ_START:`` markers from stdout
- [x] Process Monitor — ``PolicyTelemetryPanel`` + ``PolicyTelemetryTrendsPanel`` for selected ``test_sim`` processes; policy chip brush + live 2 Hz refresh (§G.15 / §A.3)

**Delivered (§A.3 Option C — hundred-twenty-ninth pass)**

- [x] ``runLabelFromLogLines`` — shared ``run_label`` derivation from process stdout + fallback id
- [x] Process Monitor — always sync ``run_label`` brush on ``test_sim`` select; process row ring highlight when global brush matches (§G.15 / §A.3)
- [x] Simulation Launcher — ``PolicyTelemetryPanel`` + ``PolicyTelemetryTrendsPanel`` during live runs; policy chip + KPI card brush + ``run_label`` auto-sync (§G.9 / §A.3)

**Status**: §A.3 Options A+B+C complete.

---

### §A.4 — RL Loss Landscape & Training Health Monitoring

**Pain**: The Lightning-based RL pipeline (`logic/src/pipeline/rl/`) logs loss values but provides no automated detection of training instability (exploding/vanishing gradients, policy collapse, reward stagnation). Researchers must manually inspect WandB logs.

**Options**

- **A** — Add a `TrainingHealthCallback` (Lightning callback) that raises structured warnings when: gradient norm > 100, reward moving average stagnates for > 50 epochs, entropy < threshold. Log to the structured logging system.
- **B** — Use `PyHessian` to compute the top-K Hessian eigenvalues of the policy network periodically; log sharpness as a training health proxy. `[Research]`
- **C** — Visualize the loss landscape slice (perturbation method) after training completes; save as a PNG artefact to `assets/analysis/`. See §G Phase 5 for the Studio's 3D loss landscape viewer.
- **D** — Add gradient norm and entropy to the existing WandB sweep metrics so Optuna / DEHB can prune unhealthy runs early.

**Recommendation**: **Option A** is a mandatory baseline — training health guardrails belong in every production RL pipeline. **Option D** pairs naturally with HPO (already integrated) and costs one additional metric log line. **Option B/C** are research-grade extras.

**Effort × Impact**: Low–Medium effort / High impact

**Delivered (§A.4 Option A — hundred-eleventh pass)**

- [x] ``TrainingHealthCallback`` — Lightning callback detecting gradient norm explosion (>100), reward stagnation (>50 epochs), and entropy collapse (<0.01); loguru warnings + alert cooldown
- [x] ``training_health_emit.py`` — ``TRAINING_HEALTH_START:`` stdout + ``training_health.jsonl`` under Lightning ``log_dir``
- [x] ``WSTrainer`` — auto-registers ``TrainingHealthCallback`` alongside checkpoint and tracking callbacks
- [x] Rust ``parse_training_health_line`` + ``load_training_health_log`` command
- [x] Studio ``TrainingHealthPanel`` — severity-coded alert list on Training Monitor; live stdout ingest + historical ``training_health.jsonl`` load
- [x] Unit tests in ``logic/test/unit/pipeline/callbacks/test_training_health.py``

**Delivered (§A.4 Option D — hundred-eighteenth pass)**

- [x] ``HpoHealthMetricsCallback`` — per-epoch ``train/grad_norm`` + ``train/entropy`` reporting to Optuna trial user attrs and WSTracker ``hpo/*`` metrics
- [x] Optuna objective — health callback wired alongside ``PyTorchLightningPruningCallback``; unhealthy trials pruned via ``TrialPruned``
- [x] DEHB objective — ``apply_dehb_health_penalty`` penalises fitness on grad explosion / entropy collapse
- [x] Ray Tune objective — per-epoch ``ray.train.report`` with ``grad_norm`` + ``entropy`` for ASHA schedulers
- [x] Studio HPO Tracker — trial health table with grad norm, entropy, and ``health_pruned`` badge (§G.18 bridge)
- [x] Unit tests in ``logic/test/unit/pipeline/callbacks/test_hpo_health.py``

**Delivered (§A.4 Option A — hundred-thirtieth pass)**

- [x] ``collectTrainingHealthFromLogLines`` — shared ``TRAINING_HEALTH_START:`` parser for process stdout
- [x] Training Hub — ``TrainingHealthPanel`` during live train/hpo runs; stdout ingest alongside metrics (§G.10 / §A.4)
- [x] Process Monitor — ``TrainingHealthPanel`` for selected ``train_`` / ``hpo_`` processes (§G.15 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-first pass)**

- [x] ``isTrainOrHpoProcess`` — shared train/HPO command matcher (Process Monitor parity)
- [x] Training Monitor — live health alerts for ``hpo_*`` processes alongside ``train_*`` (§G.17 / §A.4)
- [x] HPO Tracker — ``TrainingHealthPanel`` during live ``hpo_*`` runs; bridges §A.4 Option D trial health table (§G.18 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-second pass)**

- [x] Experiment Tracker — ``TrainingHealthPanel`` during live ``hpo_*`` runs (§G.18 / §A.4)
- [x] Training Hub — ``liveTrainProcessLabel`` for Live HPO header; ``HPO Tracker →`` shortcut during live HPO (§G.10 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-third pass)**

- [x] Training Hub — ``Experiment Tracker →`` shortcut during live HPO; ``Process Monitor →`` label parity (§G.10 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-fourth pass)**

- [x] Training Monitor / Process Monitor / HPO Tracker / Experiment Tracker — ``Training Hub →`` shortcut during live train/HPO workflows (§G.10 / §G.15 / §G.17 / §G.18 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-fifth pass)**

- [x] ``LiveTrainProgressBar`` — epoch/trial progress bar + elapsed + ETA on Training Hub, Training Monitor, HPO Tracker, and Experiment Tracker during live runs; shared ``processProgress.ts`` helpers (§D.2 / §G.10 / §G.17 / §G.18 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-sixth pass)**

- [x] Process Monitor — ``LiveTrainProgressBar`` replaces inline ``PROGRESS:`` row bar; elapsed + ETA parity on all running processes (train/hpo/sim/data gen) (§D.2 / §G.15 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-seventh pass)**

- [x] Simulation Launcher — ``LiveTrainProgressBar`` in live status panel during running ``test_sim`` processes (§D.2 / §G.9 / §A.4)
- [x] Data Generation Wizard — ``LiveTrainProgressBar`` in live progress panel during ``gen_data`` runs (§D.2 / §G.11 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-eighth pass)**

- [x] Evaluation Runner — live progress panel with per-checkpoint ``LiveTrainProgressBar`` during ``eval`` runs; multi-checkpoint aggregate status header + stdout tail (§D.2 / §G.12 / §A.4)

**Status**: §A.4 Options A+D complete — Options B/C (PyHessian, loss landscape PNG) deferred.

---

### §A.5 — HPO Analytics: Cross-Trial Visualizer

**Pain**: The HPO module supports Optuna, Ray Tune, and DEHB, but the results are stored as trial databases without a unified post-hoc analysis view. Users cannot easily compare hyperparameter importance or visualize Pareto frontiers across objectives.

**Options**

- **A** — Use `optuna.visualization` (already a transitive dependency) to render parallel-coordinates and importances plots; export to `assets/hpo_reports/`. `[Quick Win]`
- **B** — Add a dedicated HPO Analysis panel in the Studio (§G Phase 10) wrapping the Optuna visualization calls in a native WebView or exporting Plotly HTML to be rendered inline.
- **C** — Export all trial results to a Pandas DataFrame; add a `hpo_summary.ipynb` notebook template that loads and plots them.
- **D** — Integrate SHAP to compute hyperparameter contribution scores across trials. `[Research]`

**Recommendation**: **Option A** for immediate wins (one function call with Optuna's built-in plotting), **Option C** as the notebook companion for sharing results.

**Effort × Impact**: Very Low effort (Option A) / Medium impact

**Delivered (§A.5 Option A — hundred-tenth pass)**

- [x] ``hpo_reports.py`` — ``optuna.visualization`` parallel-coordinates, param-importances, and optimisation-history Plotly HTML (+ optional PNG when kaleido present) under ``assets/hpo_reports/<study>_<timestamp>/``
- [x] ``manifest.json`` per export with study metadata, artefact list, and plot errors
- [x] ``run_hpo_sim`` post-run hook auto-exports reports after fANOVA analysis
- [x] Rust ``export_optuna_reports`` command; HPO Tracker **Export Plotly** + **Reports** folder open (§G.18 bridge)
- [x] Unit tests in ``logic/test/unit/pipeline/simulations/test_hpo_reports.py``

**Status**: §A.5 Option A complete — Option B (Studio WebView inline Plotly) largely superseded by ECharts HPO Tracker; Option C (notebook template) deferred.

---

### §A.6 — Causal Simulation Failure Analysis

**Pain**: When a simulation day ends with overflows or negative profit, the root cause (fill-rate spike, capacity miscalculation, policy sub-optimality) is not automatically identified. Post-hoc debugging requires re-reading JSON logs line by line.

**Options**

- **A** — Add a `FailureAnalyzer` class to `logic/src/pipeline/simulations/` that, after each day, compares predicted vs. actual bin fill levels, flags bins that caused overflow, and writes a structured summary to the day's JSON log entry.
- **B** — Build a counterfactual engine: re-run the day with the optimal policy (Gurobi) whenever a heuristic fails, and log the gap. `[Research]`
- **C** — Visualize the failure mode as a route-diff overlay in the Studio geospatial view (§G Phase 3): bins that were skipped vs. bins that overflowed highlighted in red. Depends on §A.1.
- **D** — Use causal inference (DoWhy) to identify which features (fill_rate, capacity, graph_size) most predict failure across simulation episodes. `[Research]`

**Recommendation**: **Option A** is purely additive and requires no new dependencies — pure logic in the existing simulator. **Option C** is the natural follow-on once §G Phase 3 is implemented.

**Effort × Impact**: Medium effort / High impact

**Delivered (§A.6 Option A — hundred-twelfth pass)**

- [x] ``FailureAnalyzer`` — post-day root-cause analysis comparing predicted vs. actual fill, flagging overflow bins, fill-rate spikes, and skipped high-fill bins
- [x] ``failure_emit.py`` — ``SIM_FAILURE_START:`` stdout marker + JSONL append; embedded ``failure_analysis`` in ``GUI_DAY_LOG_START`` payloads
- [x] ``LogAction`` — runs analyzer after each day; attaches summary to daily log dict
- [x] Rust ``parse_sim_failure_line`` + ``load_sim_failure_log`` command; ``sim:failure_update`` watcher events
- [x] Studio ``FailureAnalysisPanel`` — severity-coded causes, overflow bin table, skipped high-fill chips on Simulation Monitor
- [x] Unit tests in ``logic/test/unit/pipeline/simulations/test_failure_analyzer.py``

**Delivered (§A.6 Option C — hundred-fifteenth pass)**

- [x] ``routeFailureOverlay.ts`` — shared overflow/skipped bin id extraction + tour-diff sets for multi-policy compare
- [x] ``FailureOverlayLegend`` — reusable legend for overflow (red), skipped high-fill (orange), and tour-diff rings
- [x] ``DeckRouteMap`` — failure highlight ``ScatterplotLayer`` on Mercator + OrbitView; tour-diff ring overlay when two policies compared in overlay layout
- [x] Simulation Monitor — **Show/Hide failure overlay** + **Show/Hide route diff** toggles; wired to deck.gl and ECharts ``RouteViz`` (failure colours via embedded ``failure_analysis``)
- [x] ``RouteViz`` — legend when failure bins present; ``routeViz.ts`` uses shared overlay helper

**Delivered (§A.6 Option C — hundred-sixteenth pass)**

- [x] ``routeViz.ts`` — ``showFailureOverlay`` toggle; dual-policy overlay paths; tour-diff ring borders via ``TOUR_DIFF_RGB`` on ECharts scatter nodes
- [x] ``RouteViz`` — ``compareData`` / ``showTourDiff`` props; combined ``FailureOverlayLegend`` for failure + diff modes
- [x] Simulation Monitor — ECharts overlay compare when two map policies visible; failure + route-diff toggles propagate to ``RouteViz`` (parity with deck.gl)
- [x] Simulation Summary — **Show/Hide failure overlay** + **Show/Hide route diff** toggles; overlay-compare ``RouteViz`` when exactly two brushed policies share a day

**Status**: §A.6 Options A+C complete — Options B/D (counterfactual engine, DoWhy) deferred.

---

### Effort × Impact Matrix — Analytics & Interpretability

| Item                                     | Effort    | Impact | Priority        |
| ---------------------------------------- | --------- | ------ | --------------- |
| §A.3 Option A (PolicyVizMixin → Studio)  | Very Low  | High   | P0 ✅            |
| §A.3 Option B (2 Hz live telemetry stream) | Low    | High   | P1 ✅            |
| §A.3 Option C (SQLite cross-run trending) | Low    | Medium | P2 ✅            |
| §A.5 Option A (Optuna plots)             | Very Low  | Medium | P0 ✅            |
| §A.4 Option A (TrainingHealthCallback)   | Low       | High   | P1 ✅            |
| §A.4 Option D (HPO health prune metrics) | Low       | High   | P1 ✅            |
| §A.2 Option C (WandB attention heatmaps) | Low       | High   | P1 ✅            |
| §A.2 Option A (Studio attention ring-buffer) | Medium | High | P1 ✅            |
| §A.6 Option A (FailureAnalyzer)          | Medium    | High   | P1 ✅            |
| §A.6 Option C (route-diff overlay)       | Medium    | High   | P2 ✅            |
| §A.1 Option A (ECharts route viz)        | Medium    | High   | P2 ✅            |
| §A.1 Option E (deck.gl PathLayer)        | High      | High   | P2 ✅ (§G.3/§G.16) |
| §A.4 Option B (PyHessian)                | High      | Medium | P3 `[Research]` |
| §A.6 Option B (counterfactual engine)    | Very High | High   | P3 `[Research]` |

### §A — Analytics & Interpretability Complete ✅

All P0–P2 analytics bridges are delivered (§A.1–§A.6). Remaining items are research-grade extras (PyHessian, counterfactual engine, DoWhy, BertViz) or release-adjacent notebook templates (§A.5 Option C).

---

## B — Architecture

### §B.1 — Unit Test Coverage Uplift

**Pain**: The CI pipeline enforces a coverage threshold (60%), but 218 test files across `logic/test/` cover primarily the high-level pipeline and environment modules. Core sub-components (masking utilities in `boolmask.py`, tensor-dict protocol, individual attention modules) lack unit-level isolation tests.

**Options**

- **A** — Audit uncovered lines with `coverage report --show-missing`; write targeted parametric tests (`@pytest.mark.parametrize`) for the utility and module layers until coverage reaches 75%. `[Quick Win]`
- **B** — Add mutation testing (`mutmut`) to the CI pipeline to distinguish tests that merely execute code from those that actually detect bugs.
- **C** — Set per-module coverage floors in `.coveragerc` (e.g., `logic/src/utils/` ≥ 80%, `logic/src/models/modules/` ≥ 70%) to prevent regressions in well-tested modules while allowing lower thresholds in exploratory code.
- **D** — Generate property-based tests with Hypothesis for mathematical invariants (e.g., `boolmask` always masks depot, distance matrices are symmetric after construction).

**Recommendation**: **Option C** immediately (configuration change, no new tests needed), then **Option A** to fill gaps. **Option D** is a high-value investment for mathematical correctness guarantees.

**Effort × Impact**: Low effort (Options A/C) / High impact

---

### §B.2 — Benchmark Regression CI Gate

**Pain**: The CI only checks code quality and unit correctness. There is no automated performance baseline: a refactor that accidentally degrades inference speed by 30% or increases peak GPU memory will merge silently.

**Options**

- **A** — Add a `benchmark` job to `ci.yml` that runs `pytest --benchmark-only` (pytest-benchmark) on a small fixed dataset; compare against a stored baseline JSON; fail if any metric regresses > 10%.
- **B** — Use `asv` (airspeed velocity) for a more mature benchmark suite with statistical confidence and HTML reports. Higher setup cost.
- **C** — Track benchmark results as GitHub Actions artefacts and comment the delta on every PR using `github-action-benchmark`. `[Quick Win]` for visibility without hard failure gates.
- **D** — Run benchmarks only on `push` to `main` (not on every PR) to keep CI fast; store results in a `gh-pages` branch.

**Recommendation**: **Option C** first (adds visibility with minimal CI cost), then **Option A** to enforce regression gates once baseline values are stable.

**Effort × Impact**: Low effort / High impact

---

### §B.3 — Policy Plugin System

**Pain**: Adding a new classical policy (e.g., a new metaheuristic) requires modifying multiple files: the policy registry, the CLI argument parser, the Studio dropdown list, and the simulation runner. There is no single registration point.

**Options**

- **A** — Define a `@register_policy(name, problem_types)` decorator that writes to a module-level dict in `logic/src/policies/__init__.py`; CLI and Studio query this dict at runtime. `[Quick Win]`
- **B** — Use Python entry points (`pyproject.toml` `[project.entry-points]`) for full plugin isolation; external packages can register policies without modifying the core codebase.
- **C** — Use a YAML-driven policy manifest (`assets/configs/policies.yaml`) that maps names to fully-qualified class paths; load via `importlib`.
- **D** — Use Hydra's `_target_` instantiation pattern (already in use for models) to register and instantiate policies, achieving consistency with the existing config system.

**Recommendation**: **Option D** is the most architecturally consistent choice given that Hydra is already the config backbone. **Option A** is a useful quick bridge while Option D is designed.

**Effort × Impact**: Medium effort / High impact

---

### §B.4 — Structured Logging Consolidation

**Pain**: The codebase has three parallel logging mechanisms: Python's `logging` module with a `logstash` handler and JSON formatter (in `logic/src/tracking/logging/`), `print()` statements scattered throughout model code (380 files mix both), and the simulation's own JSON file output. This makes log aggregation and filtering inconsistent.

**Options**

- **A** — Run `grep -rn "print(" logic/src/ | grep -v test | grep -v "#"` to enumerate all non-test print calls; replace with `logger.debug()` or `logger.info()`. `[Quick Win]`
- **B** — Introduce `structlog` as a unified structured logging backend; all existing `logging.getLogger()` calls are wrapped by a `structlog.BoundLogger`.
- **C** — Add a `LoggingConfig` dataclass to the Hydra config tree controlling per-module log levels, output sinks (file, stdout, logstash), and JSON vs. plain format — without changing any log call sites.
- **D** — Integrate OpenTelemetry tracing for end-to-end span propagation across training → evaluation → simulation pipeline stages.

**Recommendation**: **Option A** is immediate hygiene. **Option C** gives operators control without touching 380 files. **Option B** is the right long-term architecture once the volume justifies it.

**Effort × Impact**: Low–Medium effort / Medium impact

---

### §B.5 — Type Safety Migration: Strict MyPy

**Pain**: MyPy runs in CI but with `continue-on-error: true`, meaning type errors are never blocking. The 379 typed files use inconsistent annotation patterns, and the complex tensor-dict protocol in `logic/src/interfaces/` is only partially typed.

**Options**

- **A** — Enable `--strict` mode for a well-contained subpackage first (`logic/src/utils/`); fix all errors there; expand gradually. Remove `continue-on-error` for that subpackage. `[Quick Win]`
- **B** — Add `py.typed` marker and ship inline type stubs for the `logic` package.
- **C** — Use `pyright` (faster, better PyTorch generics support) alongside MyPy; make pyright the blocking check and MyPy the advisory check (both pyright and pylance can be used as the engine for type checking using Pyrefly).
- **D** — Use `beartype` for runtime type enforcement at public API boundaries (interfaces module). Catches issues that static analysis misses.

**Recommendation**: **Option A** for gradual strictness adoption; **Option D** as a runtime safety net for the interfaces layer where type errors cause silent mathematical bugs.

**Effort × Impact**: Medium effort / High impact

---

### §B.6 — Environment Plugin System (Analogous to §B.3)

**Pain**: Adding a new problem environment (e.g., a new VRP variant) requires modifying `logic/src/envs/problems.py`, the CLI parser, the data generator, and the Studio environment selector — no single registration point.

**Options**

- **A** — Define a `@register_env(name, problem_class)` decorator and a central env registry; CLI/Studio consult it at startup.
- **B** — Use Hydra `_target_` pattern: each env is a config group entry under `conf/env/`, instantiated via `hydra.utils.instantiate()`. Fully consistent with existing model instantiation.
- **C** — Define a `ProblemManifest` dataclass that each env module exports; a loader discovers them via `importlib.metadata`.

**Recommendation**: **Option B** — already the pattern for models; extending it to environments achieves full symmetry across the config system.

**Effort × Impact**: Medium effort / High impact

---

### §B.7 — Circular Import Prevention

**Pain**: With 1,825 Python files, implicit inter-module dependencies are likely. Circular imports surface at runtime as `ImportError` or `AttributeError` and are hard to track down post-hoc.

**Options**

- **A** — Add `pydeps` to CI (`uv run pydeps logic/src --max-bacon 3 --no-show`) to generate a dependency graph; fail if cycles are detected. `[Quick Win]`
- **B** — Enforce import order via `isort` + `ruff` rules `I` (already partially configured); add a custom `ruff` rule that flags cross-layer imports (logic → gui).
- **C** — Introduce `__all__` definitions in every `__init__.py` to make the public surface explicit and prevent accidental internal imports.

**Recommendation**: **Option A** for automated detection, **Option B** for prevention. Both are low-cost additions to CI.

**Effort × Impact**: Very Low effort / Medium impact

---

### §B.8 — Async Task & Worker Standardization

**Pain**: The existing PySide6 GUI background workers (`data_loader_worker.py`, `chart_worker.py`, `file_tailer_worker.py`) each implement `QThread` independently with inconsistent error propagation, progress signal patterns, and cancellation logic. During the Tauri migration (§G), these Qt-specific workers are replaced; however, the Python logic layer still spawns background operations (data loading, simulation orchestration, training) that need consistent cancellation and progress-reporting contracts.

**Options**

- **A** — For the transitional PySide6 GUI: define a `BaseWorker(QThread)` in `gui/src/helpers/base_worker.py` with: `progress = Signal(int)`, `error = Signal(str)`, `result = Signal(object)`, `_cancelled: bool`, and a `cancel()` method. Subclasses override `run_task()`. Superseded once §G Phase 15 is complete. `[Quick Win]`
- **B** — For the Tauri backend: define a Rust `AsyncTask` trait with `run()`, `cancel()`, and a `progress_channel: Sender<f32>`. All long-running Rust commands implement this trait; progress events are forwarded to the frontend via Tauri's event system.
- **C** — For the Python logic layer: introduce a `BackgroundTask` protocol class with `run()`, `cancel()`, `on_progress(callback)` methods used consistently across simulation, training, and data generation entry points. The Tauri backend's Rust layer calls the Python subprocess and receives structured progress lines from stdout.
- **D** — Use Python `concurrent.futures.ThreadPoolExecutor` managed by a Rust-aware bridge class that maps futures to Tauri async commands.

**Recommendation**: **Option A** for the PySide6 transitional period, **Option B + C** for the Tauri architecture. The Rust trait (B) standardizes the Tauri command layer; the Python protocol (C) standardizes the subprocess-facing API so Rust can stream progress without format-specific parsing.

**Effort × Impact**: Low effort (Option A) / Medium effort (B + C) / High impact

---

### Effort × Impact Matrix — Architecture

| Item                                                    | Effort   | Impact | Priority                  |
| ------------------------------------------------------- | -------- | ------ | ------------------------- |
| §B.7 Option A (pydeps CI)                               | Very Low | Medium | P0 `[Quick Win]`          |
| §B.1 Option C (per-module coverage floors)              | Very Low | High   | P0 `[Quick Win]`          |
| §B.4 Option A (remove print() calls)                    | Low      | Medium | P0 `[Quick Win]`          |
| §B.8 Option A (BaseWorker, transitional)                | Low      | Medium | P1 `[Quick Win]`          |
| §B.5 Option A (strict MyPy, utils subpackage)           | Medium   | High   | P1                        |
| §B.2 Option C (benchmark visibility)                    | Low      | High   | P1                        |
| §B.8 Option B+C (Tauri async trait + Python protocol)   | Medium   | High   | P2 `[Blocked]` §G Phase 0 |
| §B.3 Option D (Hydra policy plugin)                     | Medium   | High   | P2                        |
| §B.6 Option B (Hydra env plugin)                        | Medium   | High   | P2                        |
| §B.1 Option D (Hypothesis property tests)               | High     | High   | P2 `[Research]`           |
| §B.2 Option A (benchmark regression gate)               | Medium   | High   | P2                        |

---

## C — Documentation

### §C.1 — API Reference Docs (mkdocstrings + MkDocs Material)

**Pain**: The module docs in `docs/` are hand-written Markdown files that describe architecture but do not reflect live code. Developers must read source files to find parameter names, return types, and class hierarchies. There is no search-indexed API reference.

**Options**

- **A** — Add `mkdocs` + `mkdocs-material` + `mkdocstrings[python]` as dev dependencies; configure `mkdocs.yml` to auto-generate API pages from existing docstrings. `[Quick Win]`
- **B** — Use `sphinx` + `sphinx-autodoc` + `furo` theme; more established but higher configuration overhead.
- **C** — Use `pdoc` for a zero-configuration auto-generated HTML reference; simpler but less feature-rich.
- **D** — Generate docs only for the public-facing `logic/src/interfaces/` layer; leave internal modules undocumented.

**Recommendation**: **Option A** — MkDocs Material is the modern standard, integrates well with GitHub Pages, and the `.nav` configuration can include the existing hand-written `docs/` pages alongside auto-generated API pages.

**Effort × Impact**: Medium effort / High impact

---

### §C.2 — Enforce Docstring Coverage with `pydoclint`

**Pain**: Public functions in the interfaces and models layers often lack docstrings, or have docstrings that omit parameter types/descriptions. MyPy catches type errors but not documentation gaps.

**Options**

- **A** — Add `pydoclint` to the `pre-commit` hooks and CI `quality-checks` job; fail on missing/mismatched docstrings for public functions. `[Quick Win]`
- **B** — Use `interrogate` (simpler, counts docstring presence percentage) as a softer gate.
- **C** — Configure `ruff` rule `D` (pydocstyle) — already partially available in ruff — for inline enforcement without a separate tool.

**Recommendation**: **Option C** — ruff is already the linter; adding the `D` rule family requires only a config line and keeps the toolchain minimal.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

---

### §C.3 — CHANGELOG.md

**Pain**: There is no structured changelog. Contributors cannot tell what changed between training runs or what API breaks occurred across model versions.

**Options**

- **A** — Adopt `Keep a Changelog` format (`CHANGELOG.md` at repo root); commit an initial entry retroactively from `git log`. `[Quick Win]`
- **B** — Use `git-cliff` to auto-generate the changelog from conventional commit messages; integrate into the CI release job.
- **C** — Use GitHub Releases with auto-generated release notes from PR labels.

**Recommendation**: **Option A** first (manual, immediate), then **Option B** to automate future entries once contributors adopt conventional commits.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

---

### §C.4 — Architecture Diagrams as Code (Mermaid)

**Pain**: `docs/ARCHITECTURE.md` describes the system in prose. There are no visual diagrams showing the data flow from CLI → Pipeline → Environment → Model → Policy, or the Studio mediator pattern, making onboarding slow.

**Options**

- **A** — Embed Mermaid flowcharts directly in `docs/ARCHITECTURE.md`; GitHub renders them natively in Markdown. Add: training data flow, inference pipeline, simulation orchestration, Studio architecture diagrams. `[Quick Win]`
- **B** — Use `diagrams` (Python-as-code diagram library) to generate PNG architecture diagrams; commit the PNGs and Python sources.
- **C** — Use PlantUML for class diagrams of the interfaces layer; integrate into the MkDocs build (depends on §C.1).

**Recommendation**: **Option A** for immediate diagrams (zero tooling overhead, GitHub-native), **Option C** for the interfaces class diagram once §C.1 is set up.

**Effort × Impact**: Low effort / High impact `[Quick Win]`

---

### §C.5 — Jupyter Notebook Tutorials

**Pain**: There are no interactive examples showing how to: generate a VRPP instance, run inference with a trained AM model, compare ALNS vs. Gurobi on a benchmark instance, or load and visualize simulation results. Researchers must read source code to reproduce even basic experiments.

**Options**

- **A** — Add `notebooks/` directory with: `01_getting_started.ipynb`, `02_train_am_vrpp.ipynb`, `03_compare_policies.ipynb`, `04_simulation_analysis.ipynb`. Use the existing `main.py` API internally.
- **B** — Add `nbval` to CI to execute notebooks and validate outputs; prevents notebooks from rotting.
- **C** — Host interactive notebooks on Binder or Google Colab (badge in README).

**Recommendation**: **Option A** as the content investment, **Option B** to keep them passing. **Option C** is optional polish.

**Effort × Impact**: High effort / High impact

---

### §C.6 — Troubleshooting & Compatibility Docs Refresh

**Pain**: `docs/TROUBLESHOOTING.md` and `docs/COMPATIBILITY.md` exist but their content is unclear. CUDA version conflicts, Gurobi license errors, and display backend issues are the most common friction points for new contributors. With the Tauri migration, new Studio-specific setup steps (Rust toolchain, Node.js, Tauri CLI) must also be documented.

**Options**

- **A** — Audit both files; add sections for: Gurobi 11+ license setup, CUDA 12.x / PyTorch 2.2 compatibility matrix, `uv sync` common errors, HGS/PyVRP installation issues, and Tauri/Rust toolchain setup (`cargo tauri dev` prerequisites, Node.js version requirements).
- **B** — Add a `scripts/diagnose.sh` script that checks all critical dependencies and prints a structured health report.

**Recommendation**: **Option A + B** in parallel — one improves static docs, the other gives developers a live diagnostic tool.

**Effort × Impact**: Low effort / Medium impact

---

### §C.7 — CI Documentation Pipeline

**Pain**: Documentation is never validated in CI. A broken import in a module will silently prevent `mkdocstrings` from generating its API page; typos in Mermaid diagrams break rendering without any build error.

**Options**

- **A** — Add a `docs` job to `ci.yml` that runs `mkdocs build --strict`; fail on any warning. Depends on §C.1.
- **B** — Run `mkdocs gh-deploy` automatically on push to `main`, making the live docs always reflect the latest commit.
- **C** — Use `pre-commit` hooks to validate Mermaid syntax locally before commit.

**Recommendation**: **Option A** (blocking build check) + **Option B** (auto-deploy) once §C.1 is implemented.

**Effort × Impact**: Low effort / High impact (after §C.1)

---

### Effort × Impact Matrix — Documentation

| Item                                    | Effort   | Impact | Priority         |
| --------------------------------------- | -------- | ------ | ---------------- |
| §C.3 Option A (CHANGELOG.md)            | Very Low | Medium | P0 `[Quick Win]` |
| §C.2 Option C (ruff D rules)            | Very Low | Medium | P0 `[Quick Win]` |
| §C.4 Option A (Mermaid diagrams)        | Low      | High   | P0 `[Quick Win]` |
| §C.6 Option A (TROUBLESHOOTING refresh) | Low      | Medium | P1               |
| §C.1 Option A (MkDocs Material)         | Medium   | High   | P1               |
| §C.7 Option A (docs CI job)             | Low      | High   | P2 (after §C.1)  |
| §C.5 Option A (Jupyter notebooks)       | High     | High   | P2               |
| §C.5 Option B (nbval CI)                | Low      | High   | P2 (after §C.5)  |

---

## D — GUI / UX

> **Context**: The existing PySide6/Qt GUI (`gui/src/`) is being migrated to WSmart-Route Studio, a Tauri 2.0 application (§G). The requirements in this section remain valid; the implementation guidance is updated to reflect the new Tauri/React/TypeScript stack. All references to Qt-specific APIs (QApplication, QThread, QSettings, QWidget subclasses, etc.) have been replaced with their Tauri/React equivalents.

---

### §D.1 — Route Visualization Panel

**Pain**: The Studio's analysis views show dataset statistics and fill-rate charts, but have no panel for visualizing computed routes. After running a simulation or evaluation, users must read JSON output to understand what routes were computed.

**Options**

- **A** — Add a `RouteViz` React component in the Studio using ECharts `custom` series or a 2D `<canvas>` renderer: plot depot (star), customer nodes (circles sized by demand), and route edges (colour per vehicle). Load routes from simulation JSON output. Synergises with §A.1.
- **B** — Use the deck.gl `PathLayer` + `ScatterplotLayer` already integrated for the geospatial phase (§G Phase 3) in Cartesian OrbitView mode — repurpose the same renderer for abstract coordinate systems.
- **C** — Open routes in an external browser tab via a locally-served Plotly map each time the user clicks "Visualize". Breaks the desktop-app UX.

**Recommendation**: **Option A** immediately (pure React, no additional dependencies), **Option B** as the production upgrade once §G Phase 3 (deck.gl) is in place.

**Effort × Impact**: Medium effort / High impact

---

### §D.2 — Training Progress Enhancements

**Pain**: The Studio's training launcher shows progress via a streamed log view (reading subprocess stdout), but the UX is a plain text area. There is no live loss curve, no epoch progress bar, and no ETA display.

**Options**

- **A** — Parse the structured JSON log emitted by the training pipeline inside the Rust backend; forward parsed metric events to React via Tauri's event system (`emit` / `listen`). Update: a live ECharts line chart (loss / reward curves), a `<progress>` element for epoch progress, and a computed ETA label. The file-watch approach in §G Phase 15 (Real-Time Process Monitor) provides the streaming infrastructure. `[Quick Win]` for the progress bar; more work for the live chart.
- **B** — Add a Rust `TrainingMetricsWatcher` that watches the WandB run directory for new log entries and forwards them to the React frontend as Tauri events.
- **C** — Embed the live WandB dashboard URL in a Tauri WebView panel. Requires an active WandB connection.

**Recommendation**: **Option A** — parse structured logs (already JSON-formatted) via the Tauri file-watch event system. Zero external dependency; consistent with §G Phase 15 infrastructure.

**Effort × Impact**: Medium effort / High impact

---

### §D.3 — Dark / Light Theme Toggle

**Pain**: The Studio uses a fixed dark theme. There is no runtime toggle exposed to the user, and the system theme preference is not respected.

**Options**

- **A** — Implement a theme toggle in the Studio's settings panel using Tailwind CSS `dark:` variant classes combined with a root `data-theme` attribute toggled via React state. Persist selection to `localStorage`. `[Quick Win]`
- **B** — Use the Tauri Store plugin (`@tauri-apps/plugin-store`) to persist the theme preference so it is restored across app restarts. Synergises with §D.4.
- **C** — Add a system-theme-following mode using the CSS `prefers-color-scheme` media query; detect the system preference on startup and switch automatically.

**Recommendation**: **Option A + B** together — both are trivial once Tailwind dark mode is configured, and Store plugin persistence is a one-liner.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

---

### §D.4 — Session Persistence

**Pain**: When the Studio is closed and re-opened, all configured parameters (problem type, model path, dataset path, number of days, policy selections) are reset to defaults. Users must reconfigure every session.

**Options**

- **A** — Persist the current form state (all input values, selected options) using Zustand's `persist` middleware with `localStorage` as the storage backend. Restore on app mount. `[Quick Win]`
- **B** — Use the Tauri Store plugin (`@tauri-apps/plugin-store`) for cross-platform native key-value persistence — writes to an OS-appropriate config directory rather than browser storage. More robust than localStorage for a desktop app.
- **C** — Allow users to name and save multiple "session profiles" (e.g., "VRPP-50-nodes", "WCVRP-simulation") and switch between them from a dropdown.

**Recommendation**: **Option B** first (idiomatic Tauri, writes to a proper config path), **Option C** for power users.

**Effort × Impact**: Low effort / High impact

---

### §D.5 — Progress & Cancellation for Long Operations

**Pain**: Data generation, training, and simulation runs can take hours. Users have no mechanism to cancel a running operation without force-quitting the app, and there is no progress indicator for operations that don't emit epoch-level logs.

**Options**

- **A** — Add a cancel mechanism: the Rust backend spawns each long-running Python process via `tokio::process::Command`; a `cancel` Tauri command sends SIGTERM (or Windows equivalent) to the child process. A React "Cancel" button invokes this command. Implements the Rust `AsyncTask` trait from §B.8 Option B.
- **B** — For multiprocessing-based operations (simulation uses `multiprocessing`), pass a cancellation flag via a shared file sentinel (`assets/.cancel_flag`) that the Python side polls; the Rust backend creates/removes the file on cancel request.
- **C** — Show a React modal progress dialog for operations with known total steps; show an indeterminate spinner for open-ended operations. Subscribe to Tauri progress events (emitted by §G Phase 15 infrastructure) to update the progress bar.

**Recommendation**: **Option A + C** — the Tauri command (A) provides the cancel mechanism; the React progress modal (C) provides the UX. Option B as a fallback for multiprocessing operations where SIGTERM does not propagate to worker processes.

**Effort × Impact**: Medium effort / High impact

---

### §D.6 — Configuration Panel for Hydra Overrides

**Pain**: The Studio exposes only a subset of available Hydra configuration options. Advanced users who want to override `train.batch_size`, `model.embedding_dim`, or `env.num_loc` must edit config files or use the CLI — bypassing the Studio entirely.

**Options**

- **A** — Add an "Advanced Overrides" collapsible section in each launcher panel (simulation, training, data gen) rendering a React table of key-value rows. Users can add/edit/delete rows; the Rust backend translates them to Hydra override strings (`key=value`) appended to the subprocess call. `[Quick Win]`
- **B** — Parse the Hydra config schema (via `OmegaConf.to_yaml`) at startup and generate a typed React form using `react-hook-form` + `zod` for validation: dropdowns for string enums, sliders for bounded numerics, checkboxes for bools.
- **C** — Embed a Monaco Editor YAML panel that is passed directly as a Hydra config override file — maximum power, minimal guardrails.

**Recommendation**: **Option A** for immediate usefulness (generic override table, one afternoon of work), **Option B** as the polished version once the config schema introspection is stable, **Option C** for expert users who prefer raw YAML access.

**Effort × Impact**: Medium effort / High impact

---

### §D.7 — Keyboard Shortcuts & Command Palette

**Pain**: All Studio operations require mouse clicks. Power users running repeated experiments have no keyboard-driven workflow.

**Options**

- **A** — Register global keyboard shortcuts in React using `react-hotkeys-hook` (in-window shortcuts) or `@tauri-apps/plugin-global-shortcut` (OS-level shortcuts): `Ctrl+R` (run), `Ctrl+.` (cancel), `Ctrl+1`–`Ctrl+9` (switch tabs), `Ctrl+S` (save config). Display shortcuts in a Help overlay. `[Quick Win]`
- **B** — Implement a command palette (`Ctrl+Shift+P`) as a floating React component (`cmdk` library or equivalent) backed by a registry of all Studio actions; filter by typing. Particularly useful as the Studio grows beyond 10 top-level views.

**Recommendation**: **Option A** first (one `useHotkeys` call per action), **Option B** once the Studio has more than one launcher and multiple analytics views.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

**Delivered (§D.7 — hundred-thirty-ninth pass)**

- [x] ``LauncherNavMesh`` — shared cross-page sim / data-gen / eval launcher navigation component; replaces duplicated shortcut buttons on Simulation Launcher, Data Generation Wizard, Evaluation Runner, and Process Monitor (§G.9 / §G.11 / §G.12 / §G.15)
- [x] ``launcherProcess.ts`` — shared ``isSimProcess`` / ``isGenDataProcess`` / ``isEvalProcess`` helpers for Process Monitor launcher workflow panels
- [x] Keyboard shortcuts ``L`` → Simulation Launcher, ``D`` → Data Generation, ``V`` → Evaluation Runner; help overlay updated (§D.7)

**Delivered (§D.7 — hundred-forty-first pass)**

- [x] ``LauncherNavMesh`` ``Output Browser →`` + ``Load in Eval Runner →`` on completed eval processes (§G.12 / §G.14 / §G.15)
- [x] Keyboard shortcuts ``B`` → Benchmark Analysis, ``O`` → Output Browser; help overlay updated (§D.7)

**Delivered (§D.7 — hundred-forty-third pass)**

- [x] ``outputRunPath.ts`` — derive assets/output run directory from process stdout ``.jsonl`` paths (§G.14 / §G.9 / §G.15)
- [x] ``LauncherNavMesh`` / ``TrainHpoNavMesh`` — ``outputRunPath`` prop sets ``pendingRunPath`` before navigating to Output Browser (§G.14 / §D.7)
- [x] Simulation Launcher + Data Generation — post-run Output Browser deep-links to the completed run when stdout contains a log path (§G.9 / §G.11 / §G.14)
- [x] Process Monitor — ``Output Browser →`` on completed ``test_sim`` / ``gen_data`` processes with run deep-link (§G.15 / §G.14)

**Delivered (§D.7 — hundred-forty-fourth pass)**

- [x] ``outputRunPath.ts`` — Hydra snapshot / pruned-config / ``assets/output`` path parsing as fallback when no ``.jsonl`` in stdout (§G.14 / §G.9 / §G.12 / §G.15)
- [x] ``trainingRunPath.ts`` + ``pendingTrainingRunPath`` — Training Monitor deep-link from completed train/HPO processes (§G.10 / §G.17 / §D.7)
- [x] Evaluation Runner + Process Monitor eval — ``outputRunPath`` deep-link parity on completed eval workflows (§G.12 / §G.14 / §G.15)

**Delivered (§D.7 — hundred-forty-fifth pass)**

- [x] ``findRecentHpoProcessId`` / ``findRecentTrainOrHpoProcessId`` — retain newest train/HPO process after completion for post-run panels (§G.17 / §G.18 / §D.7)
- [x] HPO Tracker + Experiment Tracker — post-run ``outputRunPath`` + ``trainingRunPath`` on ``TrainHpoNavMesh`` when HPO sweep completes (§G.18 / §G.14 / §G.17 / §D.7)
- [x] Training Monitor — post-run deep-link parity on live/recent train panel; auto-refresh + select completed run from ``trainingRunPath`` (§G.17 / §G.10 / §D.7)

**Delivered (§D.7 — hundred-forty-sixth pass)**

- [x] ``findRecentLauncherProcessId`` / ``findRecentEvalProcessIds`` — retain newest sim / data-gen / eval launcher processes after completion (§G.9 / §G.11 / §G.12 / §D.7)
- [x] ``findRecentTrainProcessId`` — train-only recent process helper for Training Hub train mode (§G.10 / §D.7)
- [x] Simulation Launcher + Data Generation + Evaluation Runner + Training Hub — live/post-run panels rehydrate from ``useProcessStore`` when navigation clears local ``liveProcessId`` state (§G.9 / §G.10 / §G.11 / §G.12 / §D.7)

**Delivered (§D.7 — hundred-forty-seventh pass)**

- [x] ``trainingMetrics.ts`` — ``normalizeTrainingMetricRow`` exported for CSV + stdout parity (§G.17 / §G.10)
- [x] Training Monitor — post-run metrics/health/attention rehydrate from ``useProcessStore`` log lines; ``LIVE_KEY`` overlay chart persists after completion (§G.17 / §D.7)
- [x] HPO Tracker + Experiment Tracker — live metric snapshot row from persisted process stdout (§G.18 / §G.17 / §D.7)

**Delivered (§D.7 — hundred-forty-eighth pass)**

- [x] ``TrainingMetricSparklines`` — shared ``GradNormSparkline`` + ``LrSparkline`` + ``TrainingMetricSnapshot`` for train/HPO analytics panels (§G.15 / §G.17 / §G.18)
- [x] Process Monitor — train/HPO metrics rehydrate from ``useProcessStore``; grad-norm + LR sparklines persist after completion (§G.15 / §D.7)
- [x] HPO Tracker + Experiment Tracker — post-run grad-norm + LR sparklines from persisted HPO stdout (§G.18 / §G.17 / §D.7)

**Delivered (§D.7 — hundred-forty-ninth pass)**

- [x] Training Hub — post-run grad-norm + LR sparklines from persisted train/HPO stdout via ``TrainingMetricSparklines``; ``TrainingMetricSnapshot`` + rehydration banner (§G.10 / §D.7)
- [x] Training Monitor — deduplicated local sparklines; imports shared ``GradNormSparkline`` + ``LrSparkline`` + ``TrainingMetricSnapshot`` (§G.17 / §D.7)
- [x] §G.10 / §G.17 launcher + monitor post-run sparkline parity across all train/HPO workflow pages (§D.7)

**Delivered (§D.7 — hundred-fiftieth pass)**

- [x] ``postRunTrainingRehydrationMessage`` — shared post-run banner helper; mentions metrics, health alerts, and attention snapshots when rehydrated from ``useProcessStore`` (§G.10 / §G.15 / §G.17 / §G.18)
- [x] HPO Tracker + Experiment Tracker — deduplicated inline metric snapshot rows; import shared ``TrainingMetricSnapshot`` (§G.18 / §G.17 / §D.7)
- [x] Training Hub + Training Monitor + Process Monitor — post-run banner uses shared helper for health/attention rehydration parity (§G.10 / §G.15 / §G.17 / §D.7)
- [x] §G.18 / §G.17 analytics post-run snapshot + health/attention banner parity across all train/HPO workflow pages (§D.7)

**Delivered (§D.7 — hundred-fifty-first pass)**

- [x] ``TrainHpoAnalyticsStrip`` — shared snapshot + sparklines + health/attention + post-run banner component for train/HPO live panels (§G.10 / §G.15 / §G.17 / §G.18)
- [x] Training Hub + Process Monitor + HPO Tracker + Experiment Tracker — deduplicated inline analytics blocks; import shared ``TrainHpoAnalyticsStrip`` (§G.10 / §G.15 / §G.18 / §D.7)
- [x] Training Monitor — live/recent card uses ``TrainHpoAnalyticsStrip`` for post-run sparkline rehydration without ``LIVE_KEY`` selection (§G.17 / §D.7)
- [x] Training Hub + Training Monitor — ``metric updates`` label parity with Process Monitor / HPO / Experiment Tracker (§G.10 / §G.17 / §D.7)

**Delivered (§D.7 — hundred-fifty-second pass)**

- [x] Training Monitor — ``TrainHpoAnalyticsStrip`` receives rehydrated ``healthEntries`` + ``attentionEntries`` for post-run banner counts while page-level panels remain separate (``showHealthAttention={false}``) (§G.17 / §A.2 / §A.4 / §D.7)
- [x] Training Monitor — ``metric updates`` label on non-checkbox live/recent header when metrics are rehydrated from process store (§G.17 / §D.7)
- [x] Training Hub — ``metric updates`` label uses ``text-accent-success`` styling parity with Process Monitor / HPO / Experiment Tracker (§G.10 / §D.7)

**Delivered (§D.7 — hundred-fifty-third pass)**

- [x] ``TrainHpoRehydrationBadges`` — shared metric / health / attention count badges for train/HPO live panel headers (§G.10 / §G.15 / §G.17 / §G.18 / §A.2 / §A.4 / §D.7)
- [x] Training Hub + Process Monitor + Training Monitor + HPO Tracker + Experiment Tracker — deduplicated inline ``metric updates`` labels; header badges surface health alerts + attention snapshots when rehydrated from ``useProcessStore`` (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Training Monitor — checkbox live/recent header no longer shows ``0 metric updates`` when only health/attention are rehydrated (§G.17 / §A.2 / §A.4 / §D.7)

**Delivered (§D.7 — hundred-fifty-fourth pass)**

- [x] ``TrainHpoLivePanelHeader`` — shared status icon + title + process id + rehydration badges + ``TrainHpoNavMesh`` header row for train/HPO live panels (§G.10 / §G.15 / §G.17 / §G.18 / §A.2 / §A.4 / §D.7)
- [x] Training Hub — ``split`` layout + ``activity`` running icon parity via shared header (§G.10 / §D.7)
- [x] HPO Tracker + Experiment Tracker — deduplicated inline live HPO header blocks (§G.18 / §G.17 / §D.7)
- [x] Process Monitor — ``muted`` analytics subtitle header + badges-before-nav ordering parity (§G.15 / §D.7)

**Delivered (§D.7 — hundred-fifty-fifth pass)**

- [x] ``TrainHpoLivePanelHeader`` — ``overlaySelect`` prop for ``LIVE_KEY`` overlay-chart checkbox on Training Monitor (§G.17 / §A.2 / §A.4 / §D.7)
- [x] Training Monitor — deduplicated inline live/recent header blocks; shared status icon + title + process id + rehydration badges + ``TrainHpoNavMesh`` row (§G.17 / §D.7)
- [x] §G.10 / §G.15 / §G.17 / §G.18 train/HPO workflow header row parity across all five pages (§D.7)

**Delivered (§D.7 — hundred-fifty-sixth pass)**

- [x] ``TrainHpoLivePanel`` — shared header + ``LiveTrainProgressBar`` + ``TrainHpoAnalyticsStrip`` shell for train/HPO live/post-run panels (§G.10 / §G.15 / §G.17 / §G.18 / §A.2 / §A.4 / §D.7)
- [x] Training Hub + Process Monitor + Training Monitor + HPO Tracker + Experiment Tracker — deduplicated inline live panel card markup; ``card`` vs ``embedded`` variant parity (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Training Hub — ``footer`` process-id row + ``showAnalytics`` / ``analyticsWrapperClassName`` slots preserved via shared panel (§G.10 / §D.7)
- [x] Training Monitor — ``overlaySelect`` + ``showHealthAttention={false}`` analytics options preserved via shared panel (§G.17 / §A.2 / §A.4 / §D.7)
- [x] §G.10 / §G.15 / §G.17 / §G.18 train/HPO workflow live panel shell parity across all five pages (§D.7)

**Delivered (§D.7 — hundred-fifty-seventh pass)**

- [x] ``LauncherLivePanelHeader`` — shared status icon + title + ``LauncherNavMesh`` header row for sim / data-gen / eval launcher workflows (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] ``LauncherLivePanel`` — shared header + ``LiveTrainProgressBar`` + children shell with ``card`` vs ``embedded`` variant parity (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] Simulation Launcher + Data Generation + Evaluation Runner — deduplicated inline live progress card markup; ``navTrailing`` slot preserves sim auto-summary countdown (§G.9 / §G.11 / §G.12 / §D.7)
- [x] Process Monitor — ``embedded`` variant for selected ``test_sim`` / ``gen_data`` / ``eval`` analytics sections; run-label + live suffix parity on sim panel (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] §G.9 / §G.11 / §G.12 / §G.15 launcher workflow live panel shell parity across all four pages (§D.7)

**Delivered (§D.7 — hundred-fifty-eighth pass)**

- [x] ``ProcessIdFooter`` — shared process-id footer row for launcher and train/HPO live panels (§G.9 / §G.10 / §G.11 / §G.12 / §D.7)
- [x] Simulation Launcher + Training Hub — deduplicated inline process-id footer markup; import shared ``ProcessIdFooter`` (§G.9 / §G.10 / §D.7)
- [x] Data Generation Wizard + Evaluation Runner — ``LauncherLivePanel`` ``footer`` process-id row parity with Simulation Launcher (§G.11 / §G.12 / §D.7)
- [x] Evaluation Runner — multi-checkpoint footer lists all ``displayProcessIds`` when batch eval is active (§G.12 / §D.7)
- [x] ``EvalResultKpiRow`` — shared cost / gap / time / policy KPI row for eval live panels (§G.12 / §G.15 / §D.7)
- [x] ``EvalResultCard`` — shared eval result card with ``Open in Analytics →`` for Process Monitor embedded eval section (§G.12 / §G.15 / §D.7)
- [x] Evaluation Runner — per-checkpoint live panel uses ``EvalResultKpiRow`` ``compact`` variant (§G.12 / §D.7)
- [x] §G.12 / §G.15 eval result KPI + footer parity across Evaluation Runner and Process Monitor (§D.7)

**Delivered (§D.7 — hundred-fifty-ninth pass)**

- [x] ``ProcessIdFooter`` — monitor-page footer parity: Training Monitor, HPO Tracker, Experiment Tracker, and Process Monitor embedded sections; process id removed from inline headers (§G.15 / §G.17 / §G.18 / §D.7)
- [x] Training Monitor + HPO Tracker + Experiment Tracker — ``TrainHpoLivePanel`` ``footer`` process-id row parity with Training Hub (§G.10 / §G.17 / §G.18 / §D.7)
- [x] Process Monitor — ``LauncherLivePanel`` + ``TrainHpoLivePanel`` embedded sections use ``ProcessIdFooter``; simplified analytics subtitles without inline process id (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] ``EvalCheckpointLiveCard`` — shared per-checkpoint live eval row with KPI, progress bar, and stdout tail (§G.12 / §D.7)
- [x] Evaluation Runner — deduplicated inline per-checkpoint live panel markup; import shared ``EvalCheckpointLiveCard`` (§G.12 / §D.7)
- [x] §G.10 / §G.15 / §G.17 / §G.18 train/HPO workflow footer parity across all five pages (§D.7)

**Delivered (§D.7 — hundred-sixtieth pass)**

- [x] ``processLogTail`` — shared stdout/stderr tail helper for live eval panels (§G.12 / §G.15 / §D.7)
- [x] Process Monitor — selected ``eval`` processes use ``EvalCheckpointLiveCard`` during live runs and while waiting for structured JSON; ``EvalResultCard`` retained on completion with metrics (§G.12 / §G.15 / §D.7)
- [x] Evaluation Runner — deduplicated inline log tail formatting; import shared ``processLogTail`` (§G.12 / §D.7)
- [x] §G.12 / §G.15 eval live checkpoint card parity across Evaluation Runner and Process Monitor (§D.7)

**Delivered (§D.7 — hundred-sixty-first pass)**

- [x] ``ProcessLogTail`` — shared stdout/stderr tail display component for launcher live panels (§G.11 / §G.12 / §G.15 / §D.7)
- [x] ``EvalCheckpointLiveCard`` — deduplicated inline log tail markup; import shared ``ProcessLogTail`` (§G.12 / §D.7)
- [x] Data Generation Wizard — deduplicated inline log tail markup; ``processLogTail`` + ``ProcessLogTail`` parity (§G.11 / §D.7)
- [x] Process Monitor — selected ``gen_data`` processes show ``ProcessLogTail`` in embedded workflow section (§G.11 / §G.15 / §D.7)
- [x] §G.11 / §G.12 / §G.15 launcher log tail display parity across Data Generation, Evaluation Runner, and Process Monitor (§D.7)

---

### §D.8 — Toast Notifications for Background Completions

**Pain**: When a training job or data generation task finishes in the background, there is no notification. Users must check the process monitor tab to see if the job completed.

**Options**

- **A** — Use a React toast library (`sonner` or `react-hot-toast`) for in-app notifications: auto-dismissing toasts in the bottom-right corner for job completion, failure, and warnings. Triggered by Tauri events from the process monitor. `[Quick Win]`
- **B** — Use the Tauri notification plugin (`@tauri-apps/plugin-notification`) to display a native OS notification when a job finishes and the Studio window is not in focus.
- **C** — Play an OS sound via the Tauri shell plugin on job completion.

**Recommendation**: **Option A + B** — the React toast for when the window is focused, Tauri native notification for when the user has switched away. Option C is optional polish.

**Effort × Impact**: Low effort / High impact

---

### Effort × Impact Matrix — GUI / UX

| Item                                        | Effort   | Impact | Priority                          |
| ------------------------------------------- | -------- | ------ | --------------------------------- |
| §D.3 Option A+B (theme toggle + persist)    | Very Low | Medium | P0 `[Quick Win]` ✅              |
| §D.3 Option C (system theme following)      | Very Low | Medium | P0 `[Quick Win]` ✅              |
| §D.7 Option A (keyboard shortcuts)          | Very Low | Medium | P0 `[Quick Win]` ✅ (incl. T/H/E/B/O train + L/D/V launcher workflow) |
| §D.4 Option B (Tauri Store persistence)     | Low      | High   | P0 ✅ (Zustand persist)           |
| §D.8 Option A+B (toast + OS notification)   | Low      | High   | P1 ✅ (toast + OS notification done) |
| §D.5 Option A+C (cancel + progress modal)   | Medium   | High   | P1 ✅ (cancel + progress bars)    |
| §D.2 Option A (live training charts)        | Medium   | High   | P1 ✅ (all launchers + monitors + eval progress/ETA) |
| §D.1 Option A (ECharts route panel)         | Medium   | High   | P2 ✅ (RouteViz + Summary)        |
| §D.6 Option A (override table)              | Medium   | High   | P2 ✅ (all launchers)             |
| §D.1 Option B (deck.gl PathLayer)           | High     | High   | P2 ✅ (§G.3 / §G.16)              |
| §D.6 Option B (typed config form)           | High     | High   | P3                                |

---

## E — New Features

### §E.1 — Multi-Problem Benchmarking Suite

**Pain**: Comparing neural models (AM, TAM, DDAM, MoE) against classical policies (ALNS, HGS, Gurobi) across all three problem types (VRPP, WCVRP, SCWCVRP) and multiple graph sizes requires manually running multiple `main.py eval` commands and aggregating CSV results by hand.

**Options**

- **A** — Add a `benchmark` subcommand to `main.py` that: runs a configurable matrix of (policy × problem × graph_size), collects metrics, and writes a unified `benchmark_report.csv` and Markdown table. `[Quick Win]`
- **B** — Integrate with `ray[tune]` sweep (already a dependency) to parallelize the benchmark matrix across CPU cores.
- **C** — Add a "Benchmark" tab to the Studio (synergises with §A.5) that configures the matrix via checkboxes and shows a live results table.

**Recommendation**: **Option A** for the CLI benchmark runner, **Option C** for Studio-accessible results.

**Effort × Impact**: Medium effort / High impact

---

### §E.2 — TSPLIB / Solomon Benchmark Instance Support

**Pain**: The framework generates synthetic instances internally but cannot load standard benchmark instances (TSPLIB95, Solomon C/R/RC, Christofides). This makes it impossible to compare against published results.

**Options**

- **A** — Add a `data/loaders/tsplib_loader.py` that parses `.vrp` / `.tsp` files (standard TSPLIB format) into the framework's `TensorDict` input format.
- **B** — Use the `tsplib95` Python library (pure-Python, no C dependency) as a parser backend; wrap its output in the framework's schema.
- **C** — Add a `gen_data` subcommand option `--source tsplib --instance pr76` that downloads and converts instances from the TSPLIB repository.

**Recommendation**: **Option B** is the fastest path (the library handles all edge cases in the format spec); **Option C** makes it accessible from the CLI in one command.

**Effort × Impact**: Medium effort / Very High impact `[Research]`

---

### §E.3 — REST API for Remote Inference

**Pain**: The framework has no HTTP interface. Integrating the routing engine into a larger fleet management system requires either subprocess calls or direct Python imports, both of which are fragile.

**Options**

- **A** — Add a `main.py serve` subcommand using `FastAPI` that exposes: `POST /solve` (accepts a problem instance JSON, returns a solution), `GET /health`, and `GET /models` (lists available weights). `[Research]`
- **B** — Use `Flask` for a simpler synchronous server with lower dependency overhead.
- **C** — Implement a `gRPC` interface for higher-throughput production use cases.
- **D** — Wrap in a Docker container with a `docker-compose.yml` for deployment.

**Recommendation**: **Option A** — FastAPI is the modern standard, its async design fits the non-blocking inference pattern, and it auto-generates OpenAPI docs. **Option D** is the natural packaging step after.

**Effort × Impact**: High effort / High impact

---

### §E.4 — Online Learning / Warm-Starting

**Pain**: In multi-day waste collection simulation, each new day presents a slightly different distribution of bin fill levels. The current pipeline re-runs inference from a static checkpoint with no adaptation mechanism.

**Options**

- **A** — Add a `warm_start` mode to the training pipeline: initialize from an existing checkpoint and fine-tune for N epochs on the current day's distribution before evaluating. `[Research]`
- **B** — Implement the `MetaRNN` (already exists in `logic/src/models/meta/`) as the online adapter: on each day, perform one or more gradient steps using the day's context as the meta-input.
- **C** — Use the contextual bandit module (already in `logic/src/pipeline/rl/meta/`) to select among a portfolio of pre-trained policies based on day context, without gradient updates.
- **D** — Implement reservoir sampling of "hard" instances encountered during simulation and periodically fine-tune on them.

**Recommendation**: **Option C** is the lowest-risk path (no gradient updates in production, just policy selection) and leverages existing code. **Option B** is the research-grade approach that the MetaRNN architecture was designed for.

**Effort × Impact**: High effort / Very High impact `[Research]`

---

### §E.5 — Real-World Data Integration (Smart Bin Sensors)

**Pain**: The WCVRP and SCWCVRP environments model bin fill rates stochastically, but there is no pipeline to ingest real sensor data (IoT fill-level readings) and use it to calibrate the stochastic parameters.

**Options**

- **A** — Add a `data/loaders/sensor_loader.py` that reads the CSV format defined in `CLAUDE.md §12.3` and converts it to the framework's bin fill tensor format; expose it via `gen_data --source sensor --file bins.csv`.
- **B** — Add a `calibration` subcommand that fits the stochastic fill-rate distribution parameters (mean, variance per bin) to historical sensor data using MLE.
- **C** — Integrate with MQTT/HTTP sensor APIs for live streaming fill-level updates during simulation.

**Recommendation**: **Option A + B** as a research pipeline; **Option C** only for production deployments where live sensor APIs are available.

**Effort × Impact**: Medium effort (Options A/B) / Very High impact

---

### §E.6 — LLM-Assisted Problem Instance Generation

**Pain**: Research teams need diverse problem instances to test policy robustness. Hand-crafting instance parameters (node clustering, demand distributions, time windows) is labour-intensive.

**Options**

- **A** — Add a `gen_data --mode llm_assisted` command that uses an LLM API to generate natural-language scenario descriptions, translates them to parameter overrides, and creates instances. `[Research]`
- **B** — Use a simple constraint-satisfaction generator with richer parameter coverage (clustered vs. random vs. mixed depot placement, heterogeneous demand distributions) without LLM involvement.
- **C** — Train a conditional generator (VAE or diffusion model) on existing instance distributions to sample novel but realistic instances. `[Research]`

**Recommendation**: **Option B** is the pragmatic choice — richer parameterization of the existing generator provides immediate value without LLM API costs. **Option C** is a research-grade addition for distribution-shift robustness studies.

**Effort × Impact**: Low effort (Option B) / High impact

---

### §E.7 — Cross-Environment Generalization (Zero-Shot Transfer)

**Pain**: Models trained on VRPP do not generalize to WCVRP without retraining. There is no evaluation protocol measuring zero-shot or few-shot transfer across problem types.

**Options**

- **A** — Add a `transfer_eval` subcommand that loads a checkpoint trained on problem A and evaluates it on problem B; logs a transfer performance gap metric. `[Research]`
- **B** — Adapt the `MetaRNN` / hypernet architecture to condition on problem-type embeddings, enabling a single model to handle multiple VRP variants. `[Research]`
- **C** — Use curriculum learning: train sequentially on VRPP → WCVRP → SCWCVRP with increasing difficulty; measure generalization at each stage.

**Recommendation**: **Option A** first (pure evaluation, no training changes), to establish the baseline gap. **Option B/C** follow once the gap magnitude is known.

**Effort × Impact**: Low effort (Option A) / High impact `[Research]`

---

### Effort × Impact Matrix — New Features

| Item                                               | Effort    | Impact    | Priority        |
| -------------------------------------------------- | --------- | --------- | --------------- |
| §E.6 Option B (richer instance generator)          | Low       | High      | P0              |
| §E.1 Option A (CLI benchmark runner)               | Medium    | High      | P1              |
| §E.5 Option A (sensor data loader)                 | Medium    | Very High | P1              |
| §E.7 Option A (transfer eval command)              | Low       | High      | P1 `[Research]` |
| §E.2 Option B+C (TSPLIB loader)                    | Medium    | Very High | P2              |
| §E.5 Option B (fill-rate calibration)              | Medium    | Very High | P2              |
| §E.4 Option C (contextual bandit policy selection) | Medium    | Very High | P2 `[Research]` |
| §E.3 Option A (FastAPI server)                     | High      | High      | P3              |
| §E.4 Option B (MetaRNN online adaptation)          | Very High | Very High | P3 `[Research]` |
| §E.6 Option C (conditional generator)              | Very High | High      | P3 `[Research]` |

---

## F — Performance

### §F.1 — Batched Neural Inference Optimization

**Pain**: The neural decoders (AM, TAM, DDAM, MoE) process instances sequentially during evaluation and simulation. When evaluating against a test set of 10,000 instances, inference time dominates. No `torch.compile()` or `torch.inference_mode()` wrappers are in place.

**Options**

- **A** — Wrap all `model.forward()` evaluation calls in `torch.inference_mode()` (context manager). Zero-code-change speedup of ~10-15% by disabling gradient tracking. `[Quick Win]`
- **B** — Apply `torch.compile(model, mode='reduce-overhead')` to the decoder at evaluation time (PyTorch 2.x); measure speedup on the target GPU.
- **C** — Implement a `BatchedInferenceEngine` that collects N problem instances and runs a single batched forward pass; current code may process one instance at a time during simulation.
- **D** — Export models to ONNX / TensorRT for production inference; 2-5× speedup on NVIDIA GPUs with quantization.

**Recommendation**: **Option A** immediately (one context manager call), **Option B** as the next step (PyTorch 2.2 compile is mature for attention models), **Option C** for the simulation loop specifically.

**Effort × Impact**: Very Low–Medium effort / High impact

---

### §F.2 — GPU Memory Management

**Pain**: The framework targets GPUs with 12-24 GB VRAM, but there is no systematic GPU memory profiling. Memory leaks during long training runs (e.g., growing activation buffers from `PolicyVizMixin` or incorrect `detach()` calls) are currently invisible.

**Options**

- **A** — Add `torch.cuda.memory_summary()` logging at the end of each epoch in the training callback; log peak allocated and reserved memory. `[Quick Win]`
- **B** — Use the existing `logic/src/tracking/profiling/memory.py` profiler to generate per-epoch memory traces and write them to `assets/profiling/`.
- **C** — Add `torch.cuda.reset_peak_memory_stats()` at the start of each training epoch to get accurate per-epoch peak measurements.
- **D** — Use `torch.utils.checkpoint` (gradient checkpointing) on the encoder layers to trade compute for memory on large instances (100+ nodes).

**Recommendation**: **Options A + C** as immediate monitoring (`[Quick Win]`); **Option B** for detailed profiling when a leak is suspected; **Option D** for scaling to larger instances.

**Effort × Impact**: Very Low effort (Options A/C) / High impact

---

### §F.3 — Test Suite Speed

**Pain**: With 218 test files, the full test suite may be slow. No test parallelization is configured (no `pytest-xdist` in the CI pipeline). Fast-marked tests (`@pytest.mark.fast`) are not run in isolation by default.

**Options**

- **A** — Add `pytest-xdist` (`-n auto`) to the CI test job; ensure test files are isolation-clean (no shared mutable global state). `[Quick Win]`
- **B** — Split CI into a `fast` job (runs `@pytest.mark.fast` tests on every push) and a `full` job (all tests, runs on PR merge or nightly).
- **C** — Profile the test suite with `pytest --durations=20` to identify the slowest 20 tests; optimize or mark them `@pytest.mark.slow`.
- **D** — Use `pytest-split` to distribute tests across multiple parallel CI runners for very large suites.

**Recommendation**: **Option C** first (identify the bottlenecks), then **Option B** (fast/full split), then **Option A** (parallelization) once isolation is confirmed.

**Effort × Impact**: Low effort / High impact

---

### §F.4 — Data Loading Optimization

**Pain**: Training data is loaded from `.pkl` files (pickled tensors). For large datasets (graph sizes 100-317), loading dominates the time-to-first-batch. There is no dataset caching, pinned memory, or prefetch worker configuration.

**Options**

- **A** — Switch from `.pkl` to `.pt` (torch.save) for dataset files; PyTorch's tensor serialization is faster and avoids Python's pickle deserialization overhead. `[Quick Win]`
- **B** — Enable `DataLoader(num_workers=4, pin_memory=True, persistent_workers=True)` in the training pipeline; this overlaps CPU data loading with GPU computation.
- **C** — Use `torch.utils.data.IterableDataset` with a streaming generator to avoid loading entire datasets into RAM for very large graph sizes.
- **D** — Pre-compute and cache distance matrices for training instances to avoid recomputation each epoch.

**Recommendation**: **Option B** is the highest-leverage single change (overlapped loading eliminates GPU idle time); **Option A** reduces load latency itself. Both are non-breaking changes.

**Effort × Impact**: Low effort / High impact

---

### §F.5 — Simulation Throughput: Shared Memory & Vectorization

**Pain**: The simulation engine uses `multiprocessing` with a `Manager()` lock and counter for shared metrics synchronization (`_lock`, `_counter`, `_shared_metrics` in `simulator.py`). Manager proxies have significant IPC overhead compared to `multiprocessing.shared_memory`.

**Options**

- **A** — Replace `Manager().dict()` metrics with `multiprocessing.shared_memory.SharedMemory` backed `numpy` arrays; eliminate the Manager proxy round-trip. `[Research]`
- **B** — Use `multiprocessing.Pool.starmap()` with a return-value accumulation pattern instead of shared memory; simpler but requires collecting all results at the end of each day.
- **C** — Vectorize single-day simulation using batched tensor operations on GPU (run N days in parallel as a batch); eliminates multiprocessing entirely for GPU-local simulations.
- **D** — Profile the simulation with `cProfile` to determine whether IPC or computation is the bottleneck before optimizing.

**Recommendation**: **Option D** first (profile before optimizing), then **Option B** as a simpler refactor if IPC is the bottleneck, **Option A** for maximum throughput, **Option C** for GPU-resident simulation.

**Effort × Impact**: Low effort (Option D) / High impact

---

### §F.6 — McCabe Complexity Reduction

**Pain**: CI enforces a McCabe complexity ceiling of 15 (`--max-complexity 15`). Functions in the BPC solver, ALNS destroy/repair operators, and simulation orchestration are likely near or above this threshold, causing CI noise.

**Options**

- **A** — Run `uv run ruff check . --select C90` to identify functions above threshold; refactor the top-10 most complex functions by extracting helper methods. `[Quick Win]`
- **B** — Lower the threshold progressively (15 → 12 → 10) over three sprints to drive continuous simplification.
- **C** — Exempt well-understood algorithmic functions (BPC pricing, ALNS operators) with `# noqa: C901` inline suppressions, with a comment explaining why the complexity is justified.

**Recommendation**: **Option A** for the worst offenders (extract helpers), **Option C** for mathematically-justified complexity that cannot be reduced without obscuring the algorithm.

**Effort × Impact**: Low effort / Medium impact `[Quick Win]`

---

### §F.7 — CUDA-Aware Tensor Operations Audit

**Pain**: The codebase uses `get_device()` for device management, but there may be silent CPU fallbacks when `tensor.to(device)` is called on already-CPU tensors or when tensor operations create new CPU tensors (e.g., via `.numpy()`, `item()`, or list comprehensions inside `forward()`).

**Options**

- **A** — Add a custom `ruff` rule (or `grep` CI check) that flags `.numpy()`, `.item()`, and `list()` calls inside `*.py` files under `logic/src/models/`; these operations force CPU synchronization and break CUDA graphs.
- **B** — Enable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` in the test environment and run the test suite to surface memory-related device mismatches.
- **C** — Use `torch.autograd.set_detect_anomaly(True)` during a profiling run to catch NaN/Inf values that indicate silent device or type mismatches.

**Recommendation**: **Option A** (static analysis check) prevents future regressions; **Option C** is the fastest diagnostic for an existing issue.

**Effort × Impact**: Low effort / High impact

---

### Effort × Impact Matrix — Performance

| Item                                       | Effort    | Impact    | Priority         |
| ------------------------------------------ | --------- | --------- | ---------------- |
| §F.1 Option A (inference_mode wrapper)     | Very Low  | High      | P0 `[Quick Win]` |
| §F.2 Option A+C (GPU memory logging)       | Very Low  | High      | P0 `[Quick Win]` |
| §F.6 Option A (complexity refactor top-10) | Low       | Medium    | P0 `[Quick Win]` |
| §F.3 Option C (profile test durations)     | Low       | High      | P1               |
| §F.4 Option B (DataLoader pinned memory)   | Low       | High      | P1               |
| §F.7 Option A (CPU sync audit)             | Low       | High      | P1               |
| §F.3 Option B (fast/full CI split)         | Low       | High      | P1               |
| §F.1 Option B (torch.compile)              | Medium    | High      | P2               |
| §F.4 Option A (pkl → pt format)            | Low       | Medium    | P2               |
| §F.5 Option D (simulation profiling)       | Low       | High      | P2               |
| §F.5 Option B (Pool.starmap refactor)      | Medium    | High      | P2               |
| §F.4 Option D (cache distance matrices)    | Medium    | High      | P2               |
| §F.1 Option D (TensorRT export)            | High      | Very High | P3               |
| §F.5 Option A (SharedMemory refactor)      | High      | High      | P3 `[Research]`  |
| §F.5 Option C (GPU-vectorized simulation)  | Very High | Very High | P3 `[Research]`  |

---

## G — WSmart-Route Studio

> WSmart-Route Studio is the Tauri 2.0 desktop application that replaces the existing PySide6 GUI and extends it with deep analytics visualization, geospatial routing replay, ML introspection, and an OLAP query interface. The §D section above defines the UX requirements these phases must satisfy. The Studio is the primary interface for all user-facing operations: launching simulations and training runs, generating data, executing scripts, browsing results, and performing post-hoc analysis.

**Technology Stack**

| Concern | Technology |
| --- | --- |
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
| Process management | tokio::process (Rust), Tauri event system |
| Config persistence | Tauri Store plugin |
| OS notifications | Tauri notification plugin |

---

### §G.0 — Phase 0: Foundation & Tooling ✅

**Goal**: Establish the project scaffold, dev environment, and data pipeline so all subsequent phases have a stable base.

- [x] Bootstrap Tauri 2.0 project (`app/src-tauri/` + React/TypeScript frontend); window 1600×1000, min 1200×700
- [x] Configure Tailwind CSS with dark theme defaults (`canvas-*` / `accent-*` palette) and `dark:` class toggle (§D.3)
- [x] Set up Rust backend with `tauri 2.0`, `tauri-plugin-{notification,store,dialog,shell}`, `serde`, `tokio`, `csv`, `anyhow`
- [x] Implement Tauri Store plugin setup for session and theme persistence (§D.3, §D.4)
- [x] `tools/app/justfile` — dev/build/check/clean commands; wired to root justfile as `just studio`, `just studio-build`, `just studio-install`
- [x] Arrow IPC schema for simulation log rows and Rust CSV → Arrow IPC stream: `commands/arrow.rs` — `csv_to_arrow_ipc`, `simulation_log_to_arrow_ipc` (typed KPI schema: policy/sample_id/day + profit/km/overflows/kg_per_km/…)
- [x] Spawn DuckDB-Wasm in a Web Worker; ingest Arrow IPC on CSV/log open: `duckdbClient.ts` + `useDuckDbInit` on app mount; Data Explorer + Simulation Monitor auto-ingest
- [x] Verify end-to-end latency: Settings "Run Arrow Pipeline Benchmark" + Data Explorer timing badge; 500 ms budget constant in `arrowPipeline.ts` (§G.0 partial — hardware baseline varies)
- [x] Arrow sidecar fast-path: `runCsvArrowPipeline()` + `runSimulationArrowPipeline()` prefer sibling ``.arrow`` IPC from extracted `.wsroute` bundles via `path_exists` + `runArrowSidecarPipeline()` (skips Rust CSV/JSONL re-parse; §G.0 / §G.8)
- [x] Portfolio DuckDB union: `runPortfolioSimulationArrowPipeline()` unions multiple JSONL logs into one table with `run_label`; `formatPipelineTimingBadge()` shared toolbar timing text (§G.0 / §G.1.4)

---

### §G.1 — Phase 1: Statistical Overview Dashboard (ECharts 2D)

**Goal**: Reproduce and extend the existing static `simulation_analysis.md` charts as interactive ECharts panels.

#### 1.1 KPI Summary Bar / Box Charts
- [x] Mean ± std overflows per constructor, grouped by mandatory-selection strategy: `GroupedMetricBarChart` overflows by selection strategy on Simulation Summary; portfolio mode swaps to overflows by city/scale across loaded runs (§G.1.1)
- [x] Mean ± std kg/km per constructor, grouped by city/scale: `GroupedMetricBarChart` kg/km by constructor on Simulation Summary; portfolio mode swaps to kg/km by city/scale across loaded runs (§G.1.1)
- [x] Grouped metric bar charts follow global ``logScale``: overflows groups use symlog y-axis; kg/km groups use log y-axis; error-bar whiskers via ``errorBarBounds`` when log on (§G.1.1 / §G.7)
- [x] Interactive brushing: selecting a bar cross-filters all panels on the dashboard: `PolicyBrushBar` + `toggleBrush` dims non-selected policies across all charts; `SimulationSummary` ingests log → DuckDB + `SqlQueryPanel` `brushSqlSync` / `brushedPoliciesSql` (§G.1)

#### 1.2 Overflow vs Efficiency Scatter (Pareto Front)
- [x] 4-panel layout: Gamma-3/FTSP · Empirical/FTSP · Gamma-3/CLS · Empirical/CLS: `BenchmarkAnalysis` + `SimulationSummary` `BenchmarkParetoPanel` grid + `paretoPortfolio.ts` / `paretoPanels.ts` run classifier (§G.1.2)
- [x] Color encoding: LA · LM-CF70 · LM-CF90 · SL-SL1 · SL-SL2: `strategyColor()` on Pareto scatter + efficiency ranking bars (§G.1.2)
- [x] Marker shape: RM-100 circle · RM-170 square · FFZ-350 diamond: `citySymbol()` from `parseLogPath()` on `BenchmarkParetoPanel` multi-run scatter (§G.1.2)
- [x] Computed Pareto front drawn as white dashed step line: `PolicyParetoChart` + `BenchmarkParetoPanel` on Simulation Summary / Benchmark Analysis (§G.1.2)
- [x] Log-scale toggle on Simulation Summary policy bar charts (§G.1)
- [x] Pareto scatter follows global ``logScale``: symlog overflows y-axis + log profit x-axis on ``PolicyParetoChart`` + ``BenchmarkParetoPanel`` (§G.1.2 / §G.7)
- [x] BenchmarkParetoPanel per-facet PNG export: ``exportChartPng()`` on each 4-panel Pareto facet with toast feedback (§G.1.2 / §G.7)
- [x] Symlog bar charts: `symlog.ts` + `useSymlog` on profit · km · overflows `MetricBarChart` when log scale on; secondary log-scale row adds profit/km symlog duplicates (§G.1)
- [x] BenchmarkAnalysis multi-run comparison bar charts follow global ``logScale`` (§G.1 / §G.7)
- [x] AlgorithmComparison per-metric bar charts follow global ``logScale`` (§G.1 / §G.7)
- [x] AlgorithmComparison error-bar whiskers on metric bars: mean ± std toggle via ``showErrorBars``; log/symlog whiskers via ``errorBarBounds`` when global ``logScale`` on (§G.1 / §G.7)
- [x] AlgorithmComparison symlog overflows on log-scale metric bars (§G.1.1 / §G.7)
- [x] Policy radar chart on Simulation Summary: normalised multi-metric overlay per policy with PNG export; log-normalised axes when global ``logScale`` on (Simulation Summary + Algorithm Comparison; §G.1 / §G.7)
- [x] Error-bar whiskers on Simulation Summary bar charts: custom ECharts series showing mean ± std; log/symlog whiskers via ``errorBarBounds`` when global ``logScale`` on (§G.1 / §G.7)
- [x] Hover tooltip: all config values + KPI values: `simMetadata.ts` + `policyTooltipFooter()` on bar/Pareto/heatmap/radar/parallel charts; `BenchmarkParetoPanel` adds `formatLogMeta` + `formatPolicyMeta` per run×policy point (§G.1.2)

#### 1.3 Policy Configuration Heatmaps
- [x] Heatmap split by distribution (Gamma-3 vs Empirical): `DistributionFacetHeatmaps` on Simulation Summary when policies span distributions; portfolio mode adds `BenchmarkDistributionHeatmap` facets via `groupRunsByDistribution()` (§G.1.3)
- [x] Heatmap split by graph (RM-100 vs RM-170 vs FFZ-350): shared `BenchmarkGraphHeatmap` facets by `cityScaleLabel()` on Benchmark Analysis + Simulation Summary portfolio mode (§G.1.3)
- [x] Cell value = mean overflows or mean kg/km (toggle): unified `heatmapMode` buttons (all / overflows / kg/km) on Simulation Summary + Benchmark Analysis; portfolio distribution/graph facets share the same mode (§G.1.3)
- [x] Color gradient from dark (worst) to bright (best): `PolicyHeatmapChart` + `BenchmarkPortfolioHeatmap` + shared `heatmapMetrics.ts` normalised indigo→green gradient; portfolio policy×metric heatmap when ≥2 runs loaded (§G.1.3)
- [x] Policy configuration heatmaps follow global ``logScale``: ``buildNormalizedHeatmapCells`` symlog/log-transforms KPI values before min–max normalisation on ``PolicyHeatmapChart``, ``BenchmarkPortfolioHeatmap``, ``BenchmarkDistributionHeatmap``, and ``BenchmarkGraphHeatmap``; tooltips show raw KPI values (§G.1.3 / §G.7)
- [x] BenchmarkDistributionHeatmap / BenchmarkGraphHeatmap facet PNG export: ``exportChartPng()`` per distribution/graph facet with toast feedback (§G.1.3 / §G.7)

#### 1.4 Parallel Coordinates (Hyper-Dimensional Policy Explorer)
- [x] Axes: city · N · dist · improver · strategy · constructor · overflows · kgkm · km · profit: `PolicyParallelChart` + `parallelPolicyAxes.ts` ten-axis schema on Simulation Summary; shared `BenchmarkPortfolioParallel` on Simulation Summary + Benchmark Analysis (§G.1.4)
- [x] Each of the 480 simulation logs rendered as a polyline: `BenchmarkPortfolioParallel` + `scanOutputPortfolio()` / `loadPortfolioLogs()` batch loader (up to 480 runs) on Benchmark Analysis (§G.1.4)
- [x] Portfolio DuckDB ingest: Simulation Summary unions primary + comparison runs into `summary_sim`; Benchmark Analysis → `benchmark_sim`; City Comparison → `city_sim` with sidecar-aware timing badges (§G.1.4 / §G.6)
- [x] Brushing on any axis instantly filters all other panels: ECharts parallel-axis brush toolbox on `PolicyParallelChart` → `handleBrushPolicies` cross-filter; click polyline → `toggleBrush`; DuckDB SQL sync via `brushSqlSync` (§G.1)
- [x] Highlight corridor: drag brush on overflows ≤ threshold to identify zero-overflow configs: overflow corridor slider + parallel-axis overflows brush syncs `overflowMax` + `effectiveBrushed` cross-filter on Simulation Summary (§G.1)
- [x] Parallel coordinates follow global ``logScale``: ``PolicyParallelChart`` + ``BenchmarkPortfolioParallel`` log-normalise profit · kg/km · km axes; symlog overflows; corridor brush inverts symlog via ``invertParallelAxisValue`` (§G.1.4 / §G.7)
- [x] Color polylines by mandatory-selection strategy: `strategyColor()` on `PolicyParallelChart` polylines; `BenchmarkPortfolioParallel` colours run polylines via `resolveRunSelectionStrategy()` + `selectionStrategyColor()` from log path / dominant policy with strategy legend (§G.1.4)
- [x] BenchmarkPortfolioParallel PNG export: ``exportChartPng()`` on portfolio parallel-coordinates panel with toast feedback (§G.1.4 / §G.7)

#### 1.5 Constructor Ranking Chart
- [x] Horizontal bar chart: `EfficiencyRankingChart` ranks policies by mean kg/km, bottom-up ordering; portfolio mode adds `PortfolioEfficiencyRanking` for run×policy configs (§G.1.5)
- [x] Rank by mean kg/km across all configurations: Simulation Summary efficiency ranking + `PortfolioEfficiencyRanking` + BenchmarkAnalysis `kg/km` metric column (§G.1.5)
- [x] Error bars showing std deviation: Simulation Summary bar-chart whiskers toggle (§G.1)
- [x] Error bars on efficiency ranking chart: horizontal kg/km whiskers toggle via `showErrorBars` (§G.1)
- [x] Efficiency ranking charts follow global ``logScale``: ``EfficiencyRankingChart`` + ``PortfolioEfficiencyRanking`` log x-axis; horizontal whiskers via ``errorBarBounds`` when log on (§G.1.5 / §G.7)
- [x] BenchmarkAnalysis efficiency ranking follows global ``logScale``: multi-run ``PortfolioEfficiencyRanking`` + single-run inline chart log x-axis; shared ``showErrorBars`` toggle; horizontal kg/km whiskers via ``errorBarBounds`` when log on (§G.1.5 / §G.7)
- [x] BenchmarkAnalysis multi-run metric bars use symlog for overflows when global ``logScale`` on (§G.1.1 / §G.7)
- [x] BenchmarkAnalysis multi-run metric bar error-bar whiskers: grouped run×policy bars show mean ± std via shared ``showErrorBars`` toggle; log/symlog whiskers via ``errorBarBounds`` + ``groupedBarWhiskerX`` when global ``logScale`` on (§G.1 / §G.7)

#### 1.6 Secondary Log-Scale Views
- [x] Auto-generate log-scale version below each chart that benefits from it (overflow counts, profit ranges): duplicate profit · km · kg · symlog-overflows row when global log toggle off (§G.1)
- [x] City Comparison section follows global ``logScale``: log-scale profit + symlog-overflows bars when on; linear raw values when off; `BenchmarkAnalysis` + `SimulationSummary` + dedicated `CityComparison` page with portfolio load + summary table (§G.1.6 / §G.7)
- [x] City Comparison error-bar whiskers: profit · symlog-overflows · kg/km grouped bars show mean ± std via ``showErrorBars`` toggle on ``cityComparisonChartOption``; log/symlog whiskers via ``errorBarBounds`` + ``groupedBarWhiskerX`` on Benchmark Analysis, Simulation Summary portfolio mode, and City Comparison page (§G.1.6 / §G.7)

**Status**: §G.1 complete — all checklist items delivered.

---

### §G.2 — Phase 2: Hierarchical Drill-Down (Sunburst / Treemap)

**Goal**: Enable macro → micro navigation from algorithm family level down to individual config variants.

- [x] Top-level Sunburst chart: inner ring = city/scale · middle ring = selection strategy · outer ring = constructor: `PolicyHierarchyPanel` + `policyHierarchy.ts` on Simulation Summary; `buildPortfolioHierarchy()` multi-root sunburst when ≥2 logs loaded (§G.2)
- [x] Angular span mapped to accumulated profit; color gradient = kg/km efficiency: sunburst/treemap segment `value` = profit sum; `itemStyle.color` from kg/km gradient; middle strategy ring adds `selectionStrategyColor()` border stroke (§G.2)
- [x] Click on any segment fires DuckDB-Wasm filter query: segment click → `policiesAtPath` → `toggleBrush` cross-filter; `SqlQueryPanel` `brushSqlSync` + `autoRunOnBrushSync` executes `brushedPoliciesSql` (§G.2)
- [x] Drill-down transition: Sunburst morphs into horizontal bar chart (mean ± variance per variant): `PolicyHierarchyPanel` `universalTransition` morphs sunburst/treemap → drill-down profit bars on segment click; log-scale profit x-axis when global ``logScale`` on (§G.2 / §G.7)
- [x] Error bars on drill-down bars representing variance across Empirical vs Gamma-3 distributions: `enrichDrillChildren` profit std + Empirical↔Gamma spread whiskers on `PolicyHierarchyPanel` drill-down; log-scale profit x-axis whiskers via ``errorBarBounds`` when global ``logScale`` on (§G.2 / §G.7)
- [x] Breadcrumb trail showing current filter path; click to navigate back up: `HierarchyBreadcrumb` in `PolicyHierarchyPanel` with root **All** reset (§G.2)
- [x] Treemap alternative view: area = profit, color = overflows (toggle with Sunburst): sunburst/treemap view toggle on Simulation Summary; kg/km vs overflows colour mode selector on `PolicyHierarchyPanel` (§G.2)
- [x] Shared strategy colour legend: `SELECTION_STRATEGY_LEGEND` + `StrategyLegend` chips on `PolicyParallelChart`, `BenchmarkPortfolioParallel`, and `PolicyHierarchyPanel` (§G.1.4 / §G.2)
- [x] Drill-down profit bars coloured by mandatory-selection strategy at strategy depth; constructor depth uses kg/km or overflow gradient via `resolveDrillBarColor()` (§G.2)

**Status**: §G.2 complete — all checklist items delivered.

---

### §G.3 — Phase 3: Geospatial Routing Visualization (deck.gl)

**Goal**: Animate the physical routes constructed by each algorithm over the real-world city graphs.

#### 3.1 Base Map Layer
- [x] Integrate deck.gl with MapLibre GL (OpenStreetMap tiles): `DeckRouteMap` uses `react-map-gl/maplibre` + Carto dark basemap (§G.16)
- [x] Load node coordinates for Rio Maior (N=100, N=170) and Figueira da Foz (N=350) from graph JSON files: `graphCoords.ts` presets + SimulationMonitor "Load graph coords" (§G.3.1)
- [x] Auto-detect graph preset from log path segments or day-1 bin count: `guessGraphPreset()` + SimulationMonitor auto-select (§G.3.1)
- [x] Render nodes as ScatterplotLayer: fill-level colour-coded tour stops + dimmed idle bins in `DeckRouteMap`; radius scales with fill % and collected kg (`bin_state_collected`) (§G.3.1)
- [x] Render depot as distinct marker: gold `ScatterplotLayer` with white stroke in `DeckRouteMap`
- [x] Pan/zoom/tilt with 3D perspective: `DeckRouteMap` controlled view state + 3D pitch toggle (0°/45°); OrbitView Cartesian mode in §G.3.4 (§G.3.1)

#### 3.2 Route Animation (TripsLayer)
- [x] Parse per-day route from `tour_indices` + `all_bin_coords` into timestamped coordinate arrays (`DeckRouteMap`)
- [x] Feed routes into deck.gl `TripsLayer` with animated trail during day playback (§G.3.2 / §G.16)
- [x] Timeline slider: day scrubber with range input + ◀/▶ step buttons (SimulationMonitor)
- [x] Playback controls: play / pause / 1×·2×·4× speed multiplier on day scrubber; `TripsLayer` animated trail in Mercator and OrbitView Cartesian modes (§G.3.2)
- [x] Multi-vehicle rendering with distinct color coding per vehicle: `vehicleTours.ts` splits depot-delimited `tour` sequences; `DeckRouteMap` + `RouteMapChart` render per-vehicle paths and per-vehicle tour-stop scatter layers (§G.3.2)

#### 3.3 Algorithm Comparison Mode
- [x] Side-by-side view: overlay/split toggle in SimulationMonitor when 2 policies visible; split renders dual `DeckRouteMap` or dual ECharts `RouteMapChart` panels (§G.16)
- [x] Algorithm Comparison → map deep link: `pendingMapCompare` sets visible policies + split layout when 2 policies present
- [x] Toggle visibility per policy: map policy chip row in SimulationMonitor; `DeckRouteMap` multi-route overlay with per-policy colour paths
- [x] Overlay skipped vs visited nodes: idle bins dimmed grey, tour stops fill-coded (bright) in `DeckRouteMap`

#### 3.4 Non-Geographic Cartesian Mode (OrbitView)
- [x] Switch between geographic (Mercator) and abstract Cartesian coordinate system: Simulation Monitor ECharts vs deck.gl toggle; `DeckRouteMap` auto-selects Mercator (geo) or OrbitView (abstract) (§G.3.4)
- [x] OrbitView camera: orbit, pan, zoom a 3D point cloud: `DeckRouteMap` OrbitView with fill-scaled Z elevation on tour stops (§G.3.4)
- [x] Used for normalized/synthetic datasets where coordinates are not GPS: circular `resolveBinPositions()` layout when log lacks lat/lng (§G.3.4)

**Status**: §G.3 complete — all checklist items delivered.

---

### §G.4 — Phase 4: Topological Graph Analytics (Sigma.js / Cosmograph)

**Goal**: Visualize the raw optimization graph structure, pheromone trails, and node-edge weights.

- [x] Load distance matrix from `assets/` as a weighted edge list: `graphTopology.ts` resolves sibling `gmaps_distmat.csv` or project `data/wsr_simulator/distance_matrix/`; k-NN edge list builder (§G.4)
- [x] Render graph using Sigma.js (WebGL): node radius ∝ profit, edge thickness ∝ inverse distance: `GraphTopologyPanel` ECharts `graph` series — node size ∝ bin fill %, edge width ∝ inverse distance; View toggle adds `TopologySigmaView` Sigma.js WebGL with fill/pheromone styling + `TopologyCosmographView` dense point-mode WebGL (§G.4)
- [x] Force-directed layout (ForceAtlas2) via Graphology: `TopologySigmaView` runs `graphology-layout-forceatlas2` on force layout; ECharts path keeps Fruchterman-Reingold in `forceDirectedLayout()` (§G.4)
- [x] ACO pheromone trail visualization: edge opacity/color intensity ∝ accumulated pheromone weight after each iteration: `accumulateTourPheromone()` deposits τ on consecutive tour edges; amber edge styling in `GraphTopologyPanel` ECharts + Sigma.js + Cosmograph views (live ACO solver τ matrix deferred to logic layer)
- [x] Cross-filter from DuckDB-Wasm: brushing a profit range highlights matching nodes: fill-% dual slider + SQL "Brush profit range" / day row click → topology panel; click node in ECharts/Sigma/Cosmograph view → fill-% brush (§G.4)
- [x] Dynamic re-layout when filter applied: clusters emerge based on algorithm prioritization: "Re-layout on filter" toggle re-runs spring layout on filtered subgraph (§G.4)
- [x] Cosmograph alternative for large dense graphs (N=350): `radialDenseLayout()` + auto radial when N≥200; layout mode selector (auto/force/radial) on `GraphTopologyPanel`; `TopologyCosmographView` Sigma.js point renderer with ForceAtlas2 dense settings (§G.4)
- [x] Timeline slider synced with route animation to show pheromone evolution over iterations: pheromone day slider syncs with Simulation Monitor day scrubber + playback; "By tour step" mode steps τ per consecutive tour edge via `accumulateTourPheromoneByStep` (§G.4)
- [x] Topology pheromone trails follow global ``logScale``: ``pheromoneWeightDisplay()`` + ``normalizePheromone()`` / ``pheromoneIntensity()`` log-transform τ before edge opacity/width on ECharts, Sigma.js, and Cosmograph views; ``GraphTopologyPanel`` receives ``logScale`` from Simulation Monitor (§G.4 / §G.7)
- [x] ECharts topology PNG export: ``exportChartPng()`` on ``GraphTopologyPanel`` when View = ECharts (§G.4 / §G.7)
- [x] ECharts topology SVG export: ``exportChartSvg()`` on ``GraphTopologyPanel`` when View = ECharts; toast feedback (§G.4 / §G.7)
- [x] Sigma.js / Cosmograph WebGL PNG export: ``exportContainerCanvasPng()`` on ``GraphTopologyPanel`` when View = Sigma.js or Cosmograph; toast feedback (§G.4 / §G.7)

**Status**: §G.4 complete — all checklist items delivered.

---

### §G.5 — Phase 5: Machine Learning Introspection Dashboard

**Goal**: Expose the internals of trained neural CO models (Attention Models, Routing Transformers).

#### 5.1 TensorDict Data Pipeline
- [x] Rust backend: load `.npy`/`.npz` TensorDict files via `ndarray-npy` crate: `tensor.rs` `inspect_npz_archive` + `load_tensor_slice` (§G.5.1 — full native `.td` parse deferred to logic layer)
- [x] TensorDict (`.td`) inspect + slice via Python subprocess (`torch.load` + key/shape listing; slice export matches NPZ path): `inspect_npz_archive` / `load_tensor_slice` accept `project_root` + `python_executable`; Archive tab opens `.td` files (§G.5.1)
- [x] Memory-map large tensor files (avoid full RAM load): `load_npy_plane_mmap` + `load_npz_plane_mmap` via `memmap2` reads only the trailing 2-D plane for standalone `.npy` or stored `.npz` entries > 8 MB; `load_npz_plane_decompress` slices deflated `.npz` entries after single-entry inflate; `TensorSlicePreview.used_memmap` / `used_decompress_slice` surfaced in Archive/Attention tabs; `probe_npy_mmap` covers large stored or compressed `.npz` arrays (§G.5.1)
- [x] Stream specific tensor slices to frontend over Arrow IPC on demand: `tensor_slice_to_arrow_ipc` long-format `(row, col, value)`; `runTensorArrowPipeline` ingests into DuckDB-Wasm as `studio_tensor` from Archive tab; `.td` slices supported via Python handoff (§G.5.1)

#### 5.2 3D Loss Landscape Visualization (React Three Fiber)
- [x] Python utility script: compute loss surface grid using Li et al. filter-normalized random directions: `logic/gen/export_loss_landscape.py` with `--probe-mode auto|training|proxy` and `--batch-size` (default 4); training probe averages greedy forward-loss across N synthetic instances per grid point; bundles `probe_mode` + `batch_size` in NPZ (§G.5.2)
- [x] Export 2D grid of loss values as `.npz`: `loss_grid`, `theta1`, `theta2` keys (§G.5.2)
- [x] React Three Fiber: render grid as vertex-displaced `PlaneGeometry` 3D topography: `LossLandscape3D` lazy chunk (§G.5.2)
- [x] `InstancedMesh` voxel alternative: per-cell `boxGeometry` cubes with height ∝ loss; Loss tab "Surface mesh / InstancedMesh voxels" toggle (§G.5.2)
- [x] Color gradient: low loss = deep blue, high loss = bright red (`lossToColor` vertex colours)
- [x] Camera: orbit, zoom, perspective controls (`OrbitControls` + `Canvas`)
- [x] Overlay 2D ECharts contour map adjacent to the 3D canvas (CSS positioned): `MLIntrospectionPanel` Loss tab side-by-side grid; log-scale colour map when global ``logScale`` on with raw-loss tooltips (§G.5.2 / §G.7)
- [x] Loss landscape 3D terrain follows global ``logScale``: ``LossLandscape3D`` log-transforms height/colour via ``transformMatrixLogScale`` when on; minima sharpness analysis stays on raw loss grid (§G.5.2 / §G.7)
- [x] Project exact-solver solutions (BPC optimum) as a marker on the landscape: `export_loss_landscape.py` bundles `bpc_theta1`/`bpc_theta2`/`bpc_loss`; `load_npz_vectors` + `resolveBpcMarker` + amber octahedron in `LossLandscape3D` + ECharts `markPoint` on contour (§G.5.2)
- [x] Identify sharp vs flat minima; annotate with generalization notes (Gamma-3 vs Empirical): `analyzeLossMinima` Laplacian sharpness + ``generalizationNote`` per basin label on 3D terrain + Loss tab (§G.5.2)

#### 5.3 Attention Weight Visualization (Sigma.js overlay)
- [x] Load attention weight matrices from TensorDict for a selected simulation step: `load_tensor_slice` with leading-dim indices + decode-step slider (§G.5.3)
- [x] Render as bipartite graph on top of node coordinates: edge opacity ∝ attention weight magnitude: ECharts `buildAttentionGraphOption` + Sigma.js WebGL `AttentionSigmaView` (ForceAtlas2, lazy `sigma` chunk) with graph preset loader; View toggle: Heatmap / ECharts graph / Sigma.js (§G.5.3)
- [x] Attention head selector: `detectHeadAxis` + per-head index dropdown; Q/K/V role filter + per-role colour palettes via `classifyAttentionRole` / `groupAttentionKeys` (§G.5.3)
- [x] Timeline slider: step through sequential decoding steps: decode-step range on Attention tab (§G.5.3)
- [x] Sparse Routing Transformer mode: `applySparseTopK` keeps top-k connections per query row (§G.5.3)
- [x] Spherical k-means query-row clustering: `sphericalKMeans` + row reorder + ECharts `markArea` cluster bands; K-means selector (2–8) on Attention tab (§G.5.3)
- [x] Compare attention patterns of model trained on Empirical vs Gamma-3 distributions: Attention tab "Empirical vs Gamma-3" compare mode; dual archive picker; `inferDistributionLabel` path heuristics; side-by-side heatmaps + overlay Δ diff (§G.5.3)
- [x] Side-by-side vs overlay toggle: decode-step compare (side-by-side dual heatmap / overlay Δ diff) (§G.5.3)
- [x] Attention weight heatmaps follow global ``logScale``: ``MLIntrospectionPanel`` log-transforms raw Q/K/V weight cells when on; overlay/distribution Δ diff panels stay linear; tooltips show raw weights (§G.5.3 / §G.7)
- [x] Attention bipartite graph overlays follow global ``logScale``: ``buildAttentionGraphOption`` + ``AttentionSigmaView`` log-transform edge opacity/width via ``attentionWeightDisplay``; tooltips and edge weight attributes retain raw attention values (§G.5.3 / §G.7)
- [x] ML introspection ECharts PNG/SVG export: ``exportChartPng()`` / ``exportChartSvg()`` on ``MLIntrospectionPanel`` attention heatmap (primary + compare panels), attention bipartite graph, and loss contour map (§G.5 / §G.7)
- [x] Loss landscape 3D terrain PNG export: ``exportContainerCanvasPng()`` on ``LossLandscape3D`` R3F canvas (surface mesh + InstancedMesh voxels) via ``MLIntrospectionPanel`` Loss tab (§G.5.2 / §G.7)
- [x] Attention Sigma.js WebGL PNG export: ``exportContainerCanvasPng()`` on ``AttentionSigmaView`` canvas via ``MLIntrospectionPanel`` Attention tab (§G.5.3 / §G.7)

**Status**: §G.5 complete — all checklist items delivered.

---

### §G.6 — Phase 6: OLAP Data Cube Explorer

**Goal**: Give the researcher a free-form SQL/pivot interface backed by DuckDB-Wasm for custom analysis queries.

- [x] DuckDB-Wasm query editor with syntax highlighting (Monaco or CodeMirror): `SqlQueryPanel` lazy Monaco SQL editor on Data Explorer + standalone `OlapExplorer` page with table picker + CSV/JSONL ingest (prefers ``.arrow`` sidecars; §G.6)
- [x] Portfolio SQL panels: `SqlQueryPanel` on Benchmark Analysis (`benchmark_sim`) and City Comparison (`city_sim`) when multi-run portfolios are loaded (§G.6)
- [x] Portfolio query templates: `portfolioSqlTemplates()` cross-run robustness, run leaderboard, run×policy variance, Pareto-by-run; `SqlQueryPanel` `portfolioMode` on multi-log views (§G.6)
- [x] Algorithm Comparison DuckDB ingest: `runSimulationArrowPipeline()` → `algorithm_sim` + `SqlQueryPanel` + timing badge when Simulation Monitor watch path is active (§G.6)
- [x] Algorithm Comparison SQL templates: `algorithmSqlTemplates()` policy ranking, worst overflow days, zero-overflow rate, day-over-day profit Δ; `SqlQueryPanel` `algorithmMode` (§G.6)
- [x] Algorithm Comparison brush SQL sync: chart click → global policy filter → `brushSqlSync` + `autoRunOnBrushSync` on `algorithm_sim` (§G.6)
- [x] Benchmark Analysis brush SQL sync: efficiency ranking + metric bar click → global policy filter → `brushSqlSync` + `autoRunOnBrushSync` on `benchmark_sim` (§G.6)
- [x] City Comparison brush SQL sync: city chart / summary table click → `run_label` filter → `brushSqlSync` + `autoRunOnBrushSync` on `city_sim`; `brushedPortfolioSql()` unifies policy + run_label brushes (§G.6)
- [x] Simulation Summary portfolio run_label brush SQL sync: comparison-run click, city chart click, portfolio efficiency ranking click → `highlightRunLabels` + `brushSqlSync` on `summary_sim` (§G.6)
- [x] Benchmark Analysis city chart run_label brush: city comparison chart click → `highlightRunLabels` + `brushSqlSync` on `benchmark_sim` (§G.6)
- [x] OLAP Explorer global policy brush SQL sync: `GlobalFilterBar` policy → `brushSqlSync` + `autoRunOnBrushSync`; portfolio/algorithm template modes per ingested table (§G.6)
- [x] OLAP Explorer global run_label brush SQL sync: `GlobalFilterBar` run selector + `highlightRunLabels` on portfolio tables; distinct ``run_label`` values from DuckDB (§G.6)
- [x] SQL result row + pivot run_label cross-filter: click policy or ``run_label`` cell → `useGlobalFiltersStore` → `brushSqlSync` + row dimming (§G.6)
- [x] Portfolio global run_label filter bar: `usePortfolioRunBrush` + `GlobalFilterBar` run selector on Simulation Summary, Benchmark Analysis, and City Comparison when ≥2 runs loaded (§G.6)
- [x] Portfolio global city/scale filter bar: `brushedCity` in `useGlobalFiltersStore` + `GlobalFilterBar` city selector on Summary/Benchmark/City when ≥2 city groups loaded (§G.6)
- [x] OLAP Explorer global city/scale brush SQL sync: `groupRunLabelsByCity()` + `GlobalFilterBar` city selector on portfolio tables; `resolveBrushedRunLabels()` expands city brush to ``run_label`` IN clause via `SqlQueryPanel` ``portfolioRunLabels`` (§G.6)
- [x] Global filter bar → SQL brush sync: `SqlQueryPanel` ``brushFilter`` merges ``useGlobalFiltersStore`` policy / ``run_label`` / city brush when chart props are absent; ``autoRunOnBrushSync`` fires on filter-bar changes (§G.6)
- [x] Portfolio DuckDB ``city_scale`` column: `runPortfolioSimulationArrowPipeline()` adds parsed city/scale label alongside ``run_label``; city leaderboard SQL template (§G.6)
- [x] Portfolio single-log ``run_label`` + ``city_scale`` columns: `runPortfolioSimulationArrowPipeline()` always annotates logs (including one-run Summary/Benchmark/City/OLAP ingests) (§G.6)
- [x] SQL result row ``city_scale`` cross-filter: click ``city_scale`` cell → global ``brushedCity``; row dimming + active highlight (§G.6)
- [x] Pivot table ``city_scale`` cross-filter: `PivotTablePanel` row highlight + click sets global ``brushedCity`` (§G.6)
- [x] City×policy matrix SQL template: `portfolioSqlTemplates()` grouped ``city_scale`` × ``policy`` kg/km matrix (§G.6)
- [x] Auto-chart portfolio GROUP BY detection: `queryAutoChart.ts` prefers ``city_scale`` / ``run_label`` / ``policy`` dimensions + KPI metrics (§G.6)
- [x] Data Explorer global filter bar + SQL brush sync when CSV has ``policy`` column (§G.6)
- [x] Data Explorer CSV-derived policy / ``run_label`` / city filter bar + row cross-filter dimming (§G.6)
- [x] OLAP dynamic portfolio mode: `duckDbHasColumn()` detects ``run_label`` on any ingested table (§G.6)
- [x] Auto-chart grouped bar for multi-dimension GROUP BY (``city_scale`` × ``policy``; §G.6)
- [x] Auto-chart heatmap for city×policy / run×policy matrix query results: `queryAutoChart.ts` ``heatmap`` type (§G.6)
- [x] OLAP Explorer DuckDB-derived policy / ``city_scale`` filter bar: ``listDuckDbDistinctValues()`` on active table (§G.6)
- [x] Data Explorer cell-level cross-filter: click brush column cell only (policy / ``run_label`` / ``city_scale``) (§G.6)
- [x] Data Explorer brush-aware CSV export: export respects global filter + text search + sort (§G.6)
- [x] SQL result grid cell-level cross-filter: click brush column cell only in ``SqlQueryPanel`` (§G.6)
- [x] Auto-chart click cross-filter: bar / grouped-bar / heatmap clicks apply global policy / ``run_label`` / ``city_scale`` brush (§G.6)
- [x] Auto-chart PNG export: ``exportChartPng()`` on ``SqlQueryPanel`` auto-chart (§G.6)
- [x] Auto-chart type override: ``suggestChartAlternatives()`` chips switch bar / grouped-bar / heatmap (§G.6)
- [x] Run×policy matrix SQL template: ``portfolio-run-policy-matrix`` in ``portfolioSqlTemplates()`` (§G.6)
- [x] Pareto efficiency frontier SQL template: ``pareto-frontier`` + ``portfolio-pareto-frontier`` in ``duckdbTemplates.ts`` (§G.6)
- [x] Auto-chart scatter cross-filter: labeled profit vs overflows scatter click → global policy / ``run_label`` / ``city_scale`` brush (§G.6)
- [x] Auto-chart SVG export: ``exportChartSvg()`` on ``SqlQueryPanel`` auto-chart (§G.6)
- [x] Pre-built query templates: robustness profile, variance analysis, Pareto efficiency frontier: `duckdbTemplates.ts` template chips (§G.6)
- [x] Result grid with sortable columns, row filter search, and filtered CSV export: `SqlQueryPanel` sortable result table + search box + export respects filter (§G.6)
- [x] Auto-chart: map query result columns to ECharts chart type suggestions: `queryAutoChart.ts` + `SqlQueryPanel` bar/line/scatter/heatmap suggestion below results (§G.6)
- [x] Pivot table UI: drag dimensions/measures onto row/column/value wells: `PivotTablePanel` draggable column chips + HTML5 drop wells for row/column/value + agg selector + heatmap on `SqlQueryPanel` (§G.6)
- [x] Cross-filtering from pivot table updates all Phase 1–2 charts bidirectionally: pivot/result row click sets `useGlobalFiltersStore` policy; `GlobalFilterBar` policy highlights matching SQL rows + dims pivot heatmap rows via `highlightRowLabels` (§G.6)
- [x] Auto-chart Pareto frontier step-line overlay: labeled profit vs overflows scatter highlights frontier points + dashed ``paretoStepLine()`` (§G.6)
- [x] Auto-chart log-scale on profit vs overflows scatter: symlog overflows y-axis + log profit x-axis when global ``logScale`` on (§G.6 / §G.1 / §G.7)
- [x] Auto-chart log-scale on bar / grouped-bar / line when y-axis metric is overflow, loss, or KPI (§G.6 / §G.7)
- [x] Auto-chart heatmap log-scale visualMap: matrix cell values transformed via ``displayBarValue`` when global ``logScale`` on (§G.6 / §G.7)
- [x] Pivot table heatmap log-scale: ``PivotTablePanel`` passes global ``logScale`` + value column to ``pivotHeatmapOption`` (§G.6 / §G.7)
- [x] Auto-chart line cross-filter: time-series point click → ``onDaySelect`` when ``xKey`` is ``day`` (§G.6)
- [x] Auto-chart line type in override alternatives for day/epoch/step queries (§G.6)
- [x] Pivot table heatmap PNG export: ``exportChartPng()`` on ``PivotTablePanel`` pivot heatmap with toast feedback (§G.6 / §G.7)

**Status**: §G.6 complete — all checklist items delivered.

---

### §G.7 — Phase 7: Integrated Workflow & UX Polish

**Goal**: Connect all analytics phases into a single cohesive analytical narrative flow, and satisfy all §D UX requirements.

- [x] App-level navigation: `WorkflowNav` strip — Overview → Drill-Down → Geospatial → Registry → ML → HPO → Launch (§G.7)
- [x] Global filter state management (Zustand): `useGlobalFiltersStore` + `GlobalFilterBar` propagates policy/sample filters across SimulationMonitor, AlgorithmComparison, SimulationSummary, and BenchmarkAnalysis
- [x] Bookmarkable analysis states (serialize filter + view to URL hash for deep-linking via `useHashSync`)
- [x] Bookmarkable ``run_label`` filter: `useHashSync` serializes global ``runLabel`` as ``r`` query param; restored on load and browser back/forward (§G.7)
- [x] Bookmarkable city/scale brush: `useHashSync` serializes global ``brushedCity`` as ``c`` query param; restored on load and browser back/forward (§G.7)
- [x] Global log-scale filter: ``logScale`` in ``useGlobalFiltersStore`` + ``GlobalFilterBar`` toggle propagates to Simulation Summary (incl. per-day trajectory + policy radar + policy/portfolio parallel coordinates + hierarchy drill-down profit bars + drill-down error-bar whiskers + grouped metric bar whiskers + city-comparison error-bar whiskers + Pareto symlog scatter + policy configuration heatmaps), Benchmark Analysis (incl. portfolio parallel + Pareto panels + graph heatmaps + multi-run metric-bar error-bar whiskers + city-comparison error-bar whiskers + efficiency-ranking error-bar whiskers), Algorithm Comparison (radar + metric bars + error-bar whiskers), City Comparison (city-comparison error-bar whiskers), Evaluation Runner, Training Monitor, Training Hub, HPO Tracker (incl. parallel coordinates objective axis), Experiment Tracker (ZenML step durations + ML loss contour + 3D loss terrain + attention weight heatmaps + attention bipartite graph overlays), Simulation Monitor daily KPI charts + graph topology ACO pheromone edge styling, Data Generation demand histogram, OLAP/Data Explorer auto-charts (incl. symlog profit vs overflows scatter + heatmap visualMap) and pivot heatmaps (§G.1 / §G.7)
- [x] Bookmarkable log-scale toggle: `useHashSync` serializes global ``logScale`` as ``l=1`` query param; restored on load and browser back/forward (§G.7)
- [x] Dark/light theme toggle with Tauri Store persistence (§D.3, §D.4): `TopBar` toggle + Settings appearance radio; `useAppStore` Zustand `persist`
- [x] Keyboard shortcuts: `G` → simulation monitor, `Q` → HPO tracker, `P` → process monitor, `M` → map/simulation twin, `T`/`H`/`E` → train/HPO workflow, `L`/`D`/`V` → sim/data-gen/eval launchers, `Ctrl+.` → cancel first running process, `Ctrl+Shift+P` → process monitor, `Ctrl+R` → launch on active launcher page, digits `1`–`8` → quick nav, `?` → shortcuts help overlay (§D.7)
- [x] Keyboard shortcuts help overlay: `KeyboardShortcutsHelp` modal + TopBar button; `Escape` dismisses
- [x] Lazy-loaded page components: all 17 views behind `React.lazy` + `Suspense` in `App.tsx` (§G.7)
- [x] Command palette: `CommandPalette` fuzzy-search overlay for all views + actions; `Ctrl+K` / TopBar search button; arrow keys + Enter navigation
- [x] Vite `manualChunks`: echarts, maplibre, deck.gl, monaco, duckdb, r3f, sigma split into separate vendor bundles (§G.7)
- [x] Sidebar page prefetch: `prefetchPage()` warms lazy route chunks on nav item hover
- [x] Command palette bundle import: "Import .wsroute Bundle" action via `useWsrouteImport` hook
- [x] Recent files quick open: `useRecentFilesStore` persisted list; command palette Recent section; tracked from Simulation Monitor, Summary, Output Browser, Data Explorer
- [x] Startup route prefetch: `App.tsx` warms all 18 lazy route chunks (monitor, analytics, launch, files, settings) on mount (§G.7)
- [x] Startup vendor prefetch: echarts, maplibre-gl, @deck.gl/react, @monaco-editor/react, @duckdb/duckdb-wasm, sigma, @react-three/fiber + DeckRouteMap warmed on mount (§G.7)
- [x] Startup timing probe: `useStartupTiming` reports module-load → first React mount + route prefetch complete in Settings About (§G.7)
- [x] React toast notifications + Tauri OS notifications for background job completion when window is not focused (§D.8)
- [x] Responsive layout: `Layout` max-width `1920px` container, `sm:` padding breakpoints, `lg:` grid columns; collapsible sidebar with mobile overlay backdrop (`useLayoutStore`); sidebar auto-collapses below `lg` breakpoint via `matchMedia`; analytics chart grids use `grid-cols-1 sm:grid-cols-2` / `md:grid-cols-2` breakpoints (§G.7)
- [x] BenchmarkAnalysis responsive chart grids: Pareto panels `md:grid-cols-2`, metric bars `sm:grid-cols-2`, eval checkpoint charts `sm:grid-cols-2 lg:grid-cols-3` (§G.7)
- [x] AlgorithmComparison responsive chart grids: metric bars `sm:grid-cols-2 lg:grid-cols-4` (§G.7)
- [x] EvaluationRunner responsive inline chart grid: `sm:grid-cols-2 lg:grid-cols-3` (§G.12 / §G.7)
- [x] Performance budget probe: Settings About shows prefetch timing vs 2s target with pass/fail badge; "Run Chart Render Benchmark" measures representative ECharts first-paint vs 500 ms budget (§G.7)
- [x] Settings Arrow benchmark uses shared `formatPipelineTimingBadge()` for last-ingest summary (§G.0 / §G.7)
- [x] Export helpers with toast feedback: ``exportChartPngWithToast()`` / ``exportChartSvgWithToast()`` / ``exportContainerCanvasPngWithToast()`` / ``exportCanvasPngWithToast()`` centralise Sonner success/failure toasts in ``chartExport.ts``; ``ChartExportButtons`` pairs PNG + SVG on ECharts panels; ``CanvasExportButton`` wraps WebGL/canvas PNG export (§G.7)
- [x] ``ChartExportButtons`` propagated to portfolio facets, OLAP pivot/auto-chart, route-map preview, graph topology ECharts view, and ML introspection ECharts panels (§G.7)
- [x] ``CanvasExportButton`` propagated to deck.gl route map, graph topology Sigma.js/Cosmograph WebGL views, and ML introspection Attention Sigma.js + LossLandscape3D R3F canvas exports (§G.7)
- [x] Export: ECharts PNG export via ``exportChartPngWithToast()`` on SimulationMonitor, SimulationSummary (trajectory + radar + heatmap + Pareto + efficiency ranking + bar charts), AlgorithmComparison (radar + bar charts), BenchmarkAnalysis (sim + eval charts incl. kg/km), BenchmarkParetoPanel (per-facet Pareto scatter), BenchmarkPortfolioParallel, BenchmarkDistributionHeatmap / BenchmarkGraphHeatmap (facet heatmaps), BenchmarkPortfolioHeatmap, PortfolioEfficiencyRanking, TrainingMonitor (overlay + sparklines), TrainingHub (live chart + sparklines), DataGeneration (demand histogram), ExperimentTracker, HPOTracker charts, GraphTopologyPanel (ECharts view), MLIntrospectionPanel (attention heatmap primary + compare, attention graph, loss contour), PivotTablePanel (pivot heatmap), SqlQueryPanel auto-chart; WebGL/canvas PNG via ``CanvasExportButton`` (``exportContainerCanvasPngWithToast()`` / ``exportCanvasPngWithToast()``) on ``DeckRouteMap`` (Mercator tile / OrbitView Cartesian), GraphTopologyPanel (Sigma.js + Cosmograph views), and MLIntrospectionPanel (LossLandscape3D terrain + AttentionSigmaView); ECharts SVG via ``exportChartSvgWithToast()`` / ``ChartExportButtons`` on SimulationMonitor (route map + daily KPI timeseries), SimulationSummary (trajectory + radar + heatmap + parallel + hierarchy + Pareto + efficiency ranking + bar charts + city comparison), AlgorithmComparison (radar + metric bars), BenchmarkAnalysis (sim + eval + efficiency ranking), CityComparison, PortfolioEfficiencyRanking, TrainingMonitor (overlay + sparklines), TrainingHub (live chart + sparklines), DataGeneration (demand histogram), EvaluationRunner (inline checkpoint charts), ExperimentTracker (MLflow metric comparison), HPOTracker (history + importance + cross-study + parallel), ZenMLPipelineView (step durations), GraphTopologyPanel (ECharts view), MLIntrospectionPanel attention/loss charts, BenchmarkParetoPanel, BenchmarkPortfolioParallel, BenchmarkDistributionHeatmap / BenchmarkGraphHeatmap, BenchmarkPortfolioHeatmap, PivotTablePanel, and SqlQueryPanel auto-chart; table CSV via `downloadCsv()` on MLflow runs, ZenML runs, Simulation Summary ranking, Data Explorer; Parquet via `export_csv_to_parquet` / `export_table_parquet` on Data Explorer, Output Browser CSV viewer, Simulation Summary ranking
- [x] Data Explorer: sortable column headers (click header to toggle asc/desc numeric/text sort; §G.6)
- [x] Data Explorer: row filter search box matching any column with filtered/total row count (§G.6)
- [x] Data Explorer: CSV export respects active filter and sort order (exports visible subset; §G.6)

**Status**: §G.7 complete — all checklist items delivered.

---

### §G.8 — Phase 8: Data Export & Packaging

**Goal**: Make the Studio distributable and extend the Python pipeline to output Studio-compatible data bundles.

- [x] Python export script: `logic/gen/export_for_studio.py` — packages simulation CSV + graph JSONs + TensorDict NPZs + `.td` datasets into a `.wsroute` zip bundle with `manifest.json`; `--arrow` emits Arrow IPC (`.arrow`) sidecars for each CSV and simulation JSONL log (§G.8)
- [x] Rust bundle Arrow export: `create_wsroute_bundle(..., include_arrow)` emits `.arrow` sidecars via `write_csv_arrow_sidecar()` + `write_simulation_log_arrow_sidecar()`; Output Browser checkbox + manifest `arrow_sidecars` count (§G.8)
- [x] Studio sidecar ingest: DuckDB-Wasm pipeline auto-loads sibling `.arrow` when opening CSV or JSONL in Data Explorer / Simulation Summary / OLAP / Settings benchmark (§G.8)
- [x] Rust backend: `inspect_wsroute_bundle` lists bundle contents in Output Browser
- [x] Rust backend: `create_wsroute_bundle` packages a run directory into a `.wsroute` zip with `manifest.json`
- [x] Rust backend: `extract_wsroute_bundle` decompresses a bundle; returns first `.jsonl` path for Simulation Summary
- [x] Output Browser: "Export as .wsroute" on selected run (save dialog); "Extract & Open" on `.wsroute` files
- [x] Output Browser: drag-drop `.wsroute` bundle onto file viewer via Tauri `onDragDropEvent` (`useFileDrop` hook); inspects manifest without directory picker
- [x] Global file drop: `useGlobalFileDrop` in `Layout` extracts `.wsroute` to `assets/output/.imports/` or opens `.jsonl` logs in Simulation Summary from anywhere in the app
- [x] Integration test: `wsroute_bundle_round_trip_preserves_jsonl` + `simulation_arrow_sidecar_row_parity` Rust unit tests — create bundle → extract → verify `.jsonl` log content and Arrow sidecar row counts match parsed entries (§G.8)
- [x] Tauri bundler config: `tauri.conf.json` targets `deb`/`appimage`/`msi`/`dmg`; Linux deb section + Windows NSIS; `npm run tauri:build` / `tauri:build:linux` scripts; `createUpdaterArtifacts: true` emits `.sig` sidecars (partial — code-signing keys deferred)
- [x] App version command: `system::get_app_version` surfaced in Settings About (§G.8 / §G.19)
- [x] Update check command: `system::check_for_updates` uses Tauri updater plugin when `WSMART_UPDATER_PUBKEY` + `WSMART_UPDATE_URL` are set; falls back to JSON manifest version compare; Settings "Check for Updates" + conditional "Download & Install" button (§G.8)
- [x] Signed update install: `system::install_app_update` downloads/installs pending signed update via `tauri-plugin-updater`; `updater:default` capability; example manifest at `app/updater.example.json` (partial — release signing keys + CDN hosting deferred)

**Status**: §G.8 complete — updater plugin wired; code-signing keys and hosted signed releases deferred to release engineering.

---

### §G.9 — Phase 9: Simulation Launcher & Run Manager ✅

**Goal**: Port the PySide6 simulation tab to Tauri/React and add the improvements identified in §D.

- [x] React form: Hydra override textarea → `spawn_python_process main.py test_sim <overrides>`
- [x] Rust backend: spawn `main.py test_sim <overrides>` via `tokio::process::Command`; `process:spawn` event emitted on start; stdout streamed as `process:stdout` events
- [x] Cancel button: sends cancel signal via `tokio::sync::watch` channel (§D.5)
- [x] Toast notification on launch success / failure (§D.8) via `useSpawnProcess` hook
- [x] React form: full parameter set — 8-policy multi-select checkboxes, graph area text input, `num_loc` / `n_samples` / `cpu_cores` / `seed` number fields, data distribution radio (Normal / Gamma / Empirical); exactly mirrors `just controller::test-sim` Hydra args
- [x] "Advanced Overrides" collapsible panel: free-form textarea for arbitrary Hydra overrides (§D.6 Option A); live command preview below the form
- [x] Policy selection panel: load registered policy names from `test_sim.yaml` via `list_sim_policies` Rust command at runtime (89 policies; falls back to 8 defaults when file missing)
- [x] Live status display: after launch, subscribes to `process:stdout` events for the spawned process ID; parses `GUI_DAY_LOG_START:` markers; displays a per-policy card grid with day / profit / km / overflows in real time; "View Summary →" and "Process Monitor" navigation buttons shown on completion
- [x] On completion: auto-navigate to `simulation_summary` after 5-second countdown with cancel button; countdown driven by `useEffect` on `simStatus === "completed"`; "View Summary →" manual button always shown alongside countdown
- [x] Session persistence for form values: `useSimLauncherStore` (Zustand `persist`, key `wsroute-sim-launcher`) stores `selectedPolicies`, `area`, `numLoc`, `samples`, `nCores`, `seed`, `distribution`, `extraOverrides`; ephemeral runtime state stays in component state
- [x] Live progress + ETA (hundred-thirty-seventh pass): ``LiveTrainProgressBar`` in live status panel during running simulations (§D.2 / §G.9)
- [x] ``LauncherNavMesh`` shared navigation + ``Simulation Monitor →`` / ``Simulation Summary →`` post-run shortcuts (hundred-thirty-ninth pass; §D.7)
- [x] ``LauncherNavMesh`` ``Output Browser →`` post-run shortcut on completed simulations (hundred-forty-second pass; §G.14 / §D.7)
- [x] Post-run Output Browser deep-link via ``outputRunPath`` + ``pendingRunPath`` when stdout contains ``.jsonl`` (hundred-forty-third pass; §G.14 / §D.7)
- [x] Post-run panel persistence via ``findRecentLauncherProcessId`` when navigation clears local state (hundred-forty-sixth pass; §G.9 / §D.7)
- [x] ``LauncherLivePanel`` shared live/post-run panel shell with ``navTrailing`` auto-summary countdown slot (hundred-fifty-seventh pass; §G.9 / §D.7)
- [x] ``ProcessIdFooter`` shared process-id footer row on Simulation Launcher live panel (hundred-fifty-eighth pass; §G.9 / §D.7)

---

### §G.10 — Phase 10: Training & HPO Launch Hub ✅

**Goal**: Port the PySide6 reinforcement learning/training tab to Tauri/React.

- [x] Mode selector: train / hpo / eval
- [x] Hydra override textarea → `spawn_python_process main.py <mode> <overrides>`
- [x] Cancel and toast notifications via `useSpawnProcess` hook (§D.5, §D.8)
- [x] React form (train mode): problem selector (vrpp/wcvrp/scwcvrp), model selector (am/tam/ddam/moe), encoder selector (gat/gcn/mha), batch size, max epochs; mirrors controller justfile `train` recipe
- [x] React form (hpo mode): problem/model/encoder selectors + HPO method (nsgaii/tpe/dehb/random), trial count, num_workers; mirrors controller justfile `hpo` recipe
- [x] React form (eval mode): checkpoint path picker (Tauri dialog; .pt/.ckpt/.pth), dataset path picker (.pkl/.json/.csv), problem selector, decoding strategy (greedy/sampling/beam), val_size; mirrors controller justfile `eval` recipe
- [x] WandB toggle: adds `tracker.enabled=false` when disabled
- [x] Live command preview (via `useMemo`): exact `python main.py <mode> <args>` shown before launch
- [x] Live training progress panel (§D.2): `parseMetricLine` parses JSON and `key=value` stdout lines; `LiveChart` ECharts canvas shows train_loss (solid), val_loss (dashed), reward (dotted, right y-axis); latest snapshot row shows epoch/train_loss/val_loss/reward/grad_norm inline
- [x] Live training charts follow global ``logScale``: ``LiveChart`` + ``MiniSparkline`` log y-axis on loss/grad_norm/entropy when on; ``GlobalFilterBar`` in live progress panel (§G.10 / §G.7)
- [x] Gradient norm and entropy sparklines: `MiniSparkline` component (70 px ECharts, area fill at 13% opacity); grad_norm in red `#f87171`, entropy in purple `#a78bfa`; rendered as 2-column grid below `LiveChart`; PNG export on live chart and sparklines; component returns `null` when no data for the given metric key
- [x] On completion: "Output Browser →" button appears in live progress header when training completes successfully; navigates to `output_browser` mode
- [x] Session persistence: `useTrainHubStore` (Zustand `persist`, key `wsroute-train-hub`) stores all form fields across train/hpo/eval modes; ephemeral runtime state stays in component state
- [x] Live training health + runtime attention (§A.4 / §A.2): ``TrainingHealthPanel`` + ``RuntimeAttentionPanel`` in live progress panel during train/hpo; ``Training Monitor →`` navigation shortcut (hundred-thirtieth pass)
- [x] Live HPO label + ``HPO Tracker →`` navigation during live HPO runs (hundred-thirty-second pass)
- [x] ``TrainHpoNavMesh`` shared navigation + ``LiveTrainProgressBar`` epoch progress/ETA during live train/HPO (hundred-thirty-fifth pass; §D.2)
- [x] Post-run ``outputRunPath`` + ``trainingRunPath`` deep-links on Training Hub live panel (hundred-forty-fourth pass; §G.14 / §G.17 / §D.7)
- [x] Post-run panel persistence via ``findRecentHubProcessId`` (train/HPO/eval) when navigation clears local state (hundred-forty-sixth pass; §G.10 / §D.7)
- [x] Post-run grad-norm + LR sparklines via ``TrainingMetricSparklines``; ``TrainingMetricSnapshot`` + rehydration banner when train/HPO completes (hundred-forty-ninth pass; §G.17 / §D.7)
- [x] Post-run health/attention rehydration banner via ``postRunTrainingRehydrationMessage`` (hundred-fiftieth pass; §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoAnalyticsStrip`` shared live/post-run analytics strip (hundred-fifty-first pass; §G.10 / §D.7)
- [x] ``metric updates`` label ``text-accent-success`` styling parity with Process Monitor / HPO / Experiment Tracker (hundred-fifty-second pass; §G.10 / §D.7)
- [x] ``TrainHpoRehydrationBadges`` shared header badges for metric / health / attention rehydration counts (hundred-fifty-third pass; §G.10 / §D.7)
- [x] ``TrainHpoLivePanelHeader`` shared live panel header row with ``split`` layout + ``activity`` running icon (hundred-fifty-fourth pass; §G.10 / §D.7)
- [x] ``TrainHpoLivePanel`` shared live/post-run panel shell with ``footer`` process-id row + ``showAnalytics`` slots (hundred-fifty-sixth pass; §G.10 / §D.7)
- [x] ``ProcessIdFooter`` shared process-id footer row on Training Hub live panel (hundred-fifty-eighth pass; §G.10 / §D.7)

---

### §G.11 — Phase 11: Data Generation Wizard ✅

**Goal**: Port the PySide6 data generation tab to Tauri/React.

- [x] Script selector (generate_dataset / generate_bins / generate_routes) + extra CLI args textarea
- [x] `spawn_python_process` integration via `useSpawnProcess`; cancel and toasts
- [x] React form: problem selector (vrpp/wcvrp/scwcvrp/all), distribution checkboxes (Gamma-3/Empirical), dataset type selector (test_simulator/train/train_time), overwrite toggle; mirrors `gen_data.yaml`
- [x] Graph form: area selector (figueiradafoz/riomaior), num_loc, n_samples, n_days fields; configures `data.graphs[0]` via Hydra override
- [x] Advanced Overrides collapsible + command preview (`python main.py gen_data ...`)
- [x] TSPLIB source option: `dataSource` radio (synthetic / TSPLIB); `.vrp`/`.tsp` file picker via Tauri dialog; Hydra overrides `data.source=tsplib` + `data.tsplib_instance=<path>`; graph form hidden in TSPLIB mode
- [x] Sensor data source option: third `dataSource` radio; CSV file picker (timestamp,bin_id,fill_level,waste_type); Hydra overrides `data.source=sensor` + `data.sensor_file=<path>`
- [x] Preview panel: `preview_dataset_stats` Rust command + "Preview .pkl/.pt" button; KPI cards (instances, nodes, demand μ±σ, file size) + ECharts demand histogram with PNG export; demand histogram follows global ``logScale`` via ``GlobalFilterBar`` (§G.11 / §G.7)
- [x] Live progress: subscribes to `process:stdout` and `process:status` for the active generation run; shows last 20 stdout lines in a scrollable pre-block; status header with `Activity`/`CheckCircle`/`XCircle` icons; "Process Monitor" navigation button on completion
- [x] Session persistence: `useDataGenStore` (Zustand `persist`, key `wsroute-data-gen`) stores all form fields; ephemeral runtime state stays in component state
- [x] Live progress + ETA (hundred-thirty-seventh pass): ``LiveTrainProgressBar`` in live progress panel during ``gen_data`` runs (§D.2 / §G.11)
- [x] ``LauncherNavMesh`` + ``Data Explorer →`` post-run shortcut (hundred-thirty-ninth pass; §D.7)
- [x] ``LauncherNavMesh`` ``Output Browser →`` post-run shortcut on completed data generation runs (hundred-forty-second pass; §G.14 / §D.7)
- [x] Post-run Output Browser deep-link via ``outputRunPath`` + ``pendingRunPath`` when stdout contains a log path (hundred-forty-third pass; §G.14 / §D.7)
- [x] Post-run panel persistence via ``findRecentLauncherProcessId`` when navigation clears local state (hundred-forty-sixth pass; §G.11 / §D.7)
- [x] ``LauncherLivePanel`` shared live progress panel shell (hundred-fifty-seventh pass; §G.11 / §D.7)
- [x] ``ProcessIdFooter`` ``footer`` process-id row on Data Generation live panel (hundred-fifty-eighth pass; §G.11 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on Data Generation live panel (hundred-sixty-first pass; §G.11 / §D.7)

---

### §G.12 — Phase 12: Evaluation Runner ✅

**Goal**: Port the PySide6 evaluation tab and expose multi-checkpoint comparison.

- [x] Dynamic checkpoint list: add/remove entries, each with file picker (Tauri dialog; .pt/.ckpt/.pth)
- [x] Eval parameters: dataset path (optional, Tauri dialog), problem selector, decoding strategy (greedy/sampling/beam), device (cpu/cuda:0/cuda:1), val_size
- [x] Multi-checkpoint launch: one `spawn_python_process main.py eval` call per valid checkpoint, tagged with checkpoint filename; results stream to Process Monitor
- [x] Advanced Overrides collapsible + command preview (shows first-checkpoint invocation)
- [x] Results grid: global `process:stdout` listener parses JSON lines with `cost`/`gap`/`tour_cost`/`time`/`policy` fields; keyed by checkpoint name; dynamic column discovery from first result; updates in real time as results stream in
- [x] "Export CSV" button: builds CSV from result rows, triggers browser download via `Blob` + `URL.createObjectURL`
- [x] "Open in Analytics" button pre-loads eval results into BenchmarkAnalysis via `pendingEvalResults` store field; shows cost/gap/time bar charts + summary table
- [x] Inline results bar charts on Evaluation Runner results grid with per-metric PNG export (§G.12)
- [x] EvaluationRunner inline checkpoint charts follow global ``logScale``: log y-axis on cost/gap/time when on; ``GlobalFilterBar`` toggle above results grid (§G.12 / §G.7)
- [x] Live progress + ETA (hundred-thirty-eighth pass): per-checkpoint ``LiveTrainProgressBar`` in live progress panel during ``eval`` runs; multi-checkpoint aggregate status + stdout tail (§D.2 / §G.12)
- [x] ``LauncherNavMesh`` + ``Benchmark Analysis →`` post-run shortcut in live eval panel (hundred-thirty-ninth pass; §D.7)
- [x] ``evalResults.ts`` shared stdout JSON parsing + ``toEvalAnalyticsRows`` helpers (hundred-fortieth pass; §G.12 / §G.15)
- [x] Live progress per-checkpoint KPI row + ``LauncherNavMesh`` ``Output Browser →`` post-run shortcut (hundred-forty-first pass; §G.12 / §G.14 / §D.7)
- [x] ``checkpointPathFromEvalCommand`` + ``Load in Eval Runner →`` from completed eval processes (hundred-forty-first pass; §G.12 / §G.15)
- [x] Single-checkpoint live panel passes ``checkpointPath`` to ``LauncherNavMesh`` for post-run reload (hundred-forty-second pass; §G.12 / §D.7)
- [x] Post-run ``outputRunPath`` deep-link on Evaluation Runner live panel (hundred-forty-fourth pass; §G.14 / §D.7)
- [x] Multi-checkpoint batch persistence via ``findRecentEvalProcessIds`` + ``collectEvalResultFromLogLines`` when navigation clears local state (hundred-forty-sixth pass; §G.12 / §D.7)
- [x] ``LauncherLivePanel`` shared live progress panel shell for multi-checkpoint eval runs (hundred-fifty-seventh pass; §G.12 / §D.7)
- [x] ``ProcessIdFooter`` multi-process footer + ``EvalResultKpiRow`` per-checkpoint KPI row on Evaluation Runner live panel (hundred-fifty-eighth pass; §G.12 / §D.7)
- [x] ``EvalCheckpointLiveCard`` shared per-checkpoint live eval row on Evaluation Runner (hundred-fifty-ninth pass; §G.12 / §D.7)
- [x] ``processLogTail`` + Process Monitor ``EvalCheckpointLiveCard`` live eval parity (hundred-sixtieth pass; §G.12 / §G.15 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display inside ``EvalCheckpointLiveCard`` (hundred-sixty-first pass; §G.12 / §D.7)

---

### §G.13 — Phase 13: Configuration Editor (Hydra YAML) ✅

**Goal**: Provide a full-featured Hydra configuration editor so users never need to touch config files manually.

- [x] Three editor modes: Raw (editable textarea), Table (flat key-value, YAML parsed), Diff (compare two YAML files side-by-side)
- [x] File picker via Tauri dialog (YAML / TOML / CFG)
- [x] "Copy Overrides" button: serialises flat key=value lines to clipboard via `navigator.clipboard`
- [x] Config diff view: highlights changed keys between primary and comparison file (e.g. `pruned_config.yaml` from two different runs)
- [x] Rust `read_text_file` command for loading any text file
- [x] Rust `write_text_file` command: creates parent directories if needed; used by the Save button
- [x] "Save" button in toolbar: writes edited Raw content back to the opened file path; active only when unsaved edits exist (dirty state tracked via `savedContentRef`); `Save*` label indicates unsaved changes
- [x] Load the resolved Hydra config tree via `dump_hydra_config` Rust command (`main.py <task> --cfg job`); task selector + "Load via --cfg job" button in ConfigEditor toolbar
- [x] Form mode: fourth view toggle with typed widgets (boolean checkbox, number input, text input) inferred from flat YAML values; edits sync to Raw content via `rowsToYaml()` (OmegaConf schema introspection deferred)
- [x] Monaco Editor integration for the Raw YAML mode (§D.6 Option C): lazy-loaded `YamlEditor` with syntax highlighting and theme sync
- [x] "Apply to Launcher" button: target selector (Simulation Launcher / Training Hub / Data Generation); `applyConfigToLauncher()` maps flat YAML keys to Zustand store patches and navigates to the target page
- [x] ``Ctrl+S`` keyboard shortcut saves dirty config to disk when a file path is open (§D.7 / §G.13)

---

### §G.14 — Phase 14: Output Browser & Session Management ✅

**Goal**: Replace the PySide6 file system tab with a native file browser tailored to WSmart-Route's output directory structure.

- [x] Run list panel: `list_output_dirs` with name, path, created_at, size
- [x] File tree: `list_dir` command; lazy-loads subdirectory contents on expand; `Folder`/`FileText`/`File` icons by extension
- [x] File viewer: CSV files load via `load_csv_file` (table with 200-row preview); text/YAML/JSON via `read_text_file` (syntax-highlighted pre block)
- [x] Directory picker via Tauri dialog for browsing arbitrary directories (not just `assets/output/`)
- [x] Run metadata panel: auto-loads `pruned_config.yaml` (or `config.yaml`) when a run is selected; flat YAML parsed and filtered by `META_KEYS`; compact two-column card below the file tree
- [x] "Open in Sim Summary" button: shown for `.jsonl` files; sets `pendingLogPath` in app store then navigates to `simulation_summary` mode; `SimulationSummary` consumes `pendingLogPath` on mount via `useEffect`
- [x] Directory tree view: auto-expand `hydra/` on run selection; `sortEntries()` prioritises config and log artefacts; highlight `pruned_config.yaml` and `.jsonl` in the file tree
- [x] Simulation result summary: on `selectRun`, scans top-level entries for a `.jsonl` file ≤ 20 MB; reads it via `read_text_file`, parses each line as `DayLogEntry`, aggregates overflows / kg/km / profit per policy; displays a compact 3-column KPI table (policy / overflows / kg/km) below the config metadata card; overflows colour-coded (green = 0, amber = low, red > 20)
- [x] "Compare runs": per-run checkbox multi-select (≥2); `findRunJsonl()` locates logs in top-level or `hydra/`; navigates to BenchmarkAnalysis with `pendingBenchmarkLogs`
- [x] Session profiles (§D.4 Option C): `useSessionProfilesStore` persists named snapshots of all three launcher stores; save/load/delete UI in Output Browser sidebar (max 20 profiles)
- [x] Recent files/runs: `useRecentFilesStore` tracks last 12 opened logs, output runs, and CSVs; surfaced in command palette
- [x] Checkpoint browser (hundred-forty-second pass): auto-expand ``checkpoints/`` on run select; sidebar card lists ``.pt/.ckpt/.pth`` with **Eval →** shortcut; file tree highlights checkpoint artefacts; **Load in Eval Runner →** on selected checkpoint files via ``pendingCheckpoint`` (§G.14 / §G.12 / §G.17)
- [x] ``checkpoints.ts`` — shared ``isCheckpointEntry`` / ``filterCheckpointEntries`` helpers used by Output Browser + Training Monitor (§G.14 / §G.12)
- [x] ``outputRunPath.ts`` + ``pendingRunPath`` auto-select when opened from launcher / Process Monitor shortcuts (hundred-forty-third pass; §G.9 / §G.11 / §G.15 / §D.7)
- [x] Output Browser refreshes run list when ``pendingRunPath`` is set but the run is not yet indexed (hundred-forty-third pass; §G.14)
- [x] ``outputRunPathFromHydraArtifact`` + Hydra snapshot / pruned-config stdout parsing (hundred-forty-fourth pass; §G.14 / §G.9 / §G.12)

---

### §G.15 — Phase 15: Real-Time Process Monitor & Log Viewer ✅

**Goal**: Provide a unified view of all running and recently completed processes, replacing the PySide6 file-tailer pattern.

- [x] `ProcessRegistry` in Rust: global `OnceLock<Arc<Mutex<HashMap<String, (u32, Sender<bool>)>>>>`
- [x] `process:spawn` event emitted immediately after spawn (id, command, pid, start_time); `useProcessMonitor` hook registers process in store
- [x] `process:stdout` events for each stdout/stderr line; stored in per-process `logLines` (capped at 2000)
- [x] `process:status` event on completion/cancel/failure with exit code
- [x] Process list panel: status badge, command, inline log viewer (last 50 lines), cancel button
- [x] `cancel_process` Tauri command: sends `true` via watch channel → `child.kill()`
- [x] `which_python` resolves `<workingDir>/.venv/bin/python` first (uv-managed venv), then system PATH
- [x] Process list panel: full tabular layout — `StatusPill` + process ID + command + PID + live duration (`useLiveDuration` hook, 1s tick, stops when process ends) + exit code badge; sorted newest-first
- [x] Inline log viewer per process: expand/collapse toggle, auto-scroll checkbox, stderr lines coloured `text-accent-warning`; scroll locked at 2000 lines via process store
- [x] Structured log parsing: `LogLine` component tries `JSON.parse` on each line; if successful and has `level`/`msg`/`message` fields, renders timestamp (ISO prefix), colour-coded level badge (danger/warning/muted/gray), and message; falls back to plain text for non-JSON lines
- [x] Remove button per completed process row (`Trash2` icon); "Clear completed (N)" bulk action in the header
- [x] `clearCompleted` action added to process store: removes all non-running entries
- [x] Process history persistence: `useProcessStore` wrapped in Zustand `persist` middleware; `partialize` strips `logLines` and caps at last 50 completed processes; survives app restart
- [x] Progress bar per process: subscribe to structured progress events (epoch, day, instance count) emitted by the Python subprocess via stdout markers — `PROGRESS:{json}` protocol; `getLatestProgress()` scans last 30 log lines; deterministic bar when `total` is known, indeterminate pulse otherwise
- [x] Process row progress + ETA (hundred-thirty-sixth pass): ``LiveTrainProgressBar`` on each running process row; elapsed + ETA via shared ``processProgress.ts`` helpers (§D.2 / §G.15)
- [x] ``LauncherNavMesh`` return shortcuts for selected ``test_sim`` / ``gen_data`` / ``eval`` processes (hundred-thirty-ninth pass; §D.7)
- [x] Cancel any running process (§D.5): button in the process list row; sends SIGTERM (`cancel_process` command already wired in `ProcessRow`)
- [x] Toast notification on process completion / failure (§D.8): `useProcessMonitor` fires `toast.success/error/info` on terminal status transitions; label derived from `id.split("_")[0]`
- [x] Training analytics for ``train_`` / ``hpo_`` processes: ``TrainingHealthPanel`` + ``RuntimeAttentionPanel`` parsed from process stdout (§A.4 / §A.2 hundred-thirtieth pass)
- [x] Eval results panel (hundred-fortieth pass): selected ``eval`` processes parse structured JSON from stdout; KPI row (cost / gap / time / policy) + ``Open in Analytics →`` via ``pendingEvalResults`` (§G.12 / §G.15 / §D.7)
- [x] ``LauncherNavMesh`` ``Benchmark Analysis →`` on completed eval processes when metrics are present (hundred-fortieth pass; §D.7 / §G.12)
- [x] ``TrainHpoNavMesh`` ``Output Browser →`` on completed ``train_`` / ``hpo_`` processes (hundred-fortieth pass; §G.10 / §D.7)
- [x] ``LauncherNavMesh`` ``Output Browser →`` + ``Load in Eval Runner →`` on completed eval processes (hundred-forty-first pass; §G.12 / §G.14 / §D.7)
- [x] Process Monitor ``Output Browser →`` on completed ``test_sim`` / ``gen_data`` processes with run deep-link (hundred-forty-third pass; §G.9 / §G.11 / §G.14 / §D.7)
- [x] Process Monitor eval ``outputRunPath`` deep-link parity (hundred-forty-fourth pass; §G.12 / §G.14 / §D.7)
- [x] Process Monitor train/HPO ``outputRunPath`` + ``trainingRunPath`` deep-links on ``TrainHpoNavMesh`` (hundred-forty-fourth pass; §G.10 / §G.17 / §D.7)
- [x] Train/HPO metrics rehydration + grad-norm/LR sparklines on selected processes (hundred-forty-eighth pass; §G.15 / §G.17 / §D.7)
- [x] Post-run health/attention rehydration banner via ``postRunTrainingRehydrationMessage`` (hundred-fiftieth pass; §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoAnalyticsStrip`` shared analytics strip on selected train/HPO processes (hundred-fifty-first pass; §G.15 / §D.7)
- [x] ``TrainHpoRehydrationBadges`` shared header badges for metric / health / attention rehydration counts (hundred-fifty-third pass; §G.15 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoLivePanelHeader`` ``muted`` analytics subtitle header + badges-before-nav ordering parity (hundred-fifty-fourth pass; §G.15 / §D.7)
- [x] ``TrainHpoLivePanel`` ``embedded`` variant for selected train/HPO analytics section (hundred-fifty-sixth pass; §G.15 / §D.7)
- [x] ``LauncherLivePanelHeader`` ``embedded`` muted subtitle header + run-label + live suffix parity on selected ``test_sim`` processes (hundred-fifty-seventh pass; §G.9 / §G.15 / §D.7)
- [x] ``LauncherLivePanel`` ``embedded`` variant for selected ``test_sim`` / ``gen_data`` / ``eval`` analytics sections (hundred-fifty-seventh pass; §G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] ``EvalResultCard`` shared eval result card with ``Open in Analytics →`` on selected ``eval`` processes (hundred-fifty-eighth pass; §G.12 / §G.15 / §D.7)
- [x] ``EvalCheckpointLiveCard`` live progress + stdout tail on selected running ``eval`` processes; ``EvalResultCard`` on completion with metrics (hundred-sixtieth pass; §G.12 / §G.15 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on selected ``gen_data`` embedded workflow section (hundred-sixty-first pass; §G.11 / §G.15 / §D.7)

---

### §G.16 — Phase 16: Simulation Digital Twin Page

**Goal**: Full Streamlit `simulation` mode parity — real-time map, KPI dashboard, tour visualization, and bin-fill heatmap — superseding the basic SimulationMonitor scaffold implemented in Phase 0.

Source files ported from: `logic/src/ui/pages/simulation/{kpi,map,charts,bins,tour,summary_sections}.py`, `logic/src/ui/services/simulation_analytics.py`

- [x] **KPI dashboard** (`kpi.py` parity): primary group (profit, distance, waste, overflows) and secondary group (collections, waste lost, efficiency, cost); day-over-day delta badges; secondary group shown/hidden via toggle button
- [x] **Bin-fill strip chart**: top-25 bins sorted by fill descending; 0-100% horizontal bars colour-coded (green <80%, amber 80–99%, red ≥100%); mandatory (!) and collected (✓) badges per row; show/hide toggle
- [x] **Tour table**: stop #, bin ID, fill %, collected ✓/—, mandatory !/— columns; reads `tour_indices` preferentially, falls back to `tour`; limited to 60 rows with count shown; show/hide toggle
- [x] **Daily metrics chart**: ECharts `line` timeseries for all 4 primary KPIs across all loaded days; rendered as a 4-column grid
- [x] **Day scrubber**: ◀/▶ step buttons flanking the range slider; "Following" badge (green pulse) when `selectedDay` is null and watcher is active; "Latest ↓" button to release back to auto-follow
- [x] **Simulation Summary page** (`simulation_summary` mode) — rewritten with: sortable policy ranking table (mean ± std per metric, coloured policy dots); per-day trajectory overlay chart (all policies on one ECharts line chart, metric selector: overflows/profit/km/kg); trajectory chart follows global ``logScale`` (symlog overflows + log profit/km/kg); four metric bar charts with std dev in tooltip hover
- [x] **Route map preview** (ECharts scatter + path): Cartesian tour viz using `all_bin_coords` + `tour_indices`; fill-level colour coding; depot/tour/idle bin layers; PNG export
- [x] **Route map** (deck.gl `PathLayer`): `DeckRouteMap` renders tour path over MapLibre dark basemap; fill-level colour-coded `ScatterplotLayer` stops; idle bins as grey scatter; PNG export via `exportCanvasPng`; ECharts/deck.gl toggle in SimulationMonitor; lazy-loaded chunk (§G.16)
- [x] **Side-by-side route compare**: overlay/split layout toggle when exactly 2 policies visible; split renders labelled dual `DeckRouteMap` or dual ECharts panels (§G.16)
- [x] **Policy / Sample multi-select**: chip-toggle row shown when ≥2 policies present; `chartPolicies` state (default: all); `MetricTimeseries` refactored to accept `policySeries: { policy; entries; color }[]`; 8-colour `POLICY_COLORS` palette; ECharts legend shown when >1 series; detail panels (KpiCard, BinFill, TourTable) still use single `selectedPolicy` dropdown
- [x] **Streamlit parity check**: `PRIMARY_KPIS` and `SECONDARY_KPIS` in `SimulationMonitor.tsx` verified against `_PRIMARY_KPI_MAP` and `_SECONDARY_KPI_MAP` in `kpi.py` — exact match confirmed
- [x] **Daily KPI timeseries follow global ``logScale``**: ``MetricTimeseries`` symlog overflows + log profit/km/kg when on; ``GlobalFilterBar`` on Simulation Monitor (§G.16 / §G.7)
- [x] **deck.gl route map PNG export with toast feedback**: ``DeckRouteMap`` ``exportCanvasPng()`` names export ``route-map-tile.png`` (Mercator) or ``route-map-orbit.png`` (OrbitView) with toast feedback (§G.16 / §G.7)

**Status**: §G.16 complete — all checklist items delivered.

---

### §G.17 — Phase 17: Training Monitor Page ✅

**Goal**: Full Streamlit `training` mode parity — training run discovery, Lightning CSV metrics, hyperparameter inspection, and multi-run comparison.

Source files ported from: `logic/src/ui/pages/training.py`, `logic/src/ui/pages/training_charts.py`, `logic/src/ui/services/data_loader.py`

- [x] **Run discovery** (`discover_training_runs` parity): scan `<projectRoot>/logs/` for Lightning log directories; detect `metrics.csv` and `hparams.yaml`; checkbox multi-select
- [x] **Metrics CSV loading**: `load_training_metrics` Rust command parses Lightning `metrics.csv`; epoch/step x-axis; train_loss, val_loss, reward columns handled
- [x] **Multi-run overlay chart**: single ECharts canvas with one colour-coded series set per run (8-colour palette); train loss (solid), val loss (dashed), reward (dotted, right y-axis); scrollable legend; PNG export; replaces one-chart-per-run layout
- [x] **Global log-scale on training charts**: ``MultiRunChart`` log loss axis + grad-norm/LR sparklines log y-axis when global ``logScale`` on; ``GlobalFilterBar`` on Training Monitor (§G.17 / §G.7)
- [x] **Gradient norm sparkline**: separate compact ECharts chart for `grad_norm` column, shown per selected run
- [x] **Hyperparameter panel**: reads `hparams.yaml` via `read_text_file`; collapsible; flat `key: value` parser; shows first 8 rows with "Show all" expand; skips comment lines
- [x] **Checkpoint browser**: `list_dir` on `<run.path>/checkpoints/`; filters to `.pt/.ckpt/.pth`; shows name + file size; "Load in Eval Runner →" button sets `pendingCheckpoint` in app store and switches to `eval_runner` mode
- [x] **Learning rate schedule chart**: `lr` column rendered as a compact `LrSparkline` (step-level, amber `#fbbf24`) using the shared `MetricSparkline` base component; shown per selected run below the gradient norm sparkline
- [x] **Live training mode**: `LIVE_KEY = "__live__"` virtual entry in `metricsMap`; `activeTrainId` from `useProcessStore` (newest running `train_*` or `hpo_*` process via ``findActiveLiveTrainProcessId``); `process:stdout` listener appends parsed metric rows to `metricsMap[LIVE_KEY]` without touching the CSV; live entry auto-selected in run list with `Radio` icon + pulse animation; ``Live HPO`` label when an ``hpo_*`` process is active; live `RunPanel` shows `GradNormSparkline` + `LrSparkline`; auto-deselected when process exits (hundred-thirty-first pass extends HPO coverage)
- [x] **Column normalization**: `normalizeMetricRow()` maps Lightning CSV aliases (`train/rl_loss` → `train_loss`, `val/cost` → `val_loss`, `lr-Adam` → `lr`) applied at both CSV load time and live stdout parse time; same normalization applied to `TrainingHub.tsx`
- [x] **Streamlit parity check**: Lightning CSV columns `train_loss`, `val_loss`, `reward`, `grad_norm`, `lr`, `epoch`, `step` all rendered; aliased column variants covered by `normalizeMetricRow`
- [x] ``pendingTrainingRunPath`` auto-select when opened from Training Hub / Process Monitor train/HPO shortcuts (hundred-forty-fourth pass; §G.10 / §G.15 / §D.7)
- [x] Post-run ``outputRunPath`` + ``trainingRunPath`` deep-links on live/recent train panel; auto-select completed run from stdout ``trainingRunPath`` (hundred-forty-fifth pass; §G.10 / §G.14 / §D.7)
- [x] Post-run metrics/health/attention rehydration from ``useProcessStore`` when live streaming state clears; multi-run overlay chart persists via ``effectiveLiveMetrics`` (hundred-forty-seventh pass; §G.17 / §D.7)
- [x] ``TrainingMetricSparklines`` shared grad-norm + LR sparklines used across Process Monitor and analytics pages (hundred-forty-eighth pass; §G.15 / §G.18 / §D.7)
- [x] Training Monitor deduplicated to shared ``TrainingMetricSparklines`` + ``TrainingMetricSnapshot``; post-run sparkline rehydration banner parity (hundred-forty-ninth pass; §G.10 / §D.7)
- [x] Post-run health/attention rehydration banner via ``postRunTrainingRehydrationMessage`` (hundred-fiftieth pass; §A.2 / §A.4 / §D.7)
- [x] Live/recent card ``TrainHpoAnalyticsStrip`` for post-run sparkline rehydration without ``LIVE_KEY`` selection (hundred-fifty-first pass; §G.17 / §D.7)
- [x] ``metric updates`` label parity across Training Hub + Training Monitor (hundred-fifty-first pass; §G.10 / §D.7)
- [x] Post-run health/attention banner counts via rehydrated entries on ``TrainHpoAnalyticsStrip`` (hundred-fifty-second pass; §G.17 / §A.2 / §A.4 / §D.7)
- [x] ``metric updates`` label on non-checkbox live/recent header + Training Hub accent-success styling (hundred-fifty-second pass; §G.10 / §G.17 / §D.7)
- [x] ``TrainHpoRehydrationBadges`` shared header badges for metric / health / attention rehydration counts (hundred-fifty-third pass; §G.17 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoLivePanelHeader`` deduplicated live/recent header with ``overlaySelect`` ``LIVE_KEY`` checkbox parity (hundred-fifty-fifth pass; §G.17 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoLivePanel`` shared live/post-run panel shell with ``overlaySelect`` + ``showHealthAttention={false}`` options (hundred-fifty-sixth pass; §G.17 / §A.2 / §A.4 / §D.7)

---

### §G.18 — Phase 18: Experiment & HPO Tracker ✅

**Goal**: Full Streamlit `experiment_tracker` and `hpo_tracker` mode parity — MLflow/ZenML run browser, Optuna study visualization, and cross-experiment comparison.

Source files ported from: `logic/src/ui/pages/experiment_tracker.py`, `logic/src/ui/pages/experiment_tracker_charts.py`, `logic/src/ui/pages/hpo_tracker.py`

- [x] **MLflow run table** (`experiment_tracker.py` parity): Rust queries MLflow via Python subprocess (`mlflow.search_runs`); display runs with params, metrics, tags, artifact path
- [x] **Metric comparison chart**: select two or more MLflow runs; overlay their logged metrics as ECharts line series; metric name selector and Y-axis normalization toggle
- [x] **ZenML pipeline view** (if ZenML is configured): `list_zenml_pipeline_runs` + `load_zenml_run_steps` Rust commands; pipeline run table; step-duration horizontal bar chart (Gantt-style) with log duration axis when global ``logScale`` on; CSV/PNG export (§G.18 / §G.7)
- [x] **Optuna study browser** (`hpo_tracker.py` parity): `list_optuna_studies` + `load_optuna_study` Rust commands call Optuna via Python subprocess; trials serialised to JSON; HPOTracker displays:
  - Parallel coordinates plot (`echarts` `parallel` series) across hyperparameter dimensions
  - Optimization history scatter plot (trial number vs. objective value) with best-so-far line
  - Parameter importance bar chart (FANOVA via `optuna.importance.get_param_importances`)
- [x] **HPO charts follow global ``logScale``**: optimisation history + cross-study best-so-far lines + parallel-coordinates objective axis use log objective when on; ``GlobalFilterBar`` on HPO Tracker (§G.18 / §G.7)
- [x] **MLflow metric comparison follows global ``logScale``**: multi-run overlay chart log y-axis on loss/objective metrics when on; ``GlobalFilterBar`` on Experiment Tracker (§G.18 / §G.7)
- [x] **Best-trial highlight**: best value KPI card; "Copy best params" button writes trial `params` as Hydra override lines to clipboard
- [x] **Cross-study comparison**: "Compare with" study dropdown in HPOTracker; overlaid best-so-far optimisation history (ECharts); side-by-side best-value KPI cards for both studies
- [x] **MLflow dashboard embed fallback**: Runs/Dashboard tab toggle in ExperimentTracker; iframe embed of local MLflow UI (`http://localhost:5000` default) + open-in-browser via shell plugin (native WebView window deferred)
- [x] **Live HPO analytics** (§A.4 / §A.2 hundred-thirty-first pass): ``TrainingHealthPanel`` + ``RuntimeAttentionPanel`` when an ``hpo_*`` process is running; ``Process Monitor →`` navigation shortcut
- [x] **Experiment Tracker live HPO analytics** (§A.4 / §A.2 hundred-thirty-second pass): health + attention panels during live ``hpo_*``; ``HPO Tracker →`` + ``Process Monitor →`` shortcuts
- [x] **Cross-page train/HPO navigation** (hundred-thirty-second pass): Training Monitor ``Process Monitor →`` + ``HPO Tracker →``; HPO Tracker ``Training Monitor →``; Process Monitor ``Training Monitor →`` + ``HPO Tracker →`` for ``hpo_*`` processes
- [x] **Experiment Tracker navigation mesh** (hundred-thirty-third pass): Experiment Tracker ``Training Monitor →``; HPO Tracker / Training Monitor / Process Monitor / Training Hub ``Experiment Tracker →`` when live HPO active (§G.10 / §G.15 / §G.17 / §G.18)
- [x] **Training Hub navigation mesh** (hundred-thirty-fourth pass): Training Monitor / Process Monitor / HPO Tracker / Experiment Tracker ``Training Hub →`` during live train/HPO workflows — completes bidirectional cross-page shortcuts (§G.10 / §G.15 / §G.17 / §G.18)
- [x] **Live epoch progress + ETA** (hundred-thirty-fifth pass): ``LiveTrainProgressBar`` on Training Hub / Training Monitor / HPO Tracker / Experiment Tracker; ``processProgress.ts`` shared with Process Monitor (§D.2 / §G.17 / §G.18)
- [x] **Train/HPO keyboard shortcuts** (hundred-thirty-fifth pass): ``T`` Training Monitor · ``H`` Training Hub · ``E`` Experiment Tracker (§D.7)
- [x] Post-run ``outputRunPath`` + ``trainingRunPath`` deep-links on HPO Tracker + Experiment Tracker live panels when sweep completes (hundred-forty-fifth pass; §G.14 / §G.17 / §D.7)
- [x] Live metric snapshot row + update count from ``collectTrainingMetricsFromLogLines`` on persisted HPO process stdout (hundred-forty-seventh pass; §G.18 / §G.17 / §D.7)
- [x] Post-run grad-norm + LR sparklines from persisted HPO stdout via ``TrainingMetricSparklines`` (hundred-forty-eighth pass; §G.18 / §G.17 / §D.7)
- [x] ``TrainingMetricSnapshot`` deduplication + ``postRunTrainingRehydrationMessage`` health/attention banner parity (hundred-fiftieth pass; §G.17 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoAnalyticsStrip`` shared live/post-run analytics strip (hundred-fifty-first pass; §G.18 / §G.17 / §D.7)
- [x] ``TrainHpoRehydrationBadges`` shared header badges for metric / health / attention rehydration counts (hundred-fifty-third pass; §G.18 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoLivePanelHeader`` deduplicated live HPO header blocks on HPO Tracker + Experiment Tracker (hundred-fifty-fourth pass; §G.18 / §G.17 / §D.7)
- [x] ``TrainHpoLivePanel`` shared live/post-run panel shell on HPO Tracker + Experiment Tracker (hundred-fifty-sixth pass; §G.18 / §G.17 / §D.7)

---

### §G.19 — Phase 19: Settings & First-Run Onboarding ✅

**Goal**: Provide a persistent Settings page so users can configure the project root and Python executable without touching environment variables, and surface a first-run onboarding banner.

- [x] Settings page (`pages/Settings.tsx`): Project Root section (text input + directory picker via Tauri dialog), Python Executable section (text input + path picker, overrides `which_python` resolution), Appearance section (dark/light theme radio), About section (Studio version + links)
- [x] `store/app.ts`: `projectRoot` and `pythonPath` fields persisted via Zustand `persist` + `partialize`; `setPythonPath` action
- [x] `pythonPath` threaded through `useSpawnProcess` → `spawn_python_process` Rust command as `python_executable: Option<String>`; empty string treated as `None` (falls back to `which_python`)
- [x] First-run banner in `TopBar.tsx`: shown when `projectRoot` is empty and mode is not `"settings"`; links directly to Settings with an "Open Settings" button
- [x] Sidebar "App" section with Settings entry and gear icon
- [x] `system::validate_project_root` Rust command: checks path exists, is a directory, contains `main.py`; called on blur + before save; shows inline `CheckCircle` / `XCircle` badge
- [x] `system::probe_python` Rust command: runs `<path> --version` synchronously, handles Python 2 (stderr) and Python 3 (stdout); shows resolved version string inline; called on blur + before save
- [x] Save blocked if either validation fails; toast shown with "Fix validation errors before saving"
- [x] Import/export settings JSON: "Export Settings" serialises `{projectRoot, pythonPath, theme}` to a user-chosen JSON file via `write_text_file`; "Import Settings" reads a JSON file and populates drafts for review before saving
- [x] First-run onboarding wizard: `OnboardingDialog` modal when `projectRoot` is empty; inline directory picker + validation; dismissible via `useLayoutStore.onboardingDismissed` persistence
- [x] Guided tour: `GuidedTour` 5-step overlay with `data-tour` spotlight rings (sidebar, command palette, simulation twin, output browser, launch/monitor); TopBar compass button, command palette action, Settings "Take Guided Tour", `Ctrl+Shift+/` shortcut; auto-offered after first onboarding; dismissal persisted via `guidedTourDismissed`
- [x] System theme following (§D.3 Option C): `theme` preference extended to `dark` / `light` / `system`; `effectiveTheme` resolves `prefers-color-scheme` via `useThemeSync`; TopBar + command palette cycle all three modes; Settings Appearance radio includes System (§G.19 / §D.3)

**Status**: §G.19 complete — all checklist items delivered.

---

### §G — Studio Complete ✅

All twenty phases (§G.0–§G.19) are delivered. WSmart-Route Studio is the primary desktop interface for launching simulations and training runs, browsing results, and performing post-hoc analytics. Post-§G analytics bridges continue under §A (e.g. §A.3 Policy Telemetry in hundred-ninth pass; §A.5 Optuna Plotly export in hundred-tenth pass; §A.4 Training Health in hundred-eleventh pass; §A.6 Failure Analysis in hundred-twelfth pass; §A.2 WandB attention heatmaps in hundred-thirteenth pass; §A.1 Route Solution visualizer in hundred-fourteenth pass; §A.6 route-diff failure overlay in hundred-fifteenth pass; §A.6 ECharts route-diff parity in hundred-sixteenth pass; §A.2 Studio attention ring-buffer in hundred-seventeenth pass; §A.4 HPO health prune metrics in hundred-eighteenth pass; §A.3 live policy telemetry stream in hundred-nineteenth pass; §A.3 SQLite cross-run telemetry trending in hundred-twentieth pass; §A.3 cross-run improvement trajectory chart in hundred-twenty-first pass; §A.3 trajectory brush + Benchmark Analysis panel in hundred-twenty-second pass; §A.3 chart brush filter + Simulation Summary panel in hundred-twenty-third pass; §A.3 chart brush dimming + Algorithm/City Comparison panels in hundred-twenty-fourth pass; §A.3 run_label brush sync + OLAP Explorer panel in hundred-twenty-fifth pass; §A.3 trajectory click fix + Simulation Monitor / Data Explorer panels in hundred-twenty-sixth pass; §A.3 Output Browser trends panel + KPI brush in hundred-twenty-seventh pass; §A.3 Process Monitor telemetry + Output Browser run_label auto-brush in hundred-twenty-eighth pass; §A.3 Simulation Launcher live telemetry + Process Monitor brush parity in hundred-twenty-ninth pass). Remaining release-engineering items (code-signing keys, hosted signed update CDN) are deferred per §G.8.

| Area | Status |
| --- | --- |
| Analytics dashboard (§G.1–§G.2) | ✅ |
| Geospatial + graph topology (§G.3–§G.4) | ✅ |
| ML introspection (§G.5) | ✅ |
| OLAP explorer (§G.6) | ✅ |
| UX polish + export surface (§G.7) | ✅ |
| Data packaging (§G.8) | ✅ (signing keys + CDN deferred) |
| Launchers + monitors (§G.9–§G.15) | ✅ |
| Streamlit parity pages (§G.16–§G.18) | ✅ |
| Settings + onboarding (§G.19) | ✅ |

---

### §G — Dependency Map

```
Phase 0  →  All phases
Phase 1  →  Phase 2, Phase 6
Phase 2  →  Phase 7
Phase 3  →  Phase 4
Phase 4  →  Phase 5 (pheromone + attention share Sigma.js)
Phase 5  →  Phase 7
Phase 6  →  Phase 7
Phase 7  →  Phase 8
Phase 15 →  Phase 9, Phase 10, Phase 11, Phase 12, Phase 16 (all share process streaming)
Phase 9  →  Phase 14 (on-completion navigates to Output Browser)
Phase 10 →  Phase 14 (on-completion opens checkpoint in Output Browser)
Phase 13 →  Phase 9, Phase 10, Phase 11, Phase 12 (config editor feeds all launchers)
Phase 14 →  Phase 1 (analytics dashboard load)
Phase 16 →  Phase 3 (map uses deck.gl from §G.3)
Phase 17 →  Phase 10 (training hub spawns; monitor reads)
Phase 18 →  Phase 1, Phase 17 (builds on analytics dashboard and training runs)
```

---

### Effort × Impact Matrix — WSmart-Route Studio

| Phase | Description | Effort | Impact | Priority |
| --- | --- | --- | --- | --- |
| §G.0 | Foundation & Tooling | High | Very High | P0 |
| §G.19 | Settings & Onboarding | Low | Very High | P0 |
| §G.15 | Real-Time Process Monitor | Medium | Very High | P0 |
| §G.9 | Simulation Launcher | Medium | Very High | P1 ✅ |
| §G.10 | Training & HPO Hub | Medium | Very High | P1 ✅ |
| §G.13 | Configuration Editor | Medium | High | P1 ✅ |
| §G.14 | Output Browser | Medium | High | P1 ✅ |
| §G.1 | Statistical Dashboard | High | High | P1 ✅ |
| §G.11 | Data Generation Wizard | Low | High | P2 ✅ |
| §G.12 | Evaluation Runner | Low | High | P2 ✅ |
| §G.7 | UX Polish | Medium | High | P2 ✅ |
| §G.2 | Drill-Down Sunburst | Medium | High | P2 ✅ |
| §G.3 | Geospatial deck.gl | High | High | P2 ✅ |
| §G.6 | OLAP Explorer | Medium | High | P2 ✅ |
| §G.4 | Graph Topology | Medium | Medium | P3 ✅ |
| §G.8 | Export & Packaging | Medium | High | P3 ✅ |
| §G.5 | ML Introspection | High | High | P3 ✅ |
| §G.16 | Simulation Digital Twin | High | Very High | P1 ✅ |
| §G.17 | Training Monitor | Medium | Very High | P1 ✅ |
| §G.18 | Experiment & HPO Tracker | High | High | P2 ✅ |

---

## Cross-Cutting Themes

Several items across sections are tightly coupled and should be sequenced together:

| Cluster                        | Items                          | Rationale                                                                                |
| ------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------- |
| **Plugin System**              | §B.3, §B.6                     | Policy and env registration should share the same Hydra-based mechanism                  |
| **Async Worker Contract**      | §B.8, §D.5, §G.15              | Rust AsyncTask trait + Python BackgroundTask protocol are prerequisites for cancel UX    |
| **Route Visualization**        | §A.1, §D.1, §G.3               | All three need the same spatial renderer; deck.gl in §G.3 is the shared base             |
| **Docs Infrastructure**        | §C.1, §C.7                     | MkDocs setup is a prerequisite for the CI docs pipeline                                  |
| **Test Quality**               | §B.1, §F.3                     | Coverage uplift and test-suite speed are best addressed together                         |
| **Telemetry**                  | §A.3, §A.4                     | PolicyVizMixin and TrainingHealthCallback both feed the Studio analytics dashboard        |
| **Config System**              | §B.3, §B.6, §D.6, §G.13        | Plugin registry + Hydra `_target_` + Studio config editor all depend on a clean config schema |
| **Process Streaming**          | §G.9, §G.10, §G.11, §G.12, §G.15, §G.16 | All launchers share the same Rust→React stdout streaming infrastructure from §G.15 |
| **Streamlit Parity**           | §G.16, §G.17, §G.18            | These three phases are a 1:1 port of the three most-used Streamlit modes; complete before removing Streamlit dependency |

---

_This roadmap is a living document. Update item status inline (✅ Done, 🚧 In Progress, ❌ Blocked) and refresh the Effort × Impact matrices each quarter._
