# WSmart+ Route — Moon Roadmap

> **Version**: 1.0  
> **Last Updated**: June 2026  
> **Status**: Living Document — updated each sprint  
> **Scope**: Logic layer (`logic/src/`), GUI layer (`gui/src/`), CI/CD, documentation

This document captures medium-to-long-horizon improvements for the WSmart+ Route framework across six dimensions: Analytics & Interpretability, Architecture, Documentation, GUI/UX, New Features, and Performance. Each item follows a **Pain → Options → Recommendation** structure with effort/impact tags.

Tags: `[Quick Win]` ≤ 1 day · `[Research]` involves novel work · `[Blocked]` depends on another item

---

## Anchor Index

| Section | Topic |
|---------|-------|
| [§A — Analytics & Interpretability](#a--analytics--interpretability) | Telemetry, attention maps, policy dashboards, HPO analytics |
| [§B — Architecture](#b--architecture) | Test coverage, plugin system, logging, type safety, interfaces |
| [§C — Documentation](#c--documentation) | API docs, architecture diagrams, Jupyter notebooks, CI docs pipeline |
| [§D — GUI / UX](#d--gui--ux) | Route visualization, training progress, themes, session persistence |
| [§E — New Features](#e--new-features) | Multi-problem benchmarking, REST API, LLM integration, export formats |
| [§F — Performance](#f--performance) | Batched inference, GPU memory, test suite speed, simulation throughput |

---

## A — Analytics & Interpretability

### §A.1 — Interactive Route Solution Visualizer

**Pain**: Solutions (tours, collected bins, costs) are currently logged as JSON arrays. There is no visual overlay of routes on a spatial canvas, making debugging decoder outputs and comparing policies against each other extremely tedious.

**Options**

- **A** — Embed a Matplotlib FigureCanvas panel inside the existing `analysis/` GUI tab; render depot, nodes, edges, and colour-code routes per vehicle. Low friction, already uses FigureCanvas in `input_analysis.py`.
- **B** — Export solutions to GeoJSON and open them in a browser via Folium/Leaflet, decoupled from the Qt event loop.
- **C** — Use Plotly Dash as a standalone web dashboard, running in a background process launched from `main.py`.
- **D** — Integrate `rerun.io` for a time-scrubbing 2-D trajectory viewer (works well with simulation day-by-day replay).
- **E** — Add a pure-PySide6 `QGraphicsScene`/`QGraphicsView` canvas with zoom, pan, and node tooltips.

**Recommendation**: **Option A first** (lowest cost, reuses existing matplotlib integration), then **Option E** as the production-quality widget — `QGraphicsView` scales better and stays in the same process. Option D is interesting for multi-day simulation replay but requires an external runtime.

**Effort × Impact**: Medium effort / High impact

---

### §A.2 — Attention Map Visualization for Neural Decoders

**Pain**: The AM, TAM, DDAM, and MoE decoders compute multi-head attention over node embeddings, but these attention weights are never exported or displayed. Without visibility into what the model attends to, diagnosing routing errors or comparing trained heads is guesswork.

**Options**

- **A** — Hook `nn.MultiheadAttention` outputs with forward hooks; buffer the last batch's attention tensors in a ring-buffer on the model object. Visualize as a heatmap in the GUI analysis tab.
- **B** — Integrate `BertViz`-style row-column attention visualizer adapted for graph problems (node × node matrix).
- **C** — Log attention weights to WandB / TensorBoard as image summaries during evaluation; no GUI integration needed.
- **D** — Export attention weights to `.npz` per inference call and build a separate offline viewer script.

**Recommendation**: **Option C** for fast iteration (zero GUI work), then **Option A** for the GUI integration once Option C has validated that the data is interpretable. Option B is academic-grade but requires a browser runtime.

**Effort × Impact**: Low effort (Option C) → Medium effort (Option A) / High impact

---

### §A.3 — Policy Telemetry Dashboard (Extension of `PolicyVizMixin`)

**Pain**: `logic/src/tracking/viz_mixin.py` already records per-iteration metrics (cost, feasibility, elapsed time) into a fixed-capacity ring-buffer via `_viz_record()`. However, this data is only accessible programmatically through `get_viz_data()` and is never surfaced to the user during or after a run.

**Options**

- **A** — Wire `get_viz_data()` output into the existing `analysis/` tab: after a simulation run, populate a QTableView / matplotlib bar chart with per-policy metrics (cost trajectories, improvement curves). `[Quick Win]`
- **B** — Emit ring-buffer snapshots over a `QueueListener` + `SocketHandler` to a TUI panel refreshed at 2 Hz while the simulation runs.
- **C** — Persist ring-buffer dumps to a SQLite database (`assets/telemetry.db`) and query them across runs for cross-policy trending.
- **D** — Push telemetry to Prometheus and visualize in Grafana (overkill for single-machine runs).

**Recommendation**: **Option A** immediately (hours of work), **Option C** for multi-run analytics once the database schema is stable.

**Effort × Impact**: Very Low effort (Option A) / High impact

---

### §A.4 — RL Loss Landscape & Training Health Monitoring

**Pain**: The Lightning-based RL pipeline (`logic/src/pipeline/rl/`) logs loss values but provides no automated detection of training instability (exploding/vanishing gradients, policy collapse, reward stagnation). Researchers must manually inspect WandB logs.

**Options**

- **A** — Add a `TrainingHealthCallback` (Lightning callback) that raises structured warnings when: gradient norm > 100, reward moving average stagnates for > 50 epochs, entropy < threshold. Log to the structured logging system.
- **B** — Use `PyHessian` to compute the top-K Hessian eigenvalues of the policy network periodically; log sharpness as a training health proxy. `[Research]`
- **C** — Visualize the loss landscape slice (perturbation method) after training completes; save as a PNG artefact to `assets/analysis/`.
- **D** — Add gradient norm and entropy to the existing WandB sweep metrics so Optuna / DEHB can prune unhealthy runs early.

**Recommendation**: **Option A** is a mandatory baseline — training health guardrails belong in every production RL pipeline. **Option D** pairs naturally with HPO (already integrated) and costs one additional metric log line. **Option B/C** are research-grade extras.

**Effort × Impact**: Low–Medium effort / High impact

---

### §A.5 — HPO Analytics: Cross-Trial Visualizer

**Pain**: The HPO module supports Optuna, Ray Tune, and DEHB, but the results are stored as trial databases without a unified post-hoc analysis view. Users cannot easily compare hyperparameter importance or visualize Pareto frontiers across objectives.

**Options**

- **A** — Use `optuna.visualization` (already a transitive dependency) to render parallel-coordinates and importances plots; export to `assets/hpo_reports/`. `[Quick Win]`
- **B** — Add a dedicated HPO Analysis tab in the GUI wrapping the Optuna visualization calls in a QWebEngineView (Plotly HTML output).
- **C** — Export all trial results to a Pandas DataFrame; add a `hpo_summary.ipynb` notebook template that loads and plots them.
- **D** — Integrate SHAP to compute hyperparameter contribution scores across trials. `[Research]`

**Recommendation**: **Option A** for immediate wins (one function call with Optuna's built-in plotting), **Option C** as the notebook companion for sharing results.

**Effort × Impact**: Very Low effort (Option A) / Medium impact

---

### §A.6 — Causal Simulation Failure Analysis

**Pain**: When a simulation day ends with overflows or negative profit, the root cause (fill-rate spike, capacity miscalculation, policy sub-optimality) is not automatically identified. Post-hoc debugging requires re-reading JSON logs line by line.

**Options**

- **A** — Add a `FailureAnalyzer` class to `logic/src/pipeline/simulations/` that, after each day, compares predicted vs. actual bin fill levels, flags bins that caused overflow, and writes a structured summary to the day's JSON log entry.
- **B** — Build a counterfactual engine: re-run the day with the optimal policy (Gurobi) whenever a heuristic fails, and log the gap. `[Research]`
- **C** — Visualize the failure mode as a route-diff overlay in the GUI (bins that were skipped vs. bins that overflowed highlighted in red). Depends on §A.1.
- **D** — Use causal inference (DoWhy) to identify which features (fill_rate, capacity, graph_size) most predict failure across simulation episodes. `[Research]`

**Recommendation**: **Option A** is purely additive and requires no new dependencies — pure logic in the existing simulator. **Option C** is the natural follow-on once §A.1 is implemented.

**Effort × Impact**: Medium effort / High impact

---

### Effort × Impact Matrix — Analytics & Interpretability

| Item | Effort | Impact | Priority |
|------|--------|--------|----------|
| §A.3 Option A (PolicyVizMixin → GUI) | Very Low | High | P0 |
| §A.5 Option A (Optuna plots) | Very Low | Medium | P0 |
| §A.4 Option A (TrainingHealthCallback) | Low | High | P1 |
| §A.2 Option C (WandB attention heatmaps) | Low | High | P1 |
| §A.6 Option A (FailureAnalyzer) | Medium | High | P1 |
| §A.1 Option A (Matplotlib route viz) | Medium | High | P2 |
| §A.1 Option E (QGraphicsView canvas) | High | High | P2 |
| §A.4 Option B (PyHessian) | High | Medium | P3 `[Research]` |
| §A.6 Option B (counterfactual engine) | Very High | High | P3 `[Research]` |

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

**Pain**: Adding a new classical policy (e.g., a new metaheuristic) requires modifying multiple files: the policy registry, the CLI argument parser, the GUI dropdown list, and the simulation runner. There is no single registration point.

**Options**

- **A** — Define a `@register_policy(name, problem_types)` decorator that writes to a module-level dict in `logic/src/policies/__init__.py`; CLI and GUI query this dict at runtime. `[Quick Win]`
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
- **C** — Use `pyright` (faster, better PyTorch generics support) alongside MyPy; make pyright the blocking check and MyPy the advisory check.
- **D** — Use `beartype` for runtime type enforcement at public API boundaries (interfaces module). Catches issues that static analysis misses.

**Recommendation**: **Option A** for gradual strictness adoption; **Option D** as a runtime safety net for the interfaces layer where type errors cause silent mathematical bugs.

**Effort × Impact**: Medium effort / High impact

---

### §B.6 — Environment Plugin System (Analogous to §B.3)

**Pain**: Adding a new problem environment (e.g., a new VRP variant) requires modifying `logic/src/envs/problems.py`, the CLI parser, the data generator, and the GUI environment selector — no single registration point.

**Options**

- **A** — Define a `@register_env(name, problem_class)` decorator and a central env registry; CLI/GUI consult it at startup.
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

### §B.8 — Worker Thread Base Class Standardization

**Pain**: GUI background workers (`data_loader_worker.py`, `chart_worker.py`, `file_tailer_worker.py`) each implement `QThread` independently with inconsistent error propagation, progress signal patterns, and cancellation logic.

**Options**

- **A** — Define a `BaseWorker(QThread)` in `gui/src/helpers/base_worker.py` with: `progress = Signal(int)`, `error = Signal(str)`, `result = Signal(object)`, `_cancelled: bool`, and a `cancel()` method. Subclasses override `run_task()`. `[Quick Win]`
- **B** — Switch to `QThreadPool` + `QRunnable` for short-lived tasks; `QThread` only for persistent workers (file tailer).
- **C** — Use Python `concurrent.futures.ThreadPoolExecutor` managed by a Qt-aware bridge class.

**Recommendation**: **Option A** is the right Qt idiom and eliminates the current boilerplate fragmentation with minimal refactor cost.

**Effort × Impact**: Low effort / Medium impact

---

### Effort × Impact Matrix — Architecture

| Item | Effort | Impact | Priority |
|------|--------|--------|----------|
| §B.7 Option A (pydeps CI) | Very Low | Medium | P0 `[Quick Win]` |
| §B.1 Option C (per-module coverage floors) | Very Low | High | P0 `[Quick Win]` |
| §B.4 Option A (remove print() calls) | Low | Medium | P0 `[Quick Win]` |
| §B.8 Option A (BaseWorker) | Low | Medium | P1 `[Quick Win]` |
| §B.5 Option A (strict MyPy, utils subpackage) | Medium | High | P1 |
| §B.2 Option C (benchmark visibility) | Low | High | P1 |
| §B.3 Option D (Hydra policy plugin) | Medium | High | P2 |
| §B.6 Option B (Hydra env plugin) | Medium | High | P2 |
| §B.1 Option D (Hypothesis property tests) | High | High | P2 `[Research]` |
| §B.2 Option A (benchmark regression gate) | Medium | High | P2 |

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

**Pain**: `docs/ARCHITECTURE.md` describes the system in prose. There are no visual diagrams showing the data flow from CLI → Pipeline → Environment → Model → Policy, or the GUI mediator pattern, making onboarding slow.

**Options**

- **A** — Embed Mermaid flowcharts directly in `docs/ARCHITECTURE.md`; GitHub renders them natively in Markdown. Add: training data flow, inference pipeline, simulation orchestration, and GUI mediator diagrams. `[Quick Win]`
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

**Pain**: `docs/TROUBLESHOOTING.md` and `docs/COMPATIBILITY.md` exist but their content is unclear (referenced in `CLAUDE.md` but not validated in this audit). CUDA version conflicts, Gurobi license errors, and PySide6 display backend issues are the most common friction points for new contributors.

**Options**

- **A** — Audit both files; add sections for: Gurobi 11+ license setup, CUDA 12.x / PyTorch 2.2 compatibility matrix, `QT_QPA_PLATFORM=xcb` fix, `uv sync` common errors, and HGS/PyVRP installation issues.
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

| Item | Effort | Impact | Priority |
|------|--------|--------|----------|
| §C.3 Option A (CHANGELOG.md) | Very Low | Medium | P0 `[Quick Win]` |
| §C.2 Option C (ruff D rules) | Very Low | Medium | P0 `[Quick Win]` |
| §C.4 Option A (Mermaid diagrams) | Low | High | P0 `[Quick Win]` |
| §C.6 Option A (TROUBLESHOOTING refresh) | Low | Medium | P1 |
| §C.1 Option A (MkDocs Material) | Medium | High | P1 |
| §C.7 Option A (docs CI job) | Low | High | P2 (after §C.1) |
| §C.5 Option A (Jupyter notebooks) | High | High | P2 |
| §C.5 Option B (nbval CI) | Low | High | P2 (after §C.5) |

---

## D — GUI / UX

### §D.1 — Route Visualization Panel

**Pain**: The GUI's analysis tab shows dataset statistics and fill-rate charts, but has no panel for visualizing computed routes. After running a simulation or evaluation, users must read JSON output to understand what routes were computed.

**Options**

- **A** — Add a `RouteVizTab` (or panel inside the existing `analysis/` tab) with a Matplotlib canvas: plot depot (star), customer nodes (circles sized by demand), and route edges (colour per vehicle). Load routes from simulation JSON output. Synergises with §A.1.
- **B** — Implement a `QGraphicsView`-based canvas (`RoutePainter`) that supports zoom, pan, and node tooltip with fill level / demand. Better UX, more code.
- **C** — Open an external browser to a locally-served Plotly map each time the user clicks "Visualize". Zero Qt widget work but breaks the desktop-app UX.

**Recommendation**: **Option A** immediately, **Option B** as the production upgrade. Option C is a fallback only.

**Effort × Impact**: Medium effort / High impact

---

### §D.2 — Training Progress Enhancements

**Pain**: The `reinforcement_learning/` tab shows training progress via a file-tailer (reading stdout logs), but the UX is a plain `QTextEdit`. There is no live loss curve, no epoch progress bar, and no ETA display.

**Options**

- **A** — Parse the structured JSON log emitted by the training pipeline and update: a Matplotlib live-updating loss/reward chart (redraw every N seconds), a `QProgressBar` for epoch progress, and a computed ETA label. `[Quick Win]` for the progress bar; more work for live chart.
- **B** — Add a `TrainingMetricsWorker` (using `BaseWorker` from §B.8) that reads from the WandB run API in real time and populates the chart.
- **C** — Embed a `QWebEngineView` that renders the live WandB dashboard. Requires a browser runtime in the Qt process.

**Recommendation**: **Option A** — parse structured logs (already JSON-formatted) for zero external dependency. The `file_tailer_worker.py` already tails logs; add a JSON parser layer on top.

**Effort × Impact**: Medium effort / High impact

---

### §D.3 — Dark / Light Theme Toggle

**Pain**: The `gui/src/styles/themes/` directory already contains `dark.py` and `light.py` theme modules, but there is no runtime toggle exposed to the user. The application appears to launch with a hardcoded theme.

**Options**

- **A** — Add a theme selector to the application menu bar (`View → Theme → Dark / Light`); apply via `QApplication.setStyleSheet()` at runtime. `[Quick Win]`
- **B** — Persist the selected theme to a user preferences file (`assets/preferences.json`) so it is restored on next launch. Synergises with §D.4.
- **C** — Add a system-theme-following mode using `QStyleHints.colorScheme()` (Qt 6.5+).

**Recommendation**: **Option A + B** together — both are trivial once the theme infrastructure exists.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

---

### §D.4 — Session Persistence

**Pain**: When the GUI is closed and re-opened, all configured parameters (problem type, model path, dataset path, number of days, policy selections) are reset to defaults. Users must reconfigure every session.

**Options**

- **A** — Serialize the current widget state (all combo-box selections, line-edit values, checkbox states) to `assets/last_session.json` on `closeEvent()`; restore on startup. `[Quick Win]`
- **B** — Use `QSettings` (platform-native key-value store) for the same purpose — no manual JSON I/O.
- **C** — Allow users to name and save multiple "session profiles" (e.g., "VRPP-50-nodes", "WCVRP-simulation").

**Recommendation**: **Option B** first (idiomatic Qt, one-liner per widget), **Option C** for power users.

**Effort × Impact**: Low effort / High impact

---

### §D.5 — Progress & Cancellation for Long Operations

**Pain**: Data generation, training, and simulation runs can take hours. Users have no mechanism to cancel a running operation without killing the process, and there is no progress indicator for operations that don't emit epoch-level logs.

**Options**

- **A** — Add a `cancel()` method to all `QThread` workers (standardized by §B.8's `BaseWorker`); expose a "Cancel" button in the GUI that calls it. Use `threading.Event` to signal workers. `[Quick Win]`
- **B** — For multiprocessing-based operations (simulation uses `multiprocessing`), use a `multiprocessing.Event` or `Manager().Event()` as a shared cancellation flag.
- **C** — Show a modal `QProgressDialog` with a cancel button for operations with known total steps; show an indeterminate spinner for open-ended operations.

**Recommendation**: **Option A + C** — the `BaseWorker` from §B.8 provides the cancel mechanism; the progress dialog provides the UX.

**Effort × Impact**: Medium effort / High impact

---

### §D.6 — Configuration Panel for Hydra Overrides

**Pain**: The GUI exposes only a subset of the available Hydra configuration options. Advanced users who want to override `train.batch_size`, `model.embedding_dim`, or `env.num_loc` must edit config files or use the CLI — bypassing the GUI entirely.

**Options**

- **A** — Add an "Advanced" collapsible panel (`QGroupBox`) in each tab that renders a `QTableWidget` of key-value overrides. Users can add/edit/delete rows; the worker translates them to Hydra override strings (`key=value`). `[Quick Win]`
- **B** — Parse the Hydra config schema (`OmegaConf`) at runtime and generate a typed form (dropdowns for enums, sliders for numeric ranges, checkboxes for bools).
- **C** — Show a raw YAML editor (`QPlainTextEdit`) that is passed directly as a Hydra config file.

**Recommendation**: **Option A** for immediate usefulness (generic override table), **Option B** as the polished version once the config schema introspection is stable.

**Effort × Impact**: Medium effort / High impact

---

### §D.7 — Keyboard Shortcuts & Command Palette

**Pain**: All GUI operations require mouse clicks. Power users running repeated experiments have no keyboard-driven workflow.

**Options**

- **A** — Assign `QShortcut` bindings to common actions: `Ctrl+R` (run), `Ctrl+S` (save config), `Ctrl+.` (cancel), `Ctrl+T` (switch tab). Add a "Shortcuts" entry to the Help menu. `[Quick Win]`
- **B** — Implement a command palette (`Ctrl+Shift+P`) backed by a `QListView` of all registered actions + their shortcuts; filter by typing. Qt 6 has no built-in command palette, so this requires a custom floating widget.

**Recommendation**: **Option A** first (one `QShortcut` call per action), **Option B** if the GUI grows beyond 10 top-level tabs.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

---

### §D.8 — Toast Notifications for Background Completions

**Pain**: When a training job or data generation task finishes in the background, there is no notification. Users must switch to the output tab to check if the job completed.

**Options**

- **A** — Implement a `ToastNotification` overlay widget (`QLabel` with slide-in animation, auto-dismiss after 5s) shown in the bottom-right corner on job completion/failure. `[Quick Win]`
- **B** — Use the system tray (`QSystemTrayIcon`) to show a native OS notification balloon.
- **C** — Play an OS sound notification via `QSoundEffect`.

**Recommendation**: **Option A + B** — the in-app toast for when the app is in focus, system tray notification for when the user switches away.

**Effort × Impact**: Low effort / High impact

---

### Effort × Impact Matrix — GUI / UX

| Item | Effort | Impact | Priority |
|------|--------|--------|----------|
| §D.3 Option A+B (theme toggle + persist) | Very Low | Medium | P0 `[Quick Win]` |
| §D.7 Option A (keyboard shortcuts) | Very Low | Medium | P0 `[Quick Win]` |
| §D.4 Option B (QSettings persistence) | Low | High | P0 |
| §D.8 Option A+B (toast + tray) | Low | High | P1 |
| §D.5 Option A+C (cancel + progress) | Medium | High | P1 |
| §D.2 Option A (live training charts) | Medium | High | P1 |
| §D.1 Option A (Matplotlib route panel) | Medium | High | P2 |
| §D.6 Option A (override table) | Medium | High | P2 |
| §D.1 Option B (QGraphicsView canvas) | High | High | P2 |
| §D.6 Option B (typed config form) | High | High | P3 |

---

## E — New Features

### §E.1 — Multi-Problem Benchmarking Suite

**Pain**: Comparing neural models (AM, TAM, DDAM, MoE) against classical policies (ALNS, HGS, Gurobi) across all three problem types (VRPP, WCVRP, SCWCVRP) and multiple graph sizes requires manually running multiple `main.py eval` commands and aggregating CSV results by hand.

**Options**

- **A** — Add a `benchmark` subcommand to `main.py` that: runs a configurable matrix of (policy × problem × graph_size), collects metrics, and writes a unified `benchmark_report.csv` and Markdown table. `[Quick Win]`
- **B** — Integrate with `ray[tune]` sweep (already a dependency) to parallelize the benchmark matrix across CPU cores.
- **C** — Add a "Benchmark" tab to the GUI (synergises with §A.5) that configures the matrix via checkboxes and shows a live results table.

**Recommendation**: **Option A** for the CLI benchmark runner, **Option C** for GUI-accessible results.

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

| Item | Effort | Impact | Priority |
|------|--------|--------|----------|
| §E.6 Option B (richer instance generator) | Low | High | P0 |
| §E.1 Option A (CLI benchmark runner) | Medium | High | P1 |
| §E.5 Option A (sensor data loader) | Medium | Very High | P1 |
| §E.7 Option A (transfer eval command) | Low | High | P1 `[Research]` |
| §E.2 Option B+C (TSPLIB loader) | Medium | Very High | P2 |
| §E.5 Option B (fill-rate calibration) | Medium | Very High | P2 |
| §E.4 Option C (contextual bandit policy selection) | Medium | Very High | P2 `[Research]` |
| §E.3 Option A (FastAPI server) | High | High | P3 |
| §E.4 Option B (MetaRNN online adaptation) | Very High | Very High | P3 `[Research]` |
| §E.6 Option C (conditional generator) | Very High | High | P3 `[Research]` |

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

| Item | Effort | Impact | Priority |
|------|--------|--------|----------|
| §F.1 Option A (inference_mode wrapper) | Very Low | High | P0 `[Quick Win]` |
| §F.2 Option A+C (GPU memory logging) | Very Low | High | P0 `[Quick Win]` |
| §F.6 Option A (complexity refactor top-10) | Low | Medium | P0 `[Quick Win]` |
| §F.3 Option C (profile test durations) | Low | High | P1 |
| §F.4 Option B (DataLoader pinned memory) | Low | High | P1 |
| §F.7 Option A (CPU sync audit) | Low | High | P1 |
| §F.3 Option B (fast/full CI split) | Low | High | P1 |
| §F.1 Option B (torch.compile) | Medium | High | P2 |
| §F.4 Option A (pkl → pt format) | Low | Medium | P2 |
| §F.5 Option D (simulation profiling) | Low | High | P2 |
| §F.5 Option B (Pool.starmap refactor) | Medium | High | P2 |
| §F.4 Option D (cache distance matrices) | Medium | High | P2 |
| §F.1 Option D (TensorRT export) | High | Very High | P3 |
| §F.5 Option A (SharedMemory refactor) | High | High | P3 `[Research]` |
| §F.5 Option C (GPU-vectorized simulation) | Very High | Very High | P3 `[Research]` |

---

## Cross-Cutting Themes

Several items across sections are tightly coupled and should be sequenced together:

| Cluster | Items | Rationale |
|---------|-------|-----------|
| **Plugin System** | §B.3, §B.6 | Policy and env registration should share the same Hydra-based mechanism |
| **Worker Standardization** | §B.8, §D.5 | `BaseWorker` is a prerequisite for cancellation UX |
| **Route Visualization** | §A.1, §D.1 | Both need the same spatial rendering component; build once, use twice |
| **Docs Infrastructure** | §C.1, §C.7 | MkDocs setup is a prerequisite for the CI docs pipeline |
| **Test Quality** | §B.1, §F.3 | Coverage uplift and test-suite speed are best addressed together |
| **Telemetry** | §A.3, §A.4 | PolicyVizMixin and TrainingHealthCallback both feed the analysis dashboard |

---

*This roadmap is a living document. Update item status inline (✅ Done, 🚧 In Progress, ❌ Blocked) and refresh the Effort × Impact matrices each quarter.*
