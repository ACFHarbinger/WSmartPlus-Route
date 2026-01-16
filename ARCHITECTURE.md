# WSmart-Route Architecture

WSmart-Route is a high-performance framework designed to solve complex Combinatorial Optimization (CO) problems, specifically the Vehicle Routing Problem with Profits (VRPP) and Capacitated Waste Collection VRP (CWC VRP). It bridges the gap between Deep Reinforcement Learning (DRL) and Operations Research (OR) by providing a unified environment for training, benchmarking, and deploying intelligent agents alongside classical solvers.

## 1. High-Level Overview

The system operates on a hybrid architecture where DRL agents learn to construct solutions or gate classical heuristics. It supports:
- **Simulation**: A detailed event-driven simulator for waste collection logistics over temporal horizons (e.g., 365 days).
- **Optimization**: A suite of solvers ranging from exact methods (Branch-Cut-and-Price) to metaheuristics (ALNS, HGS) and DRL (Attention Models).
- **Interaction**: A PySide6 GUI for visualization and a modular CLI/TUI for headless execution and orchestration.

## 2. Technology Stack

- **Runtime**: Python 3.9+ (Managed by `uv`).
- **Deep Learning**: PyTorch 2.2+ (CUDA-optimized for NVIDIA Ampere/Ada architectures).
- **OR Solvers**: Gurobi Optimizer 11+, Hexaly, OR-Tools, `alns` package.
- **Data Engineering**: Pandas, NumPy, Scipy, GeoPandas.
- **GUI**: PySide6 (Qt for Python), Folium (Maps), Matplotlib (Charts).
- **CLI/TUI**: `argparse`, `rich`, `prompt-toolkit`.

---

## 3. System Layers

### 3.1. Logic Layer (`logic/src/`)
The core computational engine, strictly separated from UI concerns.

#### **Pipeline Orchestration (`pipeline/`)**
- **Simulation Engine (`pipeline/simulator/`)**:
    - **Physics**: `Bins` class manages stochastic waste accumulation (Gamma/Empirical distributions) and collection logic.
    - **Context**: `SimulationDayContext` encapsulates the state of a single simulation step.
    - **Lifecycle**: Implements the **State Pattern** (`InitializingState`, `RunningState`, `FinishingState`) to manage the simulation flow in `simulation.py`.
    - **Actions**: Implements the **Command Pattern** (`actions.py`) to decouple simulation steps (Fill, Policy, Collect, Log).
- **Training (`pipeline/train.py`)**:
    - Manages the DRL training loop, including curriculum learning and baseline updates.
    - **Reinforcement Learning (`pipeline/reinforcement_learning/`)**:
        - `WorkerTrain` & `ManagerTrain`: Distinct trainers for the routing agent (Worker) and the high-level gating agent (Manager).
        - **Meta-Learning**: `RewardWeightOptimizer` and `WeightContextualBandit` dynamically adjust objective weights (Cost vs. Waste) during training.
        - **HPO**: `hyperparameter_optimization/` implements DEHB and Ray Tune integrations.

#### **Neural Models (`models/`)**
- **Architecture**: Based on the Encoder-Decoder paradigm (Transformer/GNN).
- **Components**:
    - **Encoders**: `GATEncoder` (Graph Attention), `GCNEncoder`, `DeepGCNEncoder`.
    - **Decoders**: `AttentionDecoder` (Pointer Network style), `GATDecoder`.
    - **MoE**: Mixture of Experts layers for specialized sub-task learning.

#### **Policies & Solvers (`policies/`)**
Standardized interface returning `(routes, profit, cost)`:
- **Exact**: `BCPSolver` (Branch-Cut-and-Price), `GurobiOptimizer`.
- **Metaheuristics**:
    - **HGSSolver**: Hybrid Genetic Search with specialized split algorithms (`LinearSplit`) for VRPP.
    - **ALNSSolver**: Adaptive Large Neighborhood Search with custom destroy/repair operators.
- **Heuristics**: `LookAhead` (Rolling Horizon), `LastMinute` (Reactive).
- **Neural**: `NeuralAgent` wraps PyTorch models for inference within the simulator.

### 3.2. CLI Layer (`logic/src/cli/`)
A modular command-line interface with a strictly typed registry.
- **Parsers**: Dedicated parsers for each subcommand (`train_parser.py`, `sim_parser.py`, `gui_parser.py`).
- **Registry**: `registry.py` centralizes command dispatch.
- **TUI**: `tui.py` provides an interactive Terminal User Interface using `rich` for argument configuration and validation schemas.

### 3.3. GUI Layer (`gui/src/`)
A multi-threaded desktop application for visualization and control.
- **Architecture**: MVvm-ish with Mediator pattern.
- **Core**: `UIMediator` (`core/mediator.py`) handles signals between the `MainWindow` and functional `Tabs`.
- **Workers** (`QThread`):
    - `ChartWorker`: Asynchronously processes simulation metrics for live plotting.
    - `DataLoadWorker`: Handles heavy I/O (Excel/CSV parsing) off the main thread.
    - `FileTailerWorker`: Streams stdout/stderr to the GUI console.
- **Tabs**:
    - `TestSimulatorTab`: Configures and launches `test_sim`.
    - `EvaluationTab`: Model inference visualization.
    - `AnalysisTab`: Dataset inspection (Maps, Distributions).

---

## 4. Key Design Patterns

| Pattern | Implementation | Purpose |
| :--- | :--- | :--- |
| **State** | `logic/src/pipeline/simulator/states.py` | Manages complex simulation lifecycle states (Init -> Run -> Finish). |
| **Command** | `logic/src/pipeline/simulator/actions.py` | Encapsulates simulation steps as objects, allowing flexible composition of a "Day". |
| **Mediator** | `gui/src/core/mediator.py` | Decouples GUI components; tabs communicate via the mediator, not directly. |
| **Strategy** | `logic/src/policies/*` | Interchangeable algorithms (HGS, ALNS, DRL) implementing a common `solve` interface. |
| **Factory** | `logic/src/models/model_factory.py` | Centralized creation of complex neural architectures and subnets. |
| **Observer** | `Checkpoints` (via Hooks) | The checkpoint manager observes the simulation state to persist progress. |

---

## 5. Data Flow

### Simulation Pipeline
1.  **Configuration**: User command (CLI/GUI) is validated by `sim_parser.py` -> `Configs` object.
2.  **Initialization**: `InitializingState` loads:
    -   **Static Data**: Distance matrices (OSM/Haversine), Depot location.
    -   **Dynamic Data**: `Bins` created from `bins_waste/` (Historical fill rates).
    -   **Model**: Neural weights loaded if policy is DRL-based.
3.  **Day Loop (`simulation.py` -> `run_day`)**:
    -   **Stochastic Fill**: Bins fill based on probabilistic models.
    -   **Policy Execution**: The selected Strategy (e.g., `HGSSolver`) receives the graph state (distances, demands).
    -   **Optimization**: Solver returns `routes` maximizing VRPP objective (Profit - Cost).
    -   **Validation**: Simulator verifies constraints (Capacity, Time Windows).
    -   **Update**: `Bins.collect(routes)` empties bins; metrics (Kg, Km, Overflows) are calculated.
4.  **Logging**: `get_daily_results` aggregates metrics; `ChartWorker` emits to GUI; `WandB` logs if enabled.
5.  **Termination**: Results saved to Excel/JSON; Summary heatmaps generated.

## 6. Directory Structure Overview

```text
WSmart-Route/
├── gui/                  # Graphical User Interface
│   ├── src/
│   │   ├── components/   # Reusable Widgets
│   │   ├── core/         # Mediator & Signals
│   │   ├── helpers/      # Background Workers
│   │   ├── tabs/         # Functional UI Screens
│   │   └── windows/      # Main Application Windows
├── logic/                # Core Application Logic
│   ├── src/
│   │   ├── cli/          # Command Line Interface & TUI
│   │   ├── models/       # PyTorch Geometric/Neural Models
│   │   ├── pipeline/     # Simulator, Train, Eval loops
│   │   ├── policies/     # Optimization Algorithms (OR/RL)
│   │   ├── problems/     # Environment Physics (VRPP/CWC)
│   │   └── utils/        # Math, I/O, Transforms
│   └── test/             # Unit & Integration Tests
├── data/                 # Dataset storage (Ignored by Git)
├── assets/               # Simulation outputs & logs
├── pyproject.toml        # Project Configuration
└── justfile              # Command Automation
```
