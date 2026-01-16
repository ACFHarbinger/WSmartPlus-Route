# WSmart-Route Architecture

WSmart-Route is a high-performance framework designed to solve complex Combinatorial Optimization (CO) problems, specifically the Vehicle Routing Problem with Profits (VRPP) and Capacitated Waste Collection VRP (CWC VRP), by bridging Deep Reinforcement Learning (DRL) with Operations Research (OR).

## Technology Stack

- **Runtime**: Python 3.9+ managed by `uv`.
- **DRL/DL**: PyTorch (CUDA acceleration optimized for RTX 3090/4080).
- **OR Solvers**: Gurobi Optimizer and Hexaly.
- **GUI**: PySide6 (Qt for Python) with Folium for map visualization.
- **Data**: Pandas, NumPy, and Scipy for simulation physics and data processing.

## System Layers

### 1. Logic Layer (`logic/src/`)
The "brain" of the application, responsible for optimization, simulation physics, and neural learning.

- **models/**: Neural architectures (GAT, Attention Models, Meta-RNNs).
- **problems/**: Environment "Physics" (VRPP, WCVRP, CWCVRP).
- **pipeline/**: Orchestrates high-level workflows:
    - `train.py`: DRL training loops.
    - `eval.py`: Model evaluation.
    - `simulator/`: The physics engine (Day runner, Bins management, Action sequences).
- **policies/**: Collection of solvers including ALNS, Branch-Cut-and-Price, and Neural Agents.

### 2. GUI Layer (`gui/src/`)
The interaction surface, translating user intent into optimization tasks.

- **windows/**: Main application window and results dashboards.
- **tabs/**: Functional views (Simulator UI, Training UI, Analysis, Evaluation).
- **helpers/**: Asynchronous background workers (`ChartWorker`, `DataLoaderWorker`) ensuring a responsive UI.
- **core/mediator.py**: Orchestrates communication between the main window and functional tabs.

## Design Patterns

### State Pattern (Simulation Lifecycle)
Located in `logic/src/pipeline/simulator/states.py`.
Manages the transition between `InitializingState` (setup), `RunningState` (day-by-day execution), and `FinishingState` (result persistence).

### Command Pattern (Simulation Actions)
Located in `logic/src/pipeline/simulator/actions.py`.
Encapsulates discrete steps of a simulation day (`FillAction`, `PolicyExecutionAction`, `CollectAction`, `LogAction`) into executable objects, ensuring a decoupled and extensible pipeline.

### Mediator Pattern (GUI Orchestration)
Located in `gui/src/core/mediator.py`.
Centralizes communication between the main UI container and individual tabs, decoupling component logic and managing global command previews.

## Data Flow

1. **Input**: User configures simulation parameters in the `TestSimulator` tab.
2. **Setup**: GUI triggers `InitializingState` to load coordinates, distance matrices, and neural models.
3. **Loop**: `RunningState` executes `run_day` iteratively.
    - **Fill**: Bins accumulate waste based on stochastic or empirical distributions.
    - **Policy**: A `NeuralAgent` or OR solver determines the optimal collection route.
    - **Collect**: The simulator updates bin states and calculates costs/profits.
    - **Log**: Results are streamed to the GUI via the `ChartWorker`.
4. **Completion**: `FinishingState` generates final heatmaps and performance reports.
