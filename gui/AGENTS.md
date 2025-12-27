# GUI Submodule Agents & Components

This document documents the "Active" components of the GUI in `gui/src`. Ideally, business logic should be offloaded to `logic/`, but these components handle the translation between User Intent and System Action.

## 1. Background Workers (gui/src/helpers/)
Non-blocking agents that run in separate threads (`QThread`) to keep the UI responsive.

| Worker Name | File | Signals | Responsibilities |
| :--- | :--- | :--- | :--- |
| **ChartWorker** | `helpers/chart_worker.py` | `data_ready`, `finished` | **Data Visualizer**. Parses real-time simulation logs (JSON/CSV) and emits coordinates for live plotting of Profit, Cost, and Waste metrics. |
| **DataLoaderWorker** | `helpers/data_loader_worker.py` | `data_loaded`, `error` | **Async I/O**. Loads massive datasets (Distance Matrices, Graph Geometries) in the background interaction during startup. |
| **FileTailerWorker** | `helpers/file_tailer_worker.py` | `new_lines`, `file_changed` | **Log Streamer**. Implements a `tail -f` equivalent to stream stdout/stderr from external processes into the Main Window Console. |

## 2. Windows (gui/src/windows/)
Top-level containers that manage the application lifecycle.

| Window Name | File | Description |
| :--- | :--- | :--- |
| **MainWindow** | `windows/main_window.py` | **App Root**. The primary application container. Manages the main menu, navigation sidebar, and the central `QTabWidget` hosting all functional tabs. |
| **TSResultsWindow** | `windows/ts_results_window.py` | **Simulation Dashboard**. A specialized window that pops up after a Simulation Test. Displays Heatmaps, Route Maps (Folium), and Aggregate Statistics for the run. |

## 3. Functional Tabs (gui/src/tabs/)
The primary interaction surfaces for the user.

| Tab Group | File | Purpose |
| :--- | :--- | :--- |
| **Analysis** | `tabs/analysis/*.py` | Input/Output analysis. visualizations of problem instances and training convergence curves. |
| **Evaluation** | `tabs/evaluation/*.py` | Interfaces for running inference (`eval.py`) on trained models and comparing decoding strategies (Greedy vs Beam Search). |
| **TestSimulator** | `tabs/test_simulator/*.py` | **The Core Simulation UI**. Configures and launches `main.py test_sim`. Includes settings for Policies, Days, Vehicles, and Visualizations. |
| **TestSuite** | `tabs/test_suite.py` | Provides a gallery view of simulation results, allowing comparison of different policies (e.g., AM-GAT vs Gurobi) side-by-side. |
| **MetaRLTrain** | `tabs/meta_rl_train.py` | UI wrapper for `mrl_train` commands. Configures Meta-Learning experiments. |
| **HyperparamOptim** | `tabs/hyperparam_optim.py` | UI wrapper for `hp_optim`. Allows users to define search spaces (Grid/Random) and launch HPO jobs. |
| **FileSystem** | `tabs/file_system/*.py` | Utilities for managing local Datasets, Models, and experimental results (CRUD operations). |
