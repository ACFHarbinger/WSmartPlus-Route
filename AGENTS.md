# AGENTS.md - Instructions for Coding Assistant LLMs

## 1. Project Overview & Mission
WSmart+ Route is a high-performance framework for solving complex Combinatorial Optimization (CO) problems, specifically the Vehicle Routing Problem with Profits (VRPP) and Capacitated Waste Collection VRP (CWC VRP).
The project mission is to bridge Deep Reinforcement Learning (DRL) with Operations Research (OR). It provides a benchmarking and deployment environment where neural models (PyTorch) interact with traditional solvers (Gurobi, Hexaly).

## 2. Technical Stack & Environmental Governance
Runtime: Python 3.9+ managed strictly via uv. Use uv sync for dependency resolution.
Primary Frameworks:
- DRL/DL: PyTorch (2.2.2) optimized for NVIDIA RTX 4080 (CUDA acceleration).
- OR Solvers: Gurobi Optimizer (11.0.3) and Hexaly.
- GUI: PySide6 (Qt for Python).

Quality Control:
- Linter: ruff (Mandatory compliance).
- Formatter: black.

Testing: pytest (Rooted in logic/test and gui/test).

## 3. Core Architectural Boundaries
Maintain strict separation of concerns across these primary modules:
- **Logic Layer (logic/src/)**
    - models/: Neural architecture implementations.
    - subnets/: Discrete components like Encoders (GAT, GCN, MLP) and Decoders.
    - modules/: Atomic utility layers (Normalization, Multi-Head Attention).
    - problems/: The environment "Physics."
    - state_*.py: Critical logic for state transitions, node masking, and reward calculation.
    - policies/: Traditional heuristic and exact algorithms (ALNS, Branch-Cut-and-Price).
    - pipeline/: Orchestration logic for train.py, eval.py, and test_sim.
- **GUI Layer (gui/src/)**
    - tabs/: Module-specific UI views (Analysis, Training, Simulator).
    - helpers/: Threaded workers (e.g., chart_worker.py) for non-blocking UI.
    - windows/: Application window management.

## 4. Key CLI Entry Points (Operational Playbook)
Always reference these commands when proposing code changes or workflows:
| Action | Command |
| --- | --- |
| Sync Environment | uv sync |
| Data Generation | python main.py generate_data virtual --problem vrpp --graph_sizes 50 |
| Model Training | python main.py train --model am --problem vrpp --graph_size 50 |
| Simulation Test | python main.py test_sim --policies regular gurobi alns --days 31 |
| Launch GUI | python main.py gui |

## 5. External Access and Browser Usage Rules
The agent is authorized to use the following external tools to assist in development:

### Web Search and Documentation
Authorization: Use the @search tool (Google Search) to retrieve the latest documentation for Gurobi 11+, PySide6 API references, and PyTorch 2.2+ best practices.

Verification: When suggesting a solution involving a third-party library, perform a search to verify that the suggested API methods are not deprecated.

Problem Solving: If a CUDA error or a specific Linux driver conflict (NVIDIA 550/560) is detected in logs, use web search to find relevant GitHub issues or system-level workarounds for Ubuntu 24.04.

### Knowledge Cutoff Management
Directive: Always cross-reference your internal training data with a web search if the technology was updated after January 2024 (e.g., specific Gurobi performance tunings).

## 6. Domain-Specific Coding Standards
### Mathematical & DRL Integrity
- Invalid Move Prevention: Decoders must implement masking via logic/src/utils/boolmask.py before sampling nodes to ensure feasibility.
- Activation Scaling: Prefer custom modules in logic/src/models/modules/normalization.py over generic nn.LayerNorm for consistency across 1M token context logic.
- State Transitions: Never modify state_*.py files without ensuring logic/test/test_problems.py still passes.

### Performance & Hardware
- GPU Offloading: Optimization targets the RTX 3090ti and the RTX 4080 (laptop version). Ensure tensors are explicitly moved to device using setup_utils.py.
- GUI Threading: Heavy computations (training/loading) must inherit from QThread. Never run blocking CO logic on the main Qt thread.

## 7. AI Review & Severity Protocol
Categorize your feedback and edits using these severity levels:
- CRITICAL: Breaking state_*.py transition logic; exposing credentials; cryptographic flaws in fs_cryptography.py.
- HIGH: CUDA memory leaks; incorrect skip_connection.py usage; pyproject.toml version mismatches.
- MEDIUM: Suboptimal Pandas operations in pandas_model.py; deviations from ruff formatting.
- LOW: Documentation typos; redundant imports; UI padding/margin adjustments in globals.py.

## 8. Known Constraints & "No-Go" Areas
- Legacy Preservation: Never edit files with copy.py suffixes or those inside legacy/ folders.
- Slurm Sensitivity: Cluster scripts (scripts/slurm.sh) use specific path mappings; verify before modifying.
- Linux Stability: In the Linux environment, always include --use-angle=vulkan and --disable-gpu-sandbox when suggesting debug flags for the GUI.

## 9. Usage Note
Reference this file during project-wide analysis. When starting a new terminal session, kill any previous terminal instances, and always run `source .venv/bin/activate` to ensure your environment is correctly activated. You can use `uv sync` to update your environment, and a map of the project structure is provided in the `project_map.txt` file.

## 10. Logic Submodule Architecture

This document maintains a registry of the intelligent agents, orchestration components, and environment physics within the `logic/src/` submodule.

### 1. Pipeline Orchestrators (logic/src/pipeline/)
These components manage the training, evaluation, and testing lifecycles.

| Agent Name | File | Responsibilities |
| :--- | :--- | :--- |
| **Trainer** | `pipeline/train.py` | The central entry point for all training operations (`train`, `mrl_train`, `hp_optim`). Manages device selection, data loading, and the outer epoch loop. |
| **Evaluator** | `pipeline/eval.py` | Handles model evaluation on test datasets. Supports `greedy`, `sampling`, and `beam_search` decoding strategies. |
| **Tester** | `pipeline/test.py` | Orchestrates high-level simulation testing across multiple seeds and policies. Supports parallel execution and checkpointing. |

### 2. Neural Models (logic/src/models/)
These are the deep learning agents that learn to solve routing problems.

| Agent Name | File | Architecture | Function |
| :--- | :--- | :--- | :--- |
| **GATLSTManager** | `models/gat_lstm_manager.py` | GAT + LSTM | **High-Level Agent**. Processes temporal states (history of bin levels) and outputs a "Gating Probability" to decide *when* to activate the routing worker. |
| **AttentionModel** | `models/attention_model.py` | Transformer (Encoder-Decoder) | **Worker Agent**. Solving the routing problem (VRP/TSP) constructively. Uses Multi-Head Attention to select the next node in the tour. |
| **TemporalAM** | `models/temporal_am.py` | Transformer | A variant of AM designed to handle time-dependent features directly within the attention mechanism. |
| **MetaRNN** | `models/meta_rnn.py` | RNN/LSTM | Meta-learning component that encodes "tasks" or "distributions" to help the AM generalize across different environments. |
| **ContextEmbedder** | `models/context_embedder.py` | Embedding Layer | Abstract base class and implementations (WC/VRPP) for problem-specific context embeddings. |
| **DeepDecoderAM** | `models/deep_decoder_am.py` | Transformer | Deep decoder variant of the Attention Model. |
| **HyperNet** | `models/hypernet.py` | Hypernetwork | Network that generates weights for other networks, potentially for meta-learning. |
| **PointerNetwork** | `models/pointer_network.py` | RNN + Attention | Traditional Pointer Network architecture implementation. |
| **ModelFactory** | `models/model_factory.py` | Factory Pattern | Central factory for instantiating neural models and components. |
| **CriticNetwork** | `models/critic_network.py` | MLP (Encoder) | Estimates the State-Value function $V(s)$ to compute the baseline for REINFORCE/PPO, reducing gradient variance. |

### 3. Classical Policies (logic/src/policies/)
Traditional OR and Heuristic solvers used as baselines or fallback mechanisms.

| Policy Name | File | Type | Description |
| :--- | :--- | :--- | :--- |
| **LookAhead** | `policies/look_ahead.py` | Rolling Horizon | A planning agent that optimizes routes over a future window ($N$ days). Can deliver simple heuristics or sub-trigger exact solvers. |
| **ALNS** | `policies/adaptive_large_neighborhood_search.py` | Metaheuristic | Implements Adaptive Large Neighborhood Search. Iteratively improves solutions by applying destroy and repair operators. |
| **BCP** | `policies/branch_cut_and_price.py` | Exact / Hybrid | A dispatcher for Branch-Cut-and-Price algorithms. Supports OR-Tools, VRPy, and Gurobi engines for solving prize-collecting VRPs. |
| **HGS** | `policies/hybrid_genetic_search.py` | Genetic Algorithm | Hybrid Genetic Search that combines evolutionary operators with local search and a Split algorithm for decoding giant tours. |
| **LastMinute** | `policies/last_minute.py` | Reactive Heuristic | Trigger-based policy that initiates collection only when bin levels exceed a pre-defined threshold. |
| **MultiVehicle** | `policies/multi_vehicle.py` | OR Solver | External solver interface (PyVRP/OR-Tools) optimized for multi-vehicle routing with capacity constraints. |
| **NeuralAgent** | `policies/neural_agent.py` | Agent Wrapper | Interfaces between the simulator and Neural Model. Handles stepping, HRL gating, and result construction. |
| **PolicyVRPP** | `policies/policy_vrpp.py` | Heuristic/Policy | Specific policy logic for VRPP scenarios. |
| **VRPPOptimizer** | `policies/vrpp_optimizer.py` | Exact/Hybrid | Unified interface for Gurobi and Hexaly optimizers for solving VRPP instances. |
| **Regular** | `policies/regular.py` | Periodic | A baseline policy that visits every bin in the area on a fixed schedule (e.g., every $N$ days). |
| **SingleVehicle** | `policies/single_vehicle.py` | TSP Heuristic | Constructive heuristic for single-vehicle scenarios. Uses `fast_tsp` for sequencing and inserts depot trips as needed for capacity. |

### 4. Problem Environments (logic/src/problems/)
Defines the "Physics" of the simulation: State transitions, Constraints, and Rewards.

| Environment | Directory | Description |
| :--- | :--- | :--- |
| **VRPP** | `problems/vrpp/` | **Vehicle Routing Problem with Profits**. Nodes have rewards (profit) and demand. Vehicles choose which nodes to visit to maximize Profit - Cost. |
| **WCVRP** | `problems/wcvrp/` | **Waste Collection VRP**. A variant of CVRP where bin levels accumulate over time. |
| **CWCVRP** | `problems/wcvrp/` | **Capacitated Waste Collection VRP**. Combines WCVRP physics with VRPP rewards. The standard environment for the WSmart+ project. |

### 5. Neural Components (logic/src/models/modules/)
These low-level building blocks are used to construct the higher-level models.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **MultiHeadAttention** | `models/modules/multi_head_attention.py` | Standard multi-head attention mechanism used for context encoding and node selection. |
| **GraphConvolution** | `models/modules/graph_convolution.py` | Basic spatial encoder layer that aggregates features from neighboring nodes. |
| **DistanceAwareGC** | `models/modules/distance_graph_convolution.py` | Spatial encoder that scales influence by physical distance (Inverse, Exponential, or Learned). |
| **GatedGraphConv** | `models/modules/gated_graph_convolution.py` | Advanced RNN-style graph layer that updates both node and edge features through gating. |
| **EfficientGraphConv** | `models/modules/efficient_graph_convolution.py` | Lightweight multi-head convolution with multiple aggregators (mean, max, symnorm, etc.). |
| **FeedForward** | `models/modules/feed_forward.py` | Standard multi-layer perceptron (MLP) block for feature transformation. |
| **Normalization** | `models/modules/normalization.py` | Wrapper for Batch, Layer, and Instance normalization layers. |
| **ActivationFunction**| `models/modules/activation_function.py`| Wrapper for common activation functions (ReLU, Sigmoid, etc.). |
| **SkipConnection** | `models/modules/skip_connection.py` | Manages residual and dense connections to prevent vanishing gradients. |
| **Connections** | `models/modules/connections.py` | Factory | Factory module for creating various connection types (residual, hyper). |
| **HyperConnection** | `models/modules/hyper_connection.py` | Hyper-Network | Static and Dynamic Hyper-Connections that mix information streams in width and depth dimensions. |
| **NormActivation** | `models/modules/normalized_activation_function.py` | Activation | Combined Normalization and Activation layer. |

### 6. Neural Sub-networks (logic/src/models/subnets/)
Medium-level sub-networks that combine multiple modules into functional blocks (Encoders, Decoders).

#### Encoders
| Component Name | File | Description |
| :--- | :--- | :--- |
| **GATEncoder** | `models/subnets/gat_encoder.py` | Multi-head Graph Attention encoder for spatial relationship modeling. |
| **GACEncoder** | `models/subnets/gac_encoder.py` | Graph Attention Convolution encoder that incorporates edge features. |
| **TGCEncoder** | `models/subnets/tgc_encoder.py` | Transformer-style Graph Convolution encoder. |
| **GGACEncoder** | `models/subnets/ggac_encoder.py` | Gated Graph Attention Convolution encoder with edge-node interaction. |
| **GCNEncoder** | `models/subnets/gcn_encoder.py` | Standard Graph Convolutional Network encoder. |
| **PointerEncoder** | `models/subnets/ptr_encoder.py` | Simple encoder for Pointer Network architectures. |
| **MLPEncoder** | `models/subnets/mlp_encoder.py` | Layer-wise MLP encoder independent of graph structure. |

#### Decoders
| Component Name | File | Description |
| :--- | :--- | :--- |
| **AttentionDecoder** | `models/subnets/attention_decoder.py` | Standard attention-based decoder component. |
| **GATDecoder** | `models/subnets/gat_decoder.py` | Decodes graph embeddings into action log-likelihoods using attention. |
| **PointerDecoder** | `models/subnets/ptr_decoder.py` | Implements the pointing mechanism for constructive routing. |

#### Predictors
| Component Name | File | Description |
| :--- | :--- | :--- |
| **GatedRecurrentFillPredictor** | `models/subnets/grf_predictor.py` | Predicts future bin fill levels using gated recurrent logic. |

### 7. Simulator Hub (logic/src/pipeline/simulator/)
The "Physics Engine" and environment management layer for the WSmart-Route simulation.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **Simulation** | `pipeline/simulator/simulation.py` | Main orchestrator for running large-scale simulation experiments (sequential or parallel). |
| **Day Runner** | `pipeline/simulator/day.py` | Logic for executing a single simulation day: state transitions, policy execution, and result logging. |
| **Bins** | `pipeline/simulator/bins.py` | Manages the state and stochastic/empirical fill logic for the bin population. |
| **Checkpointing** | `pipeline/simulator/checkpoints.py` | Handles saving and resuming simulation states to prevent data loss. |
| **Loader** | `pipeline/simulator/loader.py` | Utilities for loading area-specific coordinates, waste data, and parameters. |
| **Network** | `pipeline/simulator/network.py` | Manages distance matrices, shortest path computations (OSM/Google Maps), and graph topology. |
| **Processor** | `pipeline/simulator/processor.py` | Data normalization and transformation logic for preparing inputs for neural models. |
| **Actions** | `pipeline/simulator/actions.py` | Command Pattern implementation for simulation steps (Fill, Policy Execution, Collect, Log). |
| **States** | `pipeline/simulator/states.py` | State Pattern implementation to manage the simulation lifecycle (Initializing, Running, Finishing). |

### 8. Reinforcement Learning (logic/src/pipeline/reinforcement_learning/)
Algorithms and training loops for optimizing neural agents via policy gradients.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **Reinforce** | `pipeline/reinforcement_learning/core/reinforce.py` | Implements REINFORCE and PPO algorithms. Calculates rewards, baselines, and gradients for both Manager and Worker. |
| **Epoch Manager** | `pipeline/reinforcement_learning/core/epoch.py` | Orchestrates the inner training loop. Manages state-to-model-to-optimizer interactions during a single epoch. |
| **HyperParam Optimizer** | `pipeline/reinforcement_learning/hyperparameter_optimization/hpo.py` | Implements auto-tuning algorithms like Random Search, Grid Search, and DEHB to find optimal hyperparameters. |
| **Post Processor** | `pipeline/reinforcement_learning/post_processing.py` | Optional optimization layer that refines model outputs for efficiency (kg/km) post-training. |
| **ReinforceBaselines** | `pipeline/reinforcement_learning/core/reinforce_baselines.py` | Implements baselines for policy gradients: Exponential, Critic, Rollout, and POMO. |
| **Trainers** | `pipeline/reinforcement_learning/core/trainers.py` | Template Method pattern for training loops. Includes Standard, Time-Based, HRL, and Meta-Learning trainers. |
| **DEHB** | `pipeline/reinforcement_learning/hyperparameter_optimization/dehb.py` | Differential Evolution Hyperband interaction for efficient hyperparameter tuning. |
| **ContextualBandits** | `pipeline/reinforcement_learning/meta/contextual_bandits.py` | Meta-learning strategy using Contextual Bandits (UCB, Thompson Sampling) to dynamically adjust objective weights. |

### 9. Utility Layer (logic/src/utils/)
A collection of helper modules for CLI parsing, spatial computations, I/O, and visualization.

| Category | Files | Description |
| :--- | :--- | :--- |
| **CLI & Config** | `utils/arg_parser.py`, `utils/definitions.py` | Handles project-wide CLI arguments and global constant definitions. |
| **I/O & Logging** | `utils/io_utils.py`, `utils/log_utils.py`, `utils/crypto_utils.py` | Manages file operations, WandB/terminal logging, and data encryption. |
| **Logic Helpers** | `utils/graph_utils.py`, `utils/beam_search.py`, `utils/boolmask.py`, `utils/lexsort.py` | Core mathematical, algorithmic, and sorting helpers. |
| **System & Setup** | `utils/setup_utils.py`, `utils/functions.py`, `utils/monkey_patch.py` | Initialization factories for models/envs, tensor operations, and runtime patching. |
| **Visualization & Debug** | `utils/plot_utils.py`, `utils/visualize_utils.py`, `utils/debug_utils.py` | Utilities for plotting, visual overlays, and debugging. |

### 10. Test Suite (logic/test/)
Comprehensive testing framework for validating the intelligence and physics of the system.

| Category | Files | Description |
| :--- | :--- | :--- |
| **Test Runner** | `test/test_suite.py` | A wrapper for `pytest` that supports modular execution and coverage reporting. |
| **Neural Integrity** | `test/test_models.py`, `test/test_modules.py`, `test/test_subnets.py` | Ensures neural architectures are correctly built and weights are trainable. |
| **Logic & Physics** | `test/test_policies.py`, `test/test_problems.py`, `test/test_simulator.py` | Behavioral tests for agents and classical heuristics in various environment scenarios. |
| **Pipeline Validation**| `test/test_train.py`, `test/test_mrl_train.py`, `test/test_hp_optim.py` | End-to-end verification of training, meta-learning, and HPO workflows. |

## 11. GUI Submodule Architecture

This document documents the "Active" components of the GUI in `gui/src`. Ideally, business logic should be offloaded to `logic/`, but these components handle the translation between User Intent and System Action.

### 1. Background Workers (gui/src/helpers/)
Non-blocking agents that run in separate threads (`QThread`) to keep the UI responsive.

| Worker Name | File | Signals | Responsibilities |
| :--- | :--- | :--- | :--- |
| **ChartWorker** | `helpers/chart_worker.py` | `data_ready`, `finished` | **Data Visualizer**. Parses real-time simulation logs (JSON/CSV) and emits coordinates for live plotting of Profit, Cost, and Waste metrics. |
| **DataLoaderWorker** | `helpers/data_loader_worker.py` | `data_loaded`, `error` | **Async I/O**. Loads massive datasets (Distance Matrices, Graph Geometries) in the background interaction during startup. |
| **FileTailerWorker** | `helpers/file_tailer_worker.py` | `new_lines`, `file_changed` | **Log Streamer**. Implements a `tail -f` equivalent to stream stdout/stderr from external processes into the Main Window Console. |

### 2. Windows (gui/src/windows/)
Top-level containers that manage the application lifecycle.

| Window Name | File | Description |
| :--- | :--- | :--- |
| **MainWindow** | `windows/main_window.py` | **App Root**. The primary application container. Manages the main menu, navigation sidebar, and the central `QTabWidget` hosting all functional tabs. |
| **TSResultsWindow** | `windows/ts_results_window.py` | **Simulation Dashboard**. A specialized window that pops up after a Simulation Test. Displays Heatmaps, Route Maps (Folium), and Aggregate Statistics for the run. |

### 3. Functional Tabs (gui/src/tabs/)
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

### 4. Core Components (gui/src/core/)
Foundational logic that orchestrates communication and state within the GUI.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **UIMediator** | `core/mediator.py` | **Mediator Pattern**. Central hub handling communication between the Main Window and Tabs. Updates the command preview dynamically based on active tab parameters. |

### 5. Reusable Components (gui/src/components/)
Custom UI widgets used across multiple windows.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **ClickableHeader** | `components/clickable_header.py` | A custom `QWidget` that emits a signal/callback when clicked, used for collapsible headers in the UI. |

### 6. Utility Layer (gui/src/utils/)
Centralized constants and mapping logic for the GUI.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **AppDefinitions** | `utils/app_definitions.py` | **UI Registry**. Maps user-friendly names in the GUI to internal keywords for models, encoders, and optimization algorithms. |

### 7. Styles (gui/src/styles/)
Visual design system.

| Component Name | File | Description |
| :--- | :--- | :--- |
| **GlobalStyles** | `styles/globals.py` | Defines the application-wide color palette, fonts, and shared QSS stylesheets. |

### 8. Test Suite (gui/test/)
Automated UI testing to ensure responsive and accurate user interaction.

| Category | Files | Description |
| :--- | :--- | :--- |
| **Communication** | `test/test_mediator.py` | Validates the Mediator pattern and signal/slot orchestration between components. |
| **UI Components** | `test/test_components.py`, `test/test_helpers.py`, `test/test_ts_results_window.py` | Verification of background workers and custom interactive widgets. |
| **Functional Tabs** | `test/test_tabs_*.py` | Unit tests for individual tabs, ensuring parameter passing and command previews are correct. |
