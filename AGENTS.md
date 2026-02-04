# AGENTS.md - Instructions for Coding Assistant LLMs

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Gurobi](https://img.shields.io/badge/Gurobi-11.0-ED1C24?logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/MyPy-checked-2f4f4f.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-60%25-green.svg)](https://coverage.readthedocs.io/)
[![CI](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml/badge.svg)](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml)

> **Version**: 3.0
> **Last Updated**: January 2026
> **Purpose**: Comprehensive guide for AI coding assistants working on WSmart+ Route

This document serves as the authoritative reference for AI assistants (Claude, GPT, Copilot, etc.) working on the WSmart+ Route codebase. It provides context about the project mission, architecture, coding standards, and operational guidelines.

---

## Table of Contents

1. [Project Overview & Mission](#1-project-overview--mission)
2. [Technical Stack & Environmental Governance](#2-technical-stack--environmental-governance)
3. [Core Architectural Boundaries](#3-core-architectural-boundaries)
4. [Key CLI Entry Points](#4-key-cli-entry-points-operational-playbook)
5. [External Access and Browser Usage Rules](#5-external-access-and-browser-usage-rules)
6. [Domain-Specific Coding Standards](#6-domain-specific-coding-standards)
7. [AI Review & Severity Protocol](#7-ai-review--severity-protocol)
8. [Known Constraints & "No-Go" Areas](#8-known-constraints--no-go-areas)
9. [Usage Note](#9-usage-note)
10. [Logic Submodule Architecture](#10-logic-submodule-architecture)
11. [GUI Submodule Architecture](#11-gui-submodule-architecture)
12. [Data Formats & Schemas](#12-data-formats--schemas)
13. [Testing Guidelines for AI](#13-testing-guidelines-for-ai)
14. [Common Patterns & Anti-Patterns](#14-common-patterns--anti-patterns)

---

## 1. Project Overview & Mission

**WSmart+ Route** is a high-performance framework for solving complex Combinatorial Optimization (CO) problems, specifically:

- **Vehicle Routing Problem with Profits (VRPP)**: Select profitable subset of nodes to visit
- **Capacitated Waste Collection VRP (CWC VRP)**: Dynamic waste collection with capacity constraints
- **Stochastic Demand WCVRP (SDWCVRP)**: Waste generation with uncertainty modeling

### 1.1 Mission Statement

The project bridges **Deep Reinforcement Learning (DRL)** with **Operations Research (OR)**, providing:

1. **Benchmarking Environment**: Compare neural models against classical solvers
2. **Deployment Platform**: Production-ready routing optimization
3. **Research Testbed**: Experiment with novel architectures and algorithms
4. **Simulation Engine**: Test policies on realistic multi-day scenarios

### 1.2 Key Capabilities

| Capability                 | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| **Neural Routing**         | Attention-based models (AM, TAM, DDAM) for constructive routing |
| **Classical Optimization** | Gurobi (exact), ALNS, HGS metaheuristics                        |
| **Hierarchical RL**        | Manager-Worker architecture for temporal decision-making        |
| **Meta-Learning**          | Cross-distribution generalization via MetaRNN                   |
| **Real-World Simulation**  | Multi-day scenarios with stochastic bin fill rates              |
| **Interactive GUI**        | PySide6 desktop application for visualization                   |

---

## 2. Technical Stack & Environmental Governance

### 2.1 Runtime Environment

| Component               | Specification | Notes                                   |
| ----------------------- | ------------- | --------------------------------------- |
| **Python**              | 3.9+          | Managed strictly via `uv`               |
| **Package Manager**     | `uv`          | Use `uv sync` for dependency resolution |
| **Virtual Environment** | `.venv/`      | Always activate before development      |

### 2.2 Primary Frameworks

| Framework             | Version | Purpose                            |
| --------------------- | ------- | ---------------------------------- |
| **PyTorch**           | 2.2.2   | Deep learning, CUDA-optimized      |
| **PyTorch Geometric** | 2.3.1   | Graph neural networks              |
| **Gurobi Optimizer**  | 11.0.3  | Exact optimization solver          |
| **Hexaly**            | 14.0+   | High-performance local search      |
| **OR-Tools**          | 9.4     | Google's optimization toolkit      |
| **PyVRP**             | 0.9.1+  | VRP solver library                 |
| **ALNS**              | 7.0+    | Adaptive Large Neighborhood Search |
| **PySide6**           | 6.9.0   | Qt for Python (GUI)                |

### 2.3 Quality Control Tools

| Tool       | Purpose                       | Command               |
| ---------- | ----------------------------- | --------------------- |
| **ruff**   | Linter (Mandatory compliance) | `uv run ruff check .` |
| **black**  | Formatter                     | `uv run black .`      |
| **mypy**   | Type checking                 | `uv run mypy .`       |
| **pytest** | Testing                       | `uv run pytest`       |

### 2.4 Hardware Optimization

- **Target GPUs**: NVIDIA RTX 3090 Ti, RTX 4080 (laptop)
- **CUDA Version**: 11.8+ recommended
- **Memory**: Batch sizes tuned for 12GB+ VRAM

---

## 3. Core Architectural Boundaries

Maintain **strict separation of concerns** across these primary modules:

### 3.1 Logic Layer (`logic/src/`)

The computational engine, completely independent of UI concerns.

```
logic/src/
├── cli/              # Command-line interface & argument parsing
├── models/           # Neural network architectures
│   ├── modules/      # Atomic components (attention, normalization)
│   ├── subnets/      # Encoders, decoders, predictors
│   └── policies/     # Classical policies (HGS, local_search, split)
├── policies/         # Classical & heuristic algorithms
├── envs/             # Problem environments (replacing tasks/)
├── pipeline/         # Training, evaluation, simulation orchestration
│   ├── rl/           # Lightning-based RL pipeline
│   ├── features/     # Feature-specific implementations (train, eval, test)
│   └── simulations/  # Simulator engine
├── data/             # Data generation utilities
└── utils/            # Helper functions & utilities
```

### 3.2 GUI Layer (`gui/src/`)

Desktop application for visualization and interaction.

```
gui/src/
├── windows/          # Top-level application windows
├── tabs/             # Functional UI modules
│   ├── reinforcement_learning/  # Training configuration
│   ├── evaluation/              # Model evaluation
│   ├── test_simulator/          # Simulation testing
│   ├── generate_data/           # Data generation
│   ├── analysis/                # Visualization & analysis
│   └── file_system/             # File management
├── helpers/          # Background workers (QThread)
├── core/             # Mediator pattern & signals
├── components/       # Reusable UI widgets
├── styles/           # Visual design system
└── utils/            # GUI-specific utilities
```

### 3.3 Critical Boundaries

| Boundary      | Rule                                                       |
| ------------- | ---------------------------------------------------------- |
| Logic → GUI   | Logic must **never** import from GUI                       |
| GUI → Logic   | GUI imports Logic via defined interfaces                   |
| State Files   | `state_*.py` files are **critical**; test before modifying |
| Thread Safety | Heavy computation must use `QThread`, never main Qt thread |

---

## 4. Key CLI Entry Points (Operational Playbook)

Always reference these commands when proposing code changes or workflows:

### 4.1 Environment Management

| Action             | Command                                                      |
| ------------------ | ------------------------------------------------------------ |
| Install uv         | `curl -LsSf https://astral.sh/uv/install.sh \| sh`           |
| Sync Environment   | `uv sync`                                                    |
| Activate venv      | `source .venv/bin/activate`                                  |
| Check Installation | `python -c "import torch; print(torch.cuda.is_available())"` |

### 4.2 Data Operations

| Action                   | Command                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| Generate Training Data   | `python main.py gen_data virtual --problem vrpp --graph_sizes 50`            |
| Generate Validation Data | `python main.py gen_data val --problem vrpp --graph_sizes 20 50 --seed 1234` |
| Generate Test Data       | `python main.py gen_data test --problem all --graph_sizes 20 50`             |

### 4.3 Training & Evaluation

| Action                      | Command                                                                                      |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| Train Model                 | `python main.py train_lightning model=am env.name=vrpp env.num_loc=50`                       |
| Meta-RL Training            | `python main.py train_lightning experiment=meta_rl model=am env.name=vrpp train.n_epochs=50` |
| Hyperparameter Optimization | `python main.py train_lightning experiment=hpo env.name=wcvrp`                               |
| Evaluate Model              | `python main.py eval data/vrpp/test.pkl --model ./weights/best.pt`                           |

### 4.4 Simulation & Testing

| Action             | Command                                                            |
| ------------------ | ------------------------------------------------------------------ |
| Run Simulation     | `python main.py test_sim --policies regular gurobi alns --days 31` |
| Run Test Suite     | `python main.py test_suite`                                        |
| Run Specific Tests | `python main.py test_suite --module test_models`                   |
| Launch GUI         | `python main.py gui`                                               |
| Launch TUI         | `python main.py tui`                                               |

### 4.5 Code Quality

| Action      | Command                            |
| ----------- | ---------------------------------- |
| Lint Code   | `uv run ruff check .`              |
| Format Code | `uv run ruff format`               |
| Type Check  | `uv run mypy .`                    |
| Run Full CI | `just check` (if justfile present) |

---

## 5. External Access and Browser Usage Rules

The agent is authorized to use external tools to assist in development:

### 5.1 Web Search Authorization

| Purpose                  | Authorization Level                                   |
| ------------------------ | ----------------------------------------------------- |
| **Documentation Lookup** | ✅ Authorized - Gurobi 11+, PySide6, PyTorch 2.2+     |
| **API Verification**     | ✅ Authorized - Verify methods are not deprecated     |
| **Bug Investigation**    | ✅ Authorized - GitHub issues, Stack Overflow         |
| **CUDA/Driver Issues**   | ✅ Authorized - NVIDIA driver conflicts, Ubuntu fixes |

### 5.2 Knowledge Cutoff Management

**Directive**: Cross-reference internal training data with web search for technologies updated after January 2024:

- Gurobi performance tunings (v11+)
- PyTorch 2.x features (compile, inductor)
- PySide6 recent API changes
- CUDA 12.x compatibility

### 5.3 Restricted Actions

| Action                    | Status        | Reason                |
| ------------------------- | ------------- | --------------------- |
| Modify production configs | ❌ Restricted | Requires human review |
| Delete data files         | ❌ Restricted | Potential data loss   |
| Push to main branch       | ❌ Restricted | Requires PR review    |
| Modify encryption keys    | ❌ Restricted | Security-critical     |

---

## 6. Domain-Specific Coding Standards

### 6.1 Mathematical & DRL Integrity

#### Invalid Move Prevention

Decoders **must** implement masking via `logic/src/utils/functions/boolmask.py` before sampling nodes:

```python
# CORRECT: Apply mask before softmax
logits = self.compute_logits(state)
logits = logits.masked_fill(mask, float('-inf'))
probs = F.softmax(logits, dim=-1)
action = torch.multinomial(probs, 1)

# WRONG: Sampling without masking can produce invalid routes
```

#### Activation Scaling

Prefer custom modules in `logic/src/models/modules/normalization.py` over generic `nn.LayerNorm`:

```python
# PREFERRED
from logic.src.models.modules.normalization import Normalization
norm = Normalization(dim, normalization='instance')

# AVOID (unless specifically required)
norm = nn.LayerNorm(dim)
```

#### Configuration Sanitization

When passing Hydra/OmegaConf configs to Lightning modules (saved as hyperparameters):

1.  **Deep Sanitize**: Convert `DictConfig` and `ListConfig` to primitive `dict` and `list`.
2.  **Order Matters**: Sanitize _before_ injecting complex objects like `env` or `policy`.
3.  **Removal**: Remove non-serializable objects (like `env` instance) from `hparams` manually if needed in `__init__`.

```python
# CORRECT: Sanitize first, then inject
common_kwargs = deep_sanitize(cfg.rl)
common_kwargs["env"] = env  # injected after
model = MyModule(**common_kwargs)

# WRONG: Passing DictConfig directly causes YAML errors
model = MyModule(**cfg.rl)
```

#### State Transitions

**Never modify** `state_*.py` files without ensuring `logic/test/test_problems.py` passes:

```bash
# Before any state file modification
python main.py test_suite --module test_problems
```

### 6.2 Performance & Hardware

#### GPU Offloading

Ensure tensors are explicitly moved to device using `setup_utils.py`:

```python
from logic.src.utils.configs.setup_utils import get_device

device = get_device(cuda_enabled=True)
tensor = tensor.to(device)
model = model.to(device)
```

#### Batch Size Guidelines

| GPU VRAM | Recommended Batch Size |
| -------- | ---------------------- |
| 8 GB     | 64-128                 |
| 12 GB    | 256                    |
| 24 GB    | 512-1024               |

#### GUI Threading

Heavy computations **must** inherit from `QThread`:

```python
# CORRECT: Background worker
class TrainingWorker(QThread):
    progress = Signal(int)

    def run(self):
        # Heavy computation here
        pass

# WRONG: Blocking main thread
def on_button_click(self):
    train_model()  # Freezes GUI
```

### 6.3 Code Style

#### Imports Organization

```python
# Standard library
import os
from typing import Dict, List, Optional

# Third-party
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

# Local imports
from logic.src.models.modules import MultiHeadAttention
from logic.src.utils import get_device
```

#### Type Hints

Always use type hints for public functions:

```python
def compute_route_cost(
    route: List[int],
    distance_matrix: torch.Tensor,
    capacity: float = 100.0
) -> float:
    """Compute total cost of a route."""
    ...
```

---

## 7. AI Review & Severity Protocol

Categorize feedback and edits using these severity levels:

### 7.1 CRITICAL (Must Fix Immediately)

- Breaking `state_*.py` transition logic
- Exposing credentials or API keys
- Cryptographic flaws in `crypto_utils.py`
- Data corruption in checkpoint files
- Security vulnerabilities (injection, XSS)

### 7.2 HIGH (Fix Before Merge)

- CUDA memory leaks
- Incorrect `skip_connection.py` usage
- `pyproject.toml` version mismatches
- Breaking changes to public APIs
- Test failures in CI pipeline

### 7.3 MEDIUM (Fix Soon)

- Suboptimal Pandas operations in `pandas_model.py`
- Deviations from ruff formatting
- Missing type hints on public functions
- Inefficient tensor operations
- Documentation inconsistencies

### 7.4 LOW (Nice to Have)

- Documentation typos
- Redundant imports
- UI padding/margin adjustments in `globals.py`
- Variable naming improvements
- Comment clarifications

---

## 8. Known Constraints & "No-Go" Areas

### 8.1 Legacy Preservation

| Pattern              | Rule                                  |
| -------------------- | ------------------------------------- |
| `*_copy.py` files    | Never edit - these are backups        |
| `legacy/` folders    | Read-only reference code              |
| Deprecated functions | Mark with `@deprecated`, don't delete |

### 8.2 Slurm Sensitivity

Cluster scripts (`scripts/slurm.sh`) use specific path mappings:

```bash
# Before modifying:
# 1. Verify SLURM_JOB_ID handling
# 2. Check module load commands
# 3. Validate output directory paths
```

### 8.3 Linux GUI Stability

When debugging PySide6 GUI on Linux, include these flags:

```bash
# Recommended debug flags
export QT_QPA_PLATFORM=xcb
python main.py gui --use-angle=vulkan --disable-gpu-sandbox
```

### 8.4 Protected Files

| File/Directory          | Protection Level                |
| ----------------------- | ------------------------------- |
| `assets/keys/`          | Never commit to git             |
| `assets/model_weights/` | Large files - use Git LFS       |
| `.env` files            | Never commit - contains secrets |
| `*.lic` files           | License files - never commit    |

---

## 9. Usage Note

### 9.1 Session Initialization

When starting a new terminal session:

```bash
# 1. Kill any previous instances
pkill -f "python main.py"

# 2. Activate environment
source .venv/bin/activate

# 3. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 4. Update dependencies if needed
uv sync
```

### 9.2 Reference Files

| File                 | Purpose                          |
| -------------------- | -------------------------------- |
| `project_map.txt`    | Full project structure map       |
| `CLAUDE.md`          | This file (symlink to AGENTS.md) |
| `ARCHITECTURE.md`    | System design documentation      |
| `TROUBLESHOOTING.md` | Common issues and fixes          |

---

## 10. Logic Submodule Architecture

This section maintains a registry of intelligent agents, orchestration components, and environment physics within `logic/src/`.

### 10.1 Pipeline Orchestrators (`logic/src/pipeline/`)

| Agent Name    | File                         | Responsibilities                                                                                         |
| ------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Trainer**   | `pipeline/features/train.py` | Central entry for `train`, `mrl_train`, `hp_optim`. Manages device selection, data loading, epoch loops. |
| **Evaluator** | `pipeline/features/eval.py`  | Model evaluation on test datasets. Supports `greedy`, `sampling`, `beam_search` decoding.                |
| **Tester**    | `pipeline/features/test.py`  | Simulation testing across multiple seeds and policies. Parallel execution with checkpointing.            |

### 10.2 Neural Models (`logic/src/models/`)

| Model                    | File                  | Architecture       | Function                                                          |
| ------------------------ | --------------------- | ------------------ | ----------------------------------------------------------------- |
| **AttentionModel (AM)**  | `attention_model.py`  | Transformer        | **Worker Agent**. Constructive routing with Multi-Head Attention. |
| **DeepDecoderAM (DDAM)** | `deep_decoder_am.py`  | Deep Transformer   | Deep decoder variant for complex problems.                        |
| **TemporalAM (TAM)**     | `temporal_am.py`      | Transformer        | Time-dependent attention for multi-day scenarios.                 |
| **GATLSTManager**        | `gat_lstm_manager.py` | GAT + LSTM         | **High-Level Agent**. Temporal gating for HRL.                    |
| **PointerNetwork**       | `pointer_network.py`  | RNN + Attention    | Classic pointer mechanism implementation.                         |
| **MetaRNN**              | `meta_rnn.py`         | RNN/LSTM           | Meta-learning for distribution generalization.                    |
| **ContextEmbedder**      | `context_embedder.py` | Embedding          | Problem-specific context embeddings (VRPP, WC).                   |
| **CriticNetwork**        | `critic_network.py`   | MLP                | State-Value $V(s)$ for REINFORCE/PPO baselines.                   |
| **HyperNet**             | `hypernet.py`         | Hypernetwork       | Weight generation for meta-learning.                              |
| **MOEModel**             | `moe_model.py`        | Mixture of Experts | Multi-expert routing specialization.                              |
| **ModelFactory**         | `model_factory.py`    | Factory Pattern    | Centralized model instantiation.                                  |

### 10.3 Neural Modules (`logic/src/models/modules/`)

| Module                 | File                             | Description                                                   |
| ---------------------- | -------------------------------- | ------------------------------------------------------------- |
| **MultiHeadAttention** | `multi_head_attention.py`        | Standard MHA for context encoding and node selection.         |
| **GraphConvolution**   | `graph_convolution.py`           | Basic GCN layer for neighbor aggregation.                     |
| **DistanceAwareGC**    | `distance_graph_convolution.py`  | Distance-scaled influence (Inverse, Exponential, Learned).    |
| **GatedGraphConv**     | `gated_graph_convolution.py`     | RNN-style graph layer with gating.                            |
| **EfficientGraphConv** | `efficient_graph_convolution.py` | Lightweight multi-head with aggregators (mean, max, symnorm). |
| **FeedForward**        | `feed_forward.py`                | Standard MLP block.                                           |
| **Normalization**      | `normalization.py`               | Batch, Layer, Instance, Group normalization wrapper.          |
| **ActivationFunction** | `activation_function.py`         | 21+ activation functions (ReLU, GELU, Mish, etc.).            |
| **SkipConnection**     | `skip_connection.py`             | Residual and dense connections.                               |
| **HyperConnection**    | `hyper_connection.py`            | Static/Dynamic hyper-connections for depth mixing.            |
| **MOE**                | `moe.py`                         | Mixture of Experts gating mechanism.                          |
| **MOEFeedForward**     | `moe_feed_forward.py`            | MoE-enhanced feed-forward layers.                             |

### 10.4 Neural Sub-networks (`logic/src/models/subnets/`)

#### Encoders

| Encoder            | File              | Description                                       |
| ------------------ | ----------------- | ------------------------------------------------- |
| **GATEncoder**     | `gat_encoder.py`  | Multi-head Graph Attention for spatial modeling.  |
| **GACEncoder**     | `gac_encoder.py`  | Graph Attention Convolution with edge features.   |
| **TGCEncoder**     | `tgc_encoder.py`  | Transformer-style Graph Convolution.              |
| **GGACEncoder**    | `ggac_encoder.py` | Gated Graph Attention with edge-node interaction. |
| **GCNEncoder**     | `gcn_encoder.py`  | Standard Graph Convolutional Network.             |
| **MLPEncoder**     | `mlp_encoder.py`  | Structure-independent MLP encoding.               |
| **PointerEncoder** | `ptr_encoder.py`  | Encoding for Pointer Networks.                    |
| **MOEEncoder**     | `moe_encoder.py`  | Mixture of Experts encoder routing.               |

#### Decoders

| Decoder              | File                   | Description                                       |
| -------------------- | ---------------------- | ------------------------------------------------- |
| **AttentionDecoder** | `attention_decoder.py` | Standard attention-based autoregressive decoding. |
| **GATDecoder**       | `gat_decoder.py`       | Graph attention for action log-likelihoods.       |
| **PointerDecoder**   | `ptr_decoder.py`       | Pointing mechanism for constructive routing.      |

#### Predictors

| Predictor        | File               | Description                                           |
| ---------------- | ------------------ | ----------------------------------------------------- |
| **GRFPredictor** | `grf_predictor.py` | Gated recurrent predictor for future bin fill levels. |

### 10.5 Classical Policies & Selection (`logic/src/policies/`)

#### Routing Policies

| Policy             | File                                    | Type          | Description                                         |
| ------------------ | --------------------------------------- | ------------- | --------------------------------------------------- |
| **ALNS**           | `adaptive_large_neighborhood_search.py` | Metaheuristic | Destroy-repair operators with adaptive weights.     |
| **BCP**            | `branch_cut_and_price.py`               | Exact         | Branch-Cut-and-Price via Gurobi/OR-Tools/VRPy.      |
| **HGS**            | `hybrid_genetic_search.py`              | Genetic       | Evolutionary operators with local search and Split. |
| **MultiVehicle**   | `multi_vehicle.py`                      | OR Solver     | PyVRP/OR-Tools for multi-vehicle routing.           |
| **SingleVehicle**  | `single_vehicle.py`                     | TSP Heuristic | fast_tsp for single-vehicle sequencing.             |
| **LinKernighan**   | `lin_kernighan.py`                      | Local Search  | Lin-Kernighan TSP heuristic.                        |
| **NeuralAgent**    | `neural_agent.py`                       | Agent Wrapper | Interfaces neural models with simulator.            |
| **PostProcessing** | `post_processing.py`                    | Refinement    | Route improvement heuristics.                       |
| **PolicyFactory**  | `adapters.py`                           | Factory       | Central policy instantiation via `get_adapter()`.   |

#### Selection Strategies (`logic/src/policies/selection/`)

| Strategy         | File                         | Description                                      |
| ---------------- | ---------------------------- | ------------------------------------------------ |
| **Regular**      | `selection_regular.py`       | Fixed-frequency collection (e.g., every 3 days). |
| **LastMinute**   | `selection_last_minute.py`   | Collect when fill level exceeds threshold.       |
| **LookAhead**    | `selection_lookahead.py`     | Collect if overflow predicted within N days.     |
| **Revenue**      | `selection_revenue.py`       | Collect if profit > cost.                        |
| **ServiceLevel** | `selection_service_level.py` | Statistical overflow prediction.                 |

### 10.6 Problem Environments (`logic/src/envs/`)

| Problem     | Directory        | Description                                                                       |
| ----------- | ---------------- | --------------------------------------------------------------------------------- |
| **VRPP**    | `envs/vrpp.py`   | Vehicle Routing Problem with Profits. Nodes have rewards; maximize Profit - Cost. |
| **CVRPP**   | `envs/vrpp.py`   | Capacitated VRPP with vehicle capacity constraints.                               |
| **WCVRP**   | `envs/wcvrp.py`  | Waste Collection VRP. Bin levels accumulate over time.                            |
| **CWCVRP**  | `envs/wcvrp.py`  | Capacitated WCVRP. Standard WSmart+ environment.                                  |
| **SDWCVRP** | `envs/wcvrp.py`  | Stochastic Demand WCVRP. Uncertain waste generation.                              |
| **SCWCVRP** | `envs/swcvrp.py` | Selective Capacitated WCVRP. Collect only when profitable.                        |

**Base Class**: `RL4COEnvBase` in `envs/base.py`

### 10.7 Simulator Engine (`logic/src/pipeline/simulations/`)

| Component       | File             | Description                                                     |
| --------------- | ---------------- | --------------------------------------------------------------- |
| **Simulator**   | `simulator.py`   | Main orchestrator for large-scale experiments.                  |
| **DayRunner**   | `day.py`         | Single-day execution: state transitions, policy, logging.       |
| **Bins**        | `bins.py`        | Bin state and stochastic/empirical fill logic.                  |
| **Network**     | `network.py`     | Distance matrices, shortest paths (OSM/Google Maps).            |
| **Loader**      | `loader.py`      | Area-specific data loading utilities.                           |
| **Processor**   | `processor.py`   | Data normalization for neural model inputs.                     |
| **Actions**     | `actions.py`     | Command Pattern for simulation steps (Fill, Collect, Log).      |
| **States**      | `states.py`      | State Pattern for lifecycle (Initializing, Running, Finishing). |
| **Context**     | `context.py`     | Simulation configuration encapsulation.                         |
| **Checkpoints** | `checkpoints.py` | Save/resume simulation state.                                   |

### 10.8 Reinforcement Learning Pipeline (`logic/src/pipeline/rl/`)

> **Note**: This pipeline uses PyTorch Lightning for all RL algorithms.

#### Core Algorithms (`rl/core/`)

| Component             | File                    | Description                                       |
| --------------------- | ----------------------- | ------------------------------------------------- |
| **RL4COLitModule**    | `base.py`               | Base Lightning module for all RL algorithms.      |
| **REINFORCE**         | `reinforce.py`          | Policy gradient with baselines.                   |
| **PPO**               | `ppo.py`                | Proximal Policy Optimization.                     |
| **SAPO**              | `sapo.py`               | Self-Adaptive Policy Optimization.                |
| **GSPO**              | `gspo.py`               | Gradient-Scaled Proxy Optimization.               |
| **DR-GRPO**           | `dr_grpo.py`            | Divergence-Regularized GRPO.                      |
| **POMO**              | `pomo.py`               | Policy Optimization with Multiple Optima.         |
| **SymNCO**            | `symnco.py`             | Symmetry-aware Neural Combinatorial Optimization. |
| **Imitation**         | `imitation.py`          | Imitation Learning from expert policies.          |
| **AdaptiveImitation** | `adaptive_imitation.py` | IL to RL transition.                              |
| **HRL**               | `hrl.py`                | Hierarchical RL (Manager-Worker).                 |

#### Baselines (`rl/core/baselines.py`)

| Baseline                | Description                           |
| ----------------------- | ------------------------------------- |
| **NoBaseline**          | Zero baseline (high variance).        |
| **ExponentialBaseline** | Moving average of past costs.         |
| **RolloutBaseline**     | Greedy rollout of policy.             |
| **CriticBaseline**      | Learned value network.                |
| **WarmupBaseline**      | Gradual transition between baselines. |
| **POMOBaseline**        | Multi-start best-of-N baseline.       |

#### Meta-Learning (`rl/meta/`)

| Component                  | File                    | Description                                 |
| -------------------------- | ----------------------- | ------------------------------------------- |
| **WeightContextualBandit** | `contextual_bandits.py` | UCB/Thompson Sampling for weight selection. |
| **MORLWeightOptimizer**    | `multi_objective.py`    | Multi-objective Pareto optimization.        |
| **CostWeightManager**      | `td_learning.py`        | TD-based weight learning.                   |
| **RewardWeightOptimizer**  | `weight_optimizer.py`   | Gradient-based weight optimization.         |
| **HyperNetStrategy**       | `hypernet_strategy.py`  | Meta-learning via hypernetworks.            |

#### Hyperparameter Optimization (`rl/hpo/`)

| Component                          | File            | Description                             |
| ---------------------------------- | --------------- | --------------------------------------- |
| **OptunaHPO**                      | `optuna_hpo.py` | Optuna-based HPO with various samplers. |
| **DifferentialEvolutionHyperband** | `dehb.py`       | DEHB for efficient HPO.                 |

#### Features (`rl/features/`)

| Component              | File                 | Description                      |
| ---------------------- | -------------------- | -------------------------------- |
| **prepare_epoch**      | `epoch.py`           | Epoch preparation utilities.     |
| **regenerate_dataset** | `epoch.py`           | Dynamic dataset regeneration.    |
| **TimeBasedTraining**  | `time_training.py`   | Multi-day temporal training.     |
| **PostProcessor**      | `post_processing.py` | Route refinement for efficiency. |

### 10.9 Utility Layer (`logic/src/utils/`)

| Category            | Files                                | Description                                 |
| ------------------- | ------------------------------------ | ------------------------------------------- |
| **CLI & Config**    | `definitions.py`, `config_loader.py` | Constants, mappings, configuration loading. |
| **I/O & Logging**   | `io_utils.py`, logging/              | File operations, WandB/terminal logging.    |
| **Security**        | `crypto_utils.py`                    | Data encryption/decryption.                 |
| **Data Processing** | `data_utils.py`, `task_utils.py`     | Dataset manipulation, problem utilities.    |
| **Setup**           | `setup_utils.py`                     | Model/environment initialization factories. |
| **Debug**           | `debug_utils.py`                     | Debugging and profiling helpers.            |
| **Functions**       | `functions/`                         | Beam search, masking, graph ops, lexsort.   |

### 10.10 Test Suite (`logic/test/`)

| Category        | Files                                                    | Description                            |
| --------------- | -------------------------------------------------------- | -------------------------------------- |
| **Runner**      | `test_suite.py`                                          | pytest wrapper with modular execution. |
| **Neural**      | `test_models.py`, `test_modules.py`, `test_subnets.py`   | Architecture and weight verification.  |
| **Policies**    | `test_policies.py`, `test_policies_aux.py`               | Classical algorithm behavior.          |
| **Problems**    | `test_problems.py`                                       | Environment physics validation.        |
| **Pipeline**    | `test_train.py`, `test_mrl_train.py`, `test_hp_optim.py` | End-to-end workflow tests.             |
| **Integration** | `test_integration.py`, `test_simulator.py`               | Full system integration.               |

---

## 11. GUI Submodule Architecture

Documents the "Active" components handling User Intent → System Action translation.

### 11.1 Background Workers (`gui/src/helpers/`)

| Worker               | File                    | Signals                     | Purpose                                 |
| -------------------- | ----------------------- | --------------------------- | --------------------------------------- |
| **ChartWorker**      | `chart_worker.py`       | `data_ready`, `finished`    | Parse simulation logs, emit plot data.  |
| **DataLoaderWorker** | `data_loader_worker.py` | `data_loaded`, `error`      | Async loading of large datasets.        |
| **FileTailerWorker** | `file_tailer_worker.py` | `new_lines`, `file_changed` | `tail -f` equivalent for log streaming. |

### 11.2 Windows (`gui/src/windows/`)

| Window              | File                   | Description                                                |
| ------------------- | ---------------------- | ---------------------------------------------------------- |
| **MainWindow**      | `main_window.py`       | App root. Menu, sidebar, QTabWidget for functional tabs.   |
| **TSResultsWindow** | `ts_results_window.py` | Simulation dashboard. Heatmaps, Folium routes, statistics. |

### 11.3 Functional Tabs (`gui/src/tabs/`)

| Tab Group                  | Components                         | Purpose                                                 |
| -------------------------- | ---------------------------------- | ------------------------------------------------------- |
| **Reinforcement Learning** | `rl_*.py`, `scripts.py`            | Training config: model, data, optimizer, costs, output. |
| **Evaluation**             | `eval_*.py`                        | Inference: problem, batching, I/O, decoding.            |
| **Test Simulator**         | `ts_*.py`                          | Simulation setup: settings, policies, I/O, advanced.    |
| **Generate Data**          | `gd_*.py`                          | Dataset generation: general, problem, advanced.         |
| **Analysis**               | `*_analysis.py`, `pandas_model.py` | Visualization and data analysis.                        |
| **File System**            | `fs_*.py`                          | Update, delete, cryptography operations.                |
| **Meta-RL**                | `meta_rl_train.py`                 | Meta-learning experiment configuration.                 |
| **HPO**                    | `hyperparam_optim.py`              | Hyperparameter search space and execution.              |

### 11.4 Core Components (`gui/src/core/`)

| Component      | File          | Pattern  | Description                                      |
| -------------- | ------------- | -------- | ------------------------------------------------ |
| **UIMediator** | `mediator.py` | Mediator | Central hub for MainWindow ↔ Tabs communication. |

### 11.5 Reusable Components (`gui/src/components/`)

| Component           | File                  | Description                                     |
| ------------------- | --------------------- | ----------------------------------------------- |
| **ClickableHeader** | `clickable_header.py` | Collapsible header widget with signal emission. |

### 11.6 Utility Layer (`gui/src/utils/`)

| Component          | File                 | Description                                           |
| ------------------ | -------------------- | ----------------------------------------------------- |
| **AppDefinitions** | `app_definitions.py` | UI Registry. Maps display names to internal keywords. |

### 11.7 Styles (`gui/src/styles/`)

| Component        | File         | Description                            |
| ---------------- | ------------ | -------------------------------------- |
| **GlobalStyles** | `globals.py` | Color palette, fonts, QSS stylesheets. |

### 11.8 Test Suite (`gui/test/`)

| Category          | Files                                   | Description                                  |
| ----------------- | --------------------------------------- | -------------------------------------------- |
| **Communication** | `test_mediator.py`                      | Mediator pattern and signal/slot validation. |
| **Components**    | `test_components.py`, `test_helpers.py` | Widget and worker verification.              |
| **Tabs**          | `test_tabs_*.py`                        | Individual tab functionality.                |
| **Windows**       | `test_ts_results_window.py`             | Results window behavior.                     |

---

## 12. Data Formats & Schemas

### 12.1 Problem Instance Format

```python
{
    'loc': torch.Tensor,        # (batch, n_nodes, 2) - coordinates
    'demand': torch.Tensor,     # (batch, n_nodes) - demand/fill level
    'prize': torch.Tensor,      # (batch, n_nodes) - reward for visiting
    'depot': torch.Tensor,      # (batch, 2) - depot coordinates
    'capacity': float,          # Vehicle capacity
    'max_length': float,        # Maximum route length
}
```

### 12.2 Distance Matrix Format

```python
# Shape: (n_nodes, n_nodes), symmetric
# Values: Distance in kilometers
# Diagonal: 0.0
# File formats: .npy, .csv, .pkl
```

### 12.3 Waste Fill Data Format

```csv
timestamp,bin_id,fill_level,waste_type
2024-01-01,BIN001,0.45,plastic
2024-01-01,BIN002,0.72,glass
...
```

### 12.4 Simulation Output Format

```json
{
  "day": 1,
  "policy": "gurobi",
  "kg": 125.5,
  "km": 45.2,
  "cost": 22.6,
  "profit": 50.2,
  "overflows": 0,
  "tour": [0, 5, 12, 8, 0]
}
```

---

## 13. Testing Guidelines for AI

### 13.1 Before Making Changes

```bash
# Always run relevant tests first
python main.py test_suite --module test_models  # If modifying models
python main.py test_suite --module test_problems  # If modifying state/physics
python main.py test_suite --module test_policies  # If modifying policies
```

### 13.2 After Making Changes

```bash
# Run full test suite
python main.py test_suite

# Check code quality
uv run ruff check .

# Run type checks (non-blocking)
uv run mypy . || true
```

### 13.3 Test Markers

| Marker                     | Usage                   |
| -------------------------- | ----------------------- |
| `@pytest.mark.slow`        | Long-running tests      |
| `@pytest.mark.fast`        | Quick unit tests        |
| `@pytest.mark.integration` | Full system tests       |
| `@pytest.mark.train`       | Training pipeline tests |
| `@pytest.mark.gpu`         | Tests requiring CUDA    |

---

## 14. Common Patterns & Anti-Patterns

### 14.1 Preferred Patterns

```python
# ✅ Use factory methods
model = ModelFactory.create_model('am', problem='vrpp', opts=opts)

# ✅ Use device management utilities
device = get_device(cuda_enabled=True)
tensor = tensor.to(device)

# ✅ Use context managers for files
with open(filepath, 'r') as f:
    data = json.load(f)

# ✅ Use type hints
def compute_cost(route: List[int], distances: np.ndarray) -> float:
    ...
```

### 14.2 Anti-Patterns to Avoid

```python
# ❌ Don't hardcode device
tensor = tensor.cuda()  # Fails on CPU-only machines

# ❌ Don't modify state files without tests
# Always verify: python main.py test_suite --module test_problems

# ❌ Don't block Qt main thread
def on_click(self):
    result = expensive_computation()  # Freezes GUI

# ❌ Don't ignore masks in decoders
probs = F.softmax(logits, dim=-1)  # Can select invalid nodes
```

---

## Appendix: Quick Reference

### A.1 Key File Locations

| Purpose             | Location                          |
| ------------------- | --------------------------------- |
| Main entry point    | `main.py`                         |
| CLI parsers         | `logic/src/cli/`                  |
| Neural models       | `logic/src/models/`               |
| Classical policies  | `logic/src/policies/`             |
| Problem definitions | `logic/src/envs/`                 |
| Simulator engine    | `logic/src/pipeline/simulations/` |
| GUI main window     | `gui/src/windows/main_window.py`  |
| Configuration       | `assets/configs/`                 |
| Model weights       | `assets/model_weights/`           |

### A.2 Important Constants

```python
# From logic/src/utils/definitions.py
MAX_WASTE = 1.0
VEHICLE_CAPACITY = 100.0
MAX_LENGTHS = {20: 2, 50: 3, 100: 4, 150: 5, 225: 6, 317: 7}
METRICS = ["overflows", "kg", "ncol", "kg_lost", "km", "kg/km", "cost", "profit"]
```

### A.3 Environment Variables

| Variable               | Purpose                  |
| ---------------------- | ------------------------ |
| `CUDA_VISIBLE_DEVICES` | GPU selection            |
| `GRB_LICENSE_FILE`     | Gurobi license path      |
| `WANDB_API_KEY`        | Weights & Biases logging |
| `QT_QPA_PLATFORM`      | Qt platform backend      |

---

**This guide is the authoritative reference for AI assistants working on WSmart+ Route. Keep it updated as the codebase evolves.**
