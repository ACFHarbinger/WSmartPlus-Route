# WSmart+ Route: The Definitive Developer Encyclopedia & Technical Tutorial

> **Version**: 3.0 (The "Comprehensive" Edition)
> **Target Audience**: Core Engineers, Researchers, ML Scientists

Welcome to the internal documentation of **WSmart+ Route**. This encyclopedia is designed to be the ultimate reference for any developer, researcher, or engineer working on this high-performance combinatorial optimization platform for waste collection routing.

WSmart+ Route is a masterclass in bridging Deep Reinforcement Learning with Operations Research, providing a unified framework where neural models compete with classical solvers on real-world routing problems.

---

## Table of Contents

1.  [The WSmart+ Route Philosophy](#1-the-wsmart-route-philosophy)
2.  [High-Level Architecture & Communication](#2-high-level-architecture--communication)
3.  [Logic Layer: The Intelligence Core (`logic/src/`)](#3-logic-layer-the-intelligence-core-logicsrc)
    -   [3.1 Neural Models](#31-neural-models-logicsrcmodels)
    -   [3.2 Graph Encoders & Decoders](#32-graph-encoders--decoders-logicsrcmodelssubnets)
    -   [3.3 Classical Policies](#33-classical-policies-logicsrcpolicies)
    -   [3.4 Problem Environments](#34-problem-environments-logicsrcproblems)
    -   [3.5 The Simulator Engine](#35-the-simulator-engine-logicsrcpipelinesimulator)
    -   [3.6 Reinforcement Learning Pipeline](#36-reinforcement-learning-pipeline-logicsrcpipelinereinforcement_learning)
4.  [GUI Layer: The Command Center (`gui/src/`)](#4-gui-layer-the-command-center-guisrc)
    -   [4.1 PySide6 Architecture](#41-pyside6-architecture)
    -   [4.2 Background Workers](#42-background-workers)
    -   [4.3 Real-Time Visualization](#43-real-time-visualization)
5.  [Algorithm Deep Dives](#5-algorithm-deep-dives)
    -   [5.1 Attention Mechanism for Routing](#51-attention-mechanism-for-routing)
    -   [5.2 REINFORCE with Baselines](#52-reinforce-with-baselines)
    -   [5.3 Adaptive Large Neighborhood Search (ALNS)](#53-adaptive-large-neighborhood-search-alns)
    -   [5.4 Branch-Cut-and-Price](#54-branch-cut-and-price)
    -   [5.5 Hybrid Genetic Search (HGS)](#55-hybrid-genetic-search-hgs)
    -   [5.6 Hierarchical Reinforcement Learning](#56-hierarchical-reinforcement-learning)
6.  [Development Life Cycle](#6-development-life-cycle)
    -   [6.1 Setup & Environment](#61-setup--environment)
    -   [6.2 Training Your First Model](#62-training-your-first-model)
    -   [6.3 Running Simulations](#63-running-simulations)
    -   [6.4 Using the GUI](#64-using-the-gui)
    -   [6.5 Hyperparameter Optimization](#65-hyperparameter-optimization)
    -   [6.6 Meta-Reinforcement Learning](#66-meta-reinforcement-learning)
7.  [Extending WSmart+ Route](#7-extending-wsmart-route)
    -   [7.1 Adding a New Neural Architecture](#71-adding-a-new-neural-architecture)
    -   [7.2 Adding a New Classical Policy](#72-adding-a-new-classical-policy)
    -   [7.3 Adding a New Problem Variant](#73-adding-a-new-problem-variant)
    -   [7.4 Adding a New Encoder](#74-adding-a-new-encoder)
    -   [7.5 Adding a New Decoder](#75-adding-a-new-decoder)
    -   [7.6 Adding a New RL Algorithm](#76-adding-a-new-rl-algorithm)
8.  [Testing & Quality Assurance](#8-testing--quality-assurance)
    -   [8.1 Test Suite Overview](#81-test-suite-overview)
    -   [8.2 Writing New Tests](#82-writing-new-tests)
    -   [8.3 Coverage Requirements](#83-coverage-requirements)
9.  [Advanced Topics](#9-advanced-topics)
    -   [9.1 Multi-GPU Training](#91-multi-gpu-training)
    -   [9.2 Custom Baselines](#92-custom-baselines)
    -   [9.3 Distance Matrix Computation](#93-distance-matrix-computation)
    -   [9.4 Checkpoint Management](#94-checkpoint-management)
    -   [9.5 Logging and Monitoring](#95-logging-and-monitoring)
10. [Exhaustive Code Reference](#10-exhaustive-code-reference)
11. [Glossary of Terms](#11-glossary-of-terms)
12. [Frequently Asked Questions](#12-frequently-asked-questions)

---

## 1. The WSmart+ Route Philosophy

Waste collection routing is a complex combinatorial optimization problem with real-world constraints and significant economic impact. Traditional approaches rely on either heuristics (fast but suboptimal) or exact solvers (optimal but slow). WSmart+ Route was built on three core pillars to bridge this gap:

### 1.1 Neural Models as Constructive Heuristics

Deep Learning models, particularly attention-based architectures, can learn to construct high-quality solutions in a fraction of the time required by exact methods. By training on millions of problem instances, these models generalize to unseen scenarios and adapt to different problem distributions.

**Key Insight**: The attention mechanism naturally captures the "look-ahead" reasoning that human experts use when solving routing problems—considering not just the immediate next stop, but how that choice affects future decisions.

### 1.2 Benchmarking Ecosystem

To validate neural approaches, we need rigorous comparisons. WSmart+ Route provides:
1.  **Exact Solvers**: Gurobi and Hexaly for optimal baselines.
2.  **Classical Heuristics**: ALNS, HGS, and look-ahead policies.
3.  **Neural Models**: Attention Models, Graph Neural Networks, and Meta-Learning architectures.

### 1.3 Real-World Simulation

The `test_sim` pipeline simulates multi-day waste collection scenarios with stochastic bin fill rates, capacity constraints, and realistic road networks. This allows policies to be tested under conditions that mirror actual municipal operations.

**Why Simulation Matters**: Unlike static benchmarks, real-world waste collection involves:
- **Temporal Dynamics**: Bin fill levels change daily
- **Stochastic Fill Rates**: Different bins fill at different rates
- **Overflow Penalties**: Letting bins overflow has environmental costs
- **Route Efficiency**: Minimizing travel distance saves fuel and time

---

## 2. High-Level Architecture & Communication

WSmart+ Route uses a **Layered Modular Architecture**. Each layer has distinct responsibilities:

### 2.1 Component Breakdown

```
WSmart-Route/
├── logic/                    # Core Intelligence Layer
│   ├── src/
│   │   ├── models/          # Neural architectures
│   │   ├── policies/        # Classical solvers
│   │   ├── problems/        # Environment physics
│   │   ├── pipeline/        # Training/evaluation orchestration
│   │   └── utils/           # Shared utilities
│   └── test/                # Test suite
├── gui/                      # User Interface Layer
│   ├── src/
│   │   ├── windows/         # Application windows
│   │   ├── tabs/            # Functional tabs
│   │   ├── helpers/         # Background workers
│   │   └── components/      # Reusable widgets
│   └── test/                # GUI tests
├── data/                     # Data Layer
│   ├── vrpp/                # VRPP datasets
│   ├── wcvrp/               # Waste collection datasets
│   └── real_world/          # Real-world instances
├── assets/                   # Assets Layer
│   ├── model_weights/       # Trained models
│   ├── configs/             # Configuration files
│   └── output/              # Experiment outputs
└── scripts/                  # Automation Scripts
```

### 2.2 Communication Protocols

1.  **Logic ⇄ Data**: File I/O via pickle, JSON, and CSV.
2.  **Logic ⇄ GUI**: Qt signals/slots and background workers (`QThread`).
3.  **Configuration**: Centralized YAML configs in `assets/configs/`.
4.  **Logging**: WandB for experiment tracking, loguru for local logs.

### 2.3 Data Flow

```
                              [Run Hydra Config]
                                      ↓
                              [Pipeline Orchestrator]
                                      ↓
                              [Data Generator/Loader]
                                      ↓
                              [Model/Policy Execution]
                                      ↓
                              [Results Logger]
                                      ↓
                              [Output Files/Visualizations]
```

---

## 3. Logic Layer: The Intelligence Core (`logic/src/`)

The `logic/src/` directory contains all the algorithmic intelligence—neural models, classical policies, problem definitions, and training pipelines.

### 3.1 Neural Models (`logic/src/models/`)

#### AttentionModel (`models/attention_model.py`)

The **Attention Model (AM)** is the flagship neural architecture for constructive routing. It builds solutions step-by-step by selecting the next node to visit based on learned attention weights.

**Architecture Overview:**
```python
class AttentionModel(nn.Module):
    def __init__(self, problem, embedding_dim=128, n_encode_layers=3, ...):
        # 1. Context Embedder: Encodes problem-specific information
        self.embedder = ContextEmbedder(problem)

        # 2. Graph Attention Encoder: Processes node features
        self.encoder = GraphAttentionEncoder(
            n_heads=8,
            n_layers=n_encode_layers,
            node_dim=embedding_dim
        )

        # 3. Decoder: Selects next node autoregressively
        self.decoder = AttentionDecoder(
            n_heads=8,
            embedding_dim=embedding_dim
        )
```

**Forward Pass Lifecycle:**
1.  **Embedding**: Nodes (bins) are embedded based on their features (location, demand, fill level).
2.  **Encoding**: Multi-head attention aggregates neighborhood information across all nodes.
3.  **Decoding**: At each step, the decoder computes attention scores over feasible nodes, samples or greedily selects the next node.
4.  **Masking**: Invalid moves (over-capacity, already visited) are masked with `-inf` before softmax.

**Usage Example:**
```python
from logic.src.models.attention_model import AttentionModel
from logic.src.problems.vrpp import VRPP

# Create model
problem = VRPP.NAME
model = AttentionModel(
    problem=problem,
    embedding_dim=128,
    n_encode_layers=3,
    n_heads=8
)

# Forward pass
batch = {
    'coords': torch.rand(32, 50, 2),    # 32 instances, 50 nodes
    'demand': torch.rand(32, 50),
    'prize': torch.rand(32, 50) * 10
}
tours, log_probs = model(batch, return_pi=True)
```

#### GATLSTManager (`models/gat_lstm_manager.py`)

The **GAT-LSTM Manager** is a high-level agent for Hierarchical Reinforcement Learning (HRL). It decides *when* to trigger a collection route based on temporal patterns.

**Components:**
- **GAT Encoder**: Processes spatial bin distribution.
- **LSTM**: Captures temporal dependencies (bin fill history over days).
- **Gating Probability**: Outputs a probability of triggering collection at current timestep.

**Usage in HRL:**
```python
from logic.src.models.gat_lstm_manager import GATLSTManager

manager = GATLSTManager(
    input_dim=64,
    hidden_dim=128,
    n_layers=2
)

# Process temporal state
bin_history = torch.rand(32, 7, 50, 4)  # 7 days history, 50 bins, 4 features
gate_prob = manager(bin_history)  # Probability of triggering collection
```

#### TemporalAM (`models/temporal_am.py`)

A variant of the Attention Model designed to handle time-dependent features directly within the attention mechanism.

**Key Differences:**
- Incorporates day-of-week embeddings
- Uses temporal positional encodings
- Can process sequences of problem states

#### MetaRNN (`models/meta_rnn.py`)

Implements **Meta-Learning** to generalize across different problem distributions (e.g., different cities, waste types).

**Training Procedure:**
1.  Sample a task (e.g., Rio Maior with plastic waste).
2.  Adapt model on support set (few-shot learning).
3.  Evaluate on query set.
4.  Update meta-parameters to minimize query loss.

```python
from logic.src.models.meta_rnn import MetaRNN

meta_model = MetaRNN(
    embedding_dim=128,
    hidden_dim=256,
    n_tasks=10
)

# Meta-training loop
for task in task_distribution:
    support_set, query_set = task.split()
    adapted_params = meta_model.adapt(support_set)
    loss = meta_model.evaluate(query_set, adapted_params)
    loss.backward()
```

#### DeepDecoderAM (`models/deep_decoder_am.py`)

An enhanced version of the Attention Model with a deeper decoder architecture:
- Multiple decoder layers instead of single-layer attention
- Improved long-range dependency modeling
- Better performance on larger problem instances

#### PointerNetwork (`models/pointer_network.py`)

The classic Pointer Network architecture for comparison:
- RNN encoder instead of Transformer
- Attention-based pointing mechanism
- Useful baseline for ablation studies

### 3.2 Graph Encoders & Decoders (`logic/src/models/subnets/`)

#### Encoders

| Encoder | File | Description |
|---------|------|-------------|
| **GATEncoder** | `gat_encoder.py` | Multi-head Graph Attention |
| **GACEncoder** | `gac_encoder.py` | Graph Attention Convolution with edge features |
| **TGCEncoder** | `tgc_encoder.py` | Transformer-style Graph Convolution |
| **GGACEncoder** | `ggac_encoder.py` | Gated Graph Attention Convolution |
| **GCNEncoder** | `gcn_encoder.py` | Standard Graph Convolutional Network |
| **MLPEncoder** | `mlp_encoder.py` | Simple MLP (no graph structure) |
| **PointerEncoder** | `ptr_encoder.py` | RNN-based encoder for Pointer Networks |

**GATEncoder Example:**
```python
from logic.src.models.subnets.gat_encoder import GATEncoder

class GATEncoder(nn.Module):
    def forward(self, x, edge_index):
        # x: (batch_size, num_nodes, node_dim)
        # edge_index: (2, num_edges)

        for layer in self.layers:
            # Multi-head attention aggregation
            x = layer(x, edge_index)
            x = self.normalization(x)
            x = self.activation(x)

        return x  # Encoded node embeddings
```

**Distance-Aware Encoding:**
```python
from logic.src.models.modules.distance_graph_convolution import DistanceAwareGC

# Scale attention by physical distance
encoder = DistanceAwareGC(
    embedding_dim=128,
    distance_scaling='exponential'  # or 'inverse', 'learned'
)
```

#### Decoders

| Decoder | File | Description |
|---------|------|-------------|
| **AttentionDecoder** | `attention_decoder.py` | Standard attention-based decoder |
| **GATDecoder** | `gat_decoder.py` | Graph attention decoder |
| **PointerDecoder** | `ptr_decoder.py` | RNN pointing mechanism |

**AttentionDecoder Example:**
```python
def forward(self, state, encoder_output, mask):
    # Compute compatibility scores
    query = self.project_context(state)
    keys = self.project_nodes(encoder_output)

    scores = torch.matmul(query, keys.transpose(-1, -2)) / sqrt(d_k)

    # Apply mask (invalid actions = -inf)
    scores = scores.masked_fill(mask, float('-inf'))

    # Sample action
    probs = F.softmax(scores, dim=-1)
    action = torch.multinomial(probs, 1)

    return action, probs
```

### 3.3 Classical Policies (`logic/src/policies/`)

#### ALNS (`policies/adaptive_large_neighborhood_search.py`)

**Adaptive Large Neighborhood Search** is a metaheuristic that iteratively destroys and repairs solutions.

**Algorithm:**
1.  **Destroy**: Remove a subset of customers from the current solution.
    - Random removal
    - Worst removal (highest cost customers)
    - Related removal (spatially clustered)
2.  **Repair**: Reinsert removed customers using a construction heuristic.
    - Greedy insertion (minimize cost increase)
    - Regret-k insertion
3.  **Acceptance**: Accept new solution if better, or with probability based on simulated annealing.
4.  **Adaptation**: Adjust destroy/repair operator weights based on success rates.

**Configuration:**
```python
alns_config = {
    'destroy_operators': ['random', 'worst', 'related'],
    'repair_operators': ['greedy', 'regret2', 'regret3'],
    'iterations': 10000,
    'temperature': 100,
    'cooling_rate': 0.9975,
    'destroy_fraction': 0.3,
    'segment_size': 100,
}
```

#### BCP (`policies/branch_cut_and_price.py`)

**Branch-Cut-and-Price** is an exact method for solving the VRPP optimally.

**Components:**
- **Column Generation**: Iteratively generates promising routes.
- **Branch-and-Bound**: Explores solution tree to find optimal integer solution.
- **Cutting Planes**: Strengthens LP relaxation with valid inequalities.

**Integration**: Uses Gurobi, OR-Tools, or VRPy as the backend solver.

```python
from logic.src.policies.branch_cut_and_price import BranchCutAndPrice

bcp = BranchCutAndPrice(
    solver='gurobi',
    time_limit=300,
    mip_gap=0.01,
    num_threads=4
)

solution = bcp.solve(instance)
```

#### HGS (`policies/hybrid_genetic_search.py`)

**Hybrid Genetic Search** combines evolutionary algorithms with local search.

**Key Features:**
- **Population Management**: Maintains diverse population of solutions.
- **Crossover**: OX (Order Crossover) to create offspring.
- **Local Search**: 2-opt, relocate, and swap operators.
- **Split Algorithm**: Decodes giant tour into feasible multi-route solutions.

```python
from logic.src.policies.hybrid_genetic_search import HybridGeneticSearch

hgs = HybridGeneticSearch(
    population_size=100,
    n_elite=10,
    n_close=5,
    max_iterations=1000,
    local_search_intensity=3
)

solution = hgs.solve(instance)
```

#### LookAhead (`policies/look_ahead.py`)

A planning agent that optimizes routes over a future window (N days):

```python
from logic.src.policies.look_ahead import LookAhead

policy = LookAhead(
    horizon=7,  # Plan 7 days ahead
    sub_solver='alns',
    confidence_factor=0.9
)

action = policy.get_action(current_state)
```

#### Regular (`policies/regular.py`)

A baseline policy that visits every bin on a fixed schedule:

```python
from logic.src.policies.regular import Regular

policy = Regular(interval=3)  # Visit every 3 days
```

#### LastMinute (`policies/last_minute.py`)

Trigger-based policy that initiates collection only when bin levels exceed a threshold:

```python
from logic.src.policies.last_minute import LastMinute

policy = LastMinute(threshold=0.9)  # Trigger at 90% full
```

### 3.4 Problem Environments (`logic/src/problems/`)

#### VRPP (`problems/vrpp/`)

**Vehicle Routing Problem with Profits** models scenarios where:
- Visiting a node yields a **reward** (profit).
- Not all nodes need to be visited.
- Objective: Maximize $ \text{Profit} - \text{Cost} $.

**State Representation:**
```python
class StateVRPP:
    coords: Tensor      # (batch, nodes, 2)
    demand: Tensor      # (batch, nodes)
    prize: Tensor       # (batch, nodes)
    depot: Tensor       # (batch, 1)
    visited: Tensor     # (batch, nodes) boolean mask
    current_node: Tensor
    remaining_capacity: Tensor
```

**Reward Calculation:**
```python
def get_reward(self, dataset, pi):
    # Total prize collected
    collected = (dataset['prize'] * pi.visited).sum(dim=-1)

    # Total travel cost
    cost = self._calculate_route_cost(dataset['coords'], pi)

    # Net reward
    return collected - cost
```

#### CWCVRP (`problems/wcvrp/`)

**Capacitated Waste Collection VRP** extends VRPP with:
- **Temporal Dynamics**: Bin fill levels increase over time.
- **Multi-Day Scenarios**: Decisions affect future days.
- **Capacity Constraints**: Vehicle capacity limits.

**State Transitions:**
```python
def step(self, action):
    # Execute collection action
    collected = self.collect_bins(action)

    # Update bin levels
    self.advance_day()
    self.fill_bins()

    # Calculate reward
    reward = self.calculate_reward(collected)

    # Check termination
    done = self.current_day >= self.max_days

    return observation, reward, done, info
```

**Problem Variants:**

| Variant | Description |
|---------|-------------|
| VRPP | Vehicle Routing Problem with Profits |
| CVRPP | Capacitated VRPP |
| WCVRP | Waste Collection VRP |
| CWCVRP | Capacitated Waste Collection VRP |
| SDWCVRP | Stochastic-Dynamic Waste Collection VRP |
| SCWCVRP | Selective CWCVRP |

### 3.5 The Simulator Engine (`logic/src/pipeline/simulator/`)

The simulator is the "physics engine" for waste collection scenarios.

#### Simulation (`simulator/simulation.py`)

Main orchestrator for large-scale experiments:

```python
class Simulation:
    def run(self, policies, n_days, n_samples):
        results = defaultdict(list)

        for sample in range(n_samples):
            for policy in policies:
                # Initialize environment
                bins = Bins(area, waste_type, size)

                # Run simulation
                for day in range(n_days):
                    action = policy.get_action(bins.state)
                    reward, cost = self.execute_action(action)
                    bins.advance_day()

                    results[policy].append({
                        'day': day,
                        'reward': reward,
                        'cost': cost
                    })

        return results
```

#### Bins (`simulator/bins.py`)

Manages the population of waste bins:

**Fill Rate Models:**
- **Gamma Distribution**: $ X \sim \Gamma(\alpha, \beta) $
- **Empirical Distribution**: Sampled from historical data
- **Deterministic**: Fixed fill rates for testing

```python
class Bins:
    def advance_day(self):
        for bin in self.bins:
            fill_amount = self.sample_fill_rate(bin)
            bin.level = min(bin.level + fill_amount, bin.capacity)

            if bin.level > bin.capacity * self.overflow_threshold:
                self.overflow_count += 1

    def sample_fill_rate(self, bin):
        if self.distribution == 'gamma':
            return np.random.gamma(self.alpha, self.beta)
        elif self.distribution == 'empirical':
            return np.random.choice(self.historical_data)
```

#### Network (`simulator/network.py`)

Handles distance matrices and routing networks:

**Distance Computation:**
- **OpenStreetMap**: Real road network distances via `osmnx`
- **Euclidean**: Straight-line approximation for testing
- **Google Maps**: API-based realistic travel times

```python
from logic.src.pipeline.simulator.network import Network

network = Network(
    area='riomaior',
    distance_type='osm',  # or 'euclidean', 'google'
    cache_dir='data/distance_matrices/'
)

distance = network.get_distance(node_a, node_b)
route_cost = network.compute_route_cost(tour)
```

#### Actions (`simulator/actions.py`)

Command Pattern implementation for simulation steps:

```python
class FillAction(Action):
    """Advance bin fill levels by one day."""
    def execute(self, state):
        state.bins.advance_day()
        return state

class CollectAction(Action):
    """Execute collection route."""
    def execute(self, state, route):
        collected = state.bins.collect(route)
        state.vehicle.capacity -= collected
        return state
```

#### States (`simulator/states.py`)

State Pattern for simulation lifecycle:

```python
class InitializingState(SimState):
    def handle(self, context):
        context.load_data()
        context.initialize_bins()
        return RunningState()

class RunningState(SimState):
    def handle(self, context):
        if context.day >= context.max_days:
            return FinishingState()
        context.execute_day()
        return self

class FinishingState(SimState):
    def handle(self, context):
        context.save_results()
        return None
```

### 3.6 Reinforcement Learning Pipeline (`logic/src/pipeline/reinforcement_learning/`)

#### REINFORCE (`reinforcement_learning/core/reinforce.py`)

Implements policy gradient algorithms:

**REINFORCE Algorithm:**
```python
def train_batch(self, batch):
    # Forward pass
    log_probs, actions = self.model(batch)

    # Calculate rewards
    costs = self.problem.calculate_cost(actions)

    # Baseline (reduce variance)
    baseline = self.baseline.estimate(batch)
    advantage = -costs - baseline

    # Policy gradient
    loss = -(log_probs * advantage).mean()

    # Backward pass
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()
```

**Available RL Algorithms:**

| Algorithm | File | Description |
|-----------|------|-------------|
| REINFORCE | `reinforce.py` | Vanilla policy gradient |
| PPO | `reinforce.py` | Proximal Policy Optimization |
| SAPO | `reinforce.py` | Self-Adaptive Policy Optimization |
| GSPO | `reinforce.py` | Generalized Self-Play Optimization |
| DR-GRPO | `reinforce.py` | Distributional Robust GRPO |

**Baseline Options:**
- **Rollout**: Greedy evaluation of current policy
- **Exponential**: Moving average of past costs
- **Critic**: Trained value network
- **POMO**: Policy Optimization with Multiple Optima

#### Epoch Manager (`reinforcement_learning/core/epoch.py`)

Orchestrates the inner training loop:

```python
class EpochManager:
    def run_epoch(self, model, dataloader, optimizer):
        epoch_loss = 0

        for batch in dataloader:
            # Move to device
            batch = move_to_device(batch, self.device)

            # Training step
            loss = self.reinforce.train_batch(batch)

            # Accumulate
            epoch_loss += loss

        return epoch_loss / len(dataloader)
```

#### DEHB (`reinforcement_learning/hyperparameter_optimization/dehb.py`)

**Differential Evolution Hyperband** for automated hyperparameter tuning:

**Algorithm:**
1.  Initialize population of configurations.
2.  Allocate resources via Successive Halving.
3.  Evolve population using Differential Evolution mutation.
4.  Select best configuration based on validation performance.

```python
from logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.dehb import DEHB

search_space = {
    'lr_model': (1e-5, 1e-3, 'log'),
    'embedding_dim': [64, 128, 256],
    'n_encode_layers': [2, 3, 4, 5],
    'batch_size': [256, 512, 1024],
}

dehb = DEHB(
    search_space=search_space,
    min_budget=1,
    max_budget=100,
    n_workers=4
)

best_config = dehb.run(objective_function, n_trials=100)
```

---

## 4. GUI Layer: The Command Center (`gui/src/`)

The GUI provides visual tools for training, evaluation, and analysis.

### 4.1 PySide6 Architecture

**MainWindow (`windows/main_window.py`):**
- Central container with tabbed interface
- Menu bar for file operations, help
- Status bar for real-time feedback

**Tab Structure:**
- **Training Tab**: Configure and launch training runs
- **Evaluation Tab**: Test trained models
- **Simulator Tab**: Run multi-day simulations
- **Analysis Tab**: Visualize results and logs
- **File System Tab**: Manage datasets and models
- **Meta-RL Tab**: Meta-learning experiments
- **HPO Tab**: Hyperparameter optimization

### 4.2 Background Workers (`gui/src/helpers/`)

To keep the GUI responsive, heavy computations run in separate threads.

#### ChartWorker (`helpers/chart_worker.py`)

**Purpose**: Parse simulation logs and emit chart data.

```python
class ChartWorker(QThread):
    data_ready = Signal(dict)

    def run(self):
        while self.running:
            # Read log file
            data = self.parse_log(self.log_path)

            # Emit to main thread
            self.data_ready.emit(data)

            self.msleep(1000)  # Update every second
```

#### DataLoaderWorker (`helpers/data_loader_worker.py`)

**Purpose**: Load large datasets asynchronously.

```python
class DataLoaderWorker(QThread):
    data_loaded = Signal(object)
    error = Signal(str)

    def run(self):
        try:
            data = self.load_dataset(self.path)
            self.data_loaded.emit(data)
        except Exception as e:
            self.error.emit(str(e))
```

#### FileTailerWorker (`helpers/file_tailer_worker.py`)

**Purpose**: Stream log files in real-time (like `tail -f`).

### 4.3 Real-Time Visualization

**Matplotlib Integration:**
- Embedded canvases using `FigureCanvasQTAgg`
- Interactive plots with navigation toolbar
- Real-time updates via worker signals

**Folium Maps:**
- HTML export of route visualizations
- Interactive markers for bins and depots
- Color-coded routes by policy

---

## 5. Algorithm Deep Dives

### 5.1 Attention Mechanism for Routing

The core insight: routing decisions depend on **context** (current state) and **candidates** (feasible next nodes).

**Mathematical Formulation:**

1.  **Query**: Current state embedding $ q = W_q \cdot h_{state} $
2.  **Keys**: Node embeddings $ K = W_k \cdot H_{nodes} $
3.  **Compatibility**: $ u_i = \frac{q \cdot k_i}{\sqrt{d_k}} $
4.  **Masking**: $ u_i := -\infty $ if node $i$ is infeasible
5.  **Probabilities**: $ p_i = \frac{\exp(u_i)}{\sum_j \exp(u_j)} $
6.  **Selection**: Sample $ a \sim \text{Categorical}(p) $ or $ a = \arg\max_i p_i $ (greedy)

**Why It Works:**
- Attention learns to prioritize nodes based on distance, demand, and reward.
- Multi-head attention captures diverse decision criteria (e.g., minimize distance vs. maximize profit).
- The autoregressive nature allows learning sequential dependencies.

### 5.2 REINFORCE with Baselines

**Vanilla REINFORCE:**
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau) \right] $$

**Problem**: High variance in gradient estimates.

**Solution**: Subtract a baseline $ b $:
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ (R(\tau) - b) \cdot \nabla_\theta \log \pi_\theta(\tau) \right] $$

**Baseline Options in WSmart+ Route:**

1.  **Exponential Baseline**: $ b = 0.9 \cdot b_{prev} + 0.1 \cdot R(\tau) $
2.  **Critic Network**: $ b = V_\phi(s) $ learned via MSE loss
3.  **Rollout Baseline**: $ b = R(\tau_{greedy}) $ from greedy policy
4.  **POMO**: Generate multiple solutions per instance, use best as baseline

### 5.3 Adaptive Large Neighborhood Search (ALNS)

**Pseudo-code:**
```
Initialize solution S_current
Initialize operator weights w_destroy, w_repair

while not terminated:
    # Select operators
    destroy_op = roulette_wheel_selection(w_destroy)
    repair_op = roulette_wheel_selection(w_repair)

    # Apply operators
    S_destroyed = destroy_op(S_current)
    S_new = repair_op(S_destroyed)

    # Acceptance criterion
    if accept(S_new, S_current, temperature):
        S_current = S_new
        if cost(S_new) < cost(S_best):
            S_best = S_new
            update_weights(destroy_op, repair_op, score=3)
        else:
            update_weights(destroy_op, repair_op, score=2)
    else:
        update_weights(destroy_op, repair_op, score=1)

    # Cool down
    temperature *= cooling_rate

return S_best
```

**Key Parameters:**
- `destroy_fraction`: Percentage of customers to remove (default: 0.3)
- `temperature`: Initial temperature for simulated annealing
- `cooling_rate`: Temperature decay (default: 0.99)

### 5.4 Branch-Cut-and-Price

**Overview**: Exact method that combines:
- **Branch-and-Bound**: Tree search over solution space
- **Column Generation**: Efficiently generates promising routes
- **Cutting Planes**: Adds constraints to tighten LP relaxation

**Workflow:**
1.  **Master Problem**: Select subset of routes to cover all customers
2.  **Pricing Problem**: Find new routes with negative reduced cost
3.  **Branch**: If solution is fractional, branch on fractional variable
4.  **Cut**: Add valid inequalities to strengthen LP

**Complexity**: Exponential worst-case, but often practical for instances up to 200 nodes.

### 5.5 Hybrid Genetic Search (HGS)

**Overview**: Evolutionary algorithm with local search intensification.

**Key Components:**

1. **Population Structure**:
   - Feasible population
   - Infeasible population (for diversity)
   - Elite solutions

2. **Genetic Operators**:
   - Order Crossover (OX)
   - Mutation via local search

3. **Local Search**:
   - 2-opt (reverse segment)
   - Relocate (move customer)
   - Swap (exchange customers)
   - Or-opt (move sequence)

4. **Split Algorithm**:
   - Decodes giant tour into feasible routes
   - Respects vehicle capacity constraints

### 5.6 Hierarchical Reinforcement Learning

**Two-Level Architecture:**

1. **Manager (High-Level)**:
   - Observes temporal bin states
   - Decides *when* to trigger collection
   - Trained with REINFORCE

2. **Worker (Low-Level)**:
   - Given trigger signal, solves routing
   - Uses Attention Model
   - Trained separately or jointly

**Communication:**
```python
# Manager observes state
gate_prob = manager(temporal_state)
trigger = bernoulli(gate_prob)

if trigger:
    # Worker solves routing
    route = worker(current_state)
    execute(route)
```

---

## 6. Development Life Cycle

### 6.1 Setup & Environment

**Quick Start:**
```bash
# Clone repository
git clone https://github.com/ACFHarbinger/WSmart-Route.git
cd WSmart-Route

# Setup environment
uv sync
source .venv/bin/activate

# Verify installation
python main.py test_suite --module test_models
```

### 6.2 Training Your First Model

**Step 1: Generate Data**
```bash
# Generate training data (on-the-fly)
# Or pre-generate validation/test data
python main.py generate_data val --problem vrpp --graph_sizes 20 --seed 1234 --data_distribution gamma1
python main.py generate_data test --problem vrpp --graph_sizes 20 --seed 1234 --data_distribution gamma1
```

**Step 2: Train Model**
```bash
python main.py train_lightning model=am env.name=vrpp env.num_loc=50 \
  --n_epochs 100 \
  --batch_size 512 \
  --lr_model 1e-4 \
  --baseline rollout \
  --val_dataset data/vrpp/vrpp20_val_seed1234.pkl
```

**Step 3: Monitor Training**
```bash
# Watch logs
tail -f outputs/vrpp_20_*/log.txt

# Or use WandB
wandb login
python main.py train --wandb
```

**Step 4: Evaluate Model**
```bash
python main.py eval \
  data/vrpp/vrpp20_test_seed1234.pkl \
  --model assets/model_weights/vrpp_20/am/epoch-99.pt \
  --decode_strategy greedy
```

### 6.3 Running Simulations

**Basic Simulation:**
```bash
python main.py test_sim \
  --policies regular last_minute gurobi \
  --problem vrpp \
  --size 20 \
  --days 31 \
  --data_distribution gamma1 \
  --n_vehicles 1
```

**With Neural Model:**
```bash
python main.py test_sim \
  --policies am \
  --problem vrpp \
  --size 20 \
  --days 31 \
  --model_path assets/model_weights/vrpp_20/am/epoch-99.pt \
  --decode_strategy greedy
```

**Multi-Sample Testing:**
```bash
python main.py test_sim \
  --policies gurobi alns am \
  --problem cwcvrp \
  --size 50 \
  --days 365 \
  --n_samples 10 \
  --cpu_cores -1 \
  --resume
```

### 6.4 Using the GUI

```bash
python main.py gui
```

**Workflow:**
1.  **Training Tab**: Configure model, problem, and hyperparameters → Click "Start Training"
2.  **Analysis Tab**: Load training logs, view convergence curves
3.  **Simulator Tab**: Set policies, days, area → Click "Run Simulation"
4.  **Results**: View aggregate statistics, route maps, and performance comparisons

### 6.5 Hyperparameter Optimization

**Random Search:**
```bash
python main.py hp_optim \
  --model am \
  --problem vrpp \
  --graph_size 20 \
  --search_strategy random \
  --n_trials 50 \
  --budget 10
```

**DEHB (Differential Evolution Hyperband):**
```bash
python main.py hp_optim \
  --model am \
  --problem vrpp \
  --graph_size 20 \
  --search_strategy dehb \
  --min_budget 1 \
  --max_budget 50 \
  --n_workers 4
```

### 6.6 Meta-Reinforcement Learning

**Training MetaRNN:**
```bash
python main.py mrl_train \
  --model meta_rnn \
  --problem vrpp \
  --graph_size 20 \
  --n_tasks 10 \
  --k_support 5 \
  --k_query 10 \
  --meta_lr 1e-3
```

---

## 7. Extending WSmart+ Route

### 7.1 Adding a New Neural Architecture

**Step 1: Create Model File**
```python
# logic/src/models/my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    NAME = 'mymodel'

    def __init__(self, problem, embedding_dim=128, **kwargs):
        super().__init__()
        self.embedder = ContextEmbedder(problem)
        self.encoder = MyCustomEncoder(embedding_dim)
        self.decoder = MyCustomDecoder(embedding_dim)

    def forward(self, input, return_pi=False):
        embeddings = self.embedder(input)
        encoded = self.encoder(embeddings)
        actions, log_probs = self.decoder(encoded)

        if return_pi:
            return actions, log_probs
        return actions
```

**Step 2: Register in Factory**
```python
# logic/src/models/model_factory.py
from logic.src.models.my_model import MyModel

MODEL_REGISTRY = {
    'am': AttentionModel,
    'mymodel': MyModel,
    # ... other models
}

def get_model(name, problem, opts):
    return MODEL_REGISTRY[name](problem, **opts)
```

**Step 3: Add CLI Support**
```python
# logic/src/utils/parsers/train_parser.py
parser.add_argument('--model', choices=['am', 'transgcn', 'mymodel', ...])
```

**Step 4: Write Tests**
```python
# logic/test/test_models.py
class TestMyModel:
    def test_forward_pass(self):
        model = MyModel(problem='vrpp')
        batch = create_test_batch()
        output = model(batch)
        assert output is not None
```

### 7.2 Adding a New Classical Policy

**Step 1: Create Policy File**
```python
# logic/src/policies/my_policy.py
def policy_my_heuristic(state, opts):
    """
    Custom heuristic policy.

    Args:
        state: Current problem state (bins, depot, vehicle)
        opts: Configuration options

    Returns:
        route: List of node indices
        cost: Total route cost
    """
    # Your logic here
    route = construct_route(state, opts)
    cost = calculate_cost(route, state)

    return route, cost
```

**Step 2: Register Policy**
```python
# logic/src/policies/__init__.py
from .my_policy import policy_my_heuristic

__all__ = [..., 'policy_my_heuristic']
```

**Step 3: Add to Policy Registry**
```python
# logic/src/utils/definitions.py
POLICY_REGISTRY = {
    'regular': Regular,
    'my_heuristic': policy_my_heuristic,
    # ...
}
```

**Step 4: Use in Simulation**
```bash
python main.py test_sim --policies my_heuristic --size 20 --days 31
```

### 7.3 Adding a New Problem Variant

**Step 1: Define Problem Class**
```python
# logic/src/problems/my_problem/problem.py
from logic.src.problems.base_problem import BaseProblem

class MyProblem(BaseProblem):
    NAME = 'my_problem'

    def get_costs(self, dataset, pi):
        # Calculate routing costs
        return costs

    def get_reward(self, dataset, pi):
        # Calculate rewards
        return rewards

    @staticmethod
    def make_dataset(*args, **kwargs):
        # Generate problem instances
        return dataset
```

**Step 2: Add State Definitions**
```python
# logic/src/problems/my_problem/state_my_problem.py
from logic.src.problems.base_state import BaseState

class StateMyProblem(BaseState):
    def __init__(self, input):
        self.coords = input['coords']
        self.demand = input['demand']
        # ... custom attributes

    def get_mask(self):
        # Return feasibility mask
        return mask

    def update(self, action):
        # Update state after action
        return new_state
```

**Step 3: Register Problem**
```python
# logic/src/utils/setup_utils.py
from logic.src.problems.my_problem import MyProblem

PROBLEM_REGISTRY = {
    'vrpp': VRPP,
    'cwcvrp': CWCVRP,
    'my_problem': MyProblem,
}
```

### 7.4 Adding a New Encoder

**Step 1: Create Encoder**
```python
# logic/src/models/subnets/my_encoder.py
import torch.nn as nn

class MyEncoder(nn.Module):
    def __init__(self, embedding_dim, n_layers, n_heads, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            MyEncoderLayer(embedding_dim, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x, edge_index=None):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
```

**Step 2: Register Encoder**
```python
# logic/src/models/model_factory.py
ENCODER_REGISTRY = {
    'gat': GATEncoder,
    'gcn': GCNEncoder,
    'my_encoder': MyEncoder,
}
```

### 7.5 Adding a New Decoder

**Step 1: Create Decoder**
```python
# logic/src/models/subnets/my_decoder.py
class MyDecoder(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_heads)

    def forward(self, context, nodes, mask):
        # Compute attention scores
        scores = self.attention(context, nodes)

        # Apply mask
        scores = scores.masked_fill(mask, float('-inf'))

        # Get probabilities
        probs = F.softmax(scores, dim=-1)

        return probs
```

### 7.6 Adding a New RL Algorithm

**Step 1: Create Algorithm**
```python
# logic/src/pipeline/reinforcement_learning/core/my_algorithm.py
class MyRLAlgorithm:
    def __init__(self, model, optimizer, **kwargs):
        self.model = model
        self.optimizer = optimizer

    def train_batch(self, batch):
        # Forward pass
        outputs = self.model(batch)

        # Calculate loss
        loss = self.compute_loss(outputs, batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, outputs, batch):
        # Your loss computation
        raise NotImplementedError
```

---

## 8. Testing & Quality Assurance

### 8.1 Test Suite Overview

**Directory Structure:**
```
logic/test/
├── test_models.py          # Neural architecture tests
├── test_policies.py        # Classical policy tests
├── test_simulator.py       # Simulation engine tests
├── test_generate_data.py   # Data generation tests
├── test_train.py           # Training pipeline tests
└── test_visualize.py       # Visualization tests
```

**Running Tests:**
```bash
# All tests
python main.py test_suite

# Specific module
python main.py test_suite --module test_models

# Specific test
python main.py test_suite --test test_attention_model

# With coverage
pytest --cov=logic/src logic/test/
```

### 8.2 Writing New Tests

**Example Test:**
```python
# logic/test/test_my_feature.py
import pytest
import torch
from logic.src.models.my_model import MyModel

class TestMyModel:
    @pytest.fixture
    def model(self):
        return MyModel(problem='vrpp', embedding_dim=128)

    @pytest.fixture
    def sample_input(self):
        batch_size, graph_size = 10, 20
        return {
            'coords': torch.rand(batch_size, graph_size, 2),
            'demand': torch.rand(batch_size, graph_size)
        }

    def test_forward_pass(self, model, sample_input):
        output = model(sample_input)

        assert output is not None
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    def test_variable_batch_sizes(self, model, batch_size):
        input_data = {
            'coords': torch.rand(batch_size, 20, 2),
            'demand': torch.rand(batch_size, 20)
        }
        output = model(input_data)
        assert output.shape[0] == batch_size
```

### 8.3 Coverage Requirements

| Component | Minimum | Target |
|-----------|---------|--------|
| Overall | 60% | 80% |
| `models/` | 70% | 85% |
| `policies/` | 60% | 80% |
| `pipeline/` | 60% | 75% |

---

## 9. Advanced Topics

### 9.1 Multi-GPU Training

```bash
# Use all available GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train --model am

# Specify specific GPUs
CUDA_VISIBLE_DEVICES=0,2 python main.py train --model am
```

**DataParallel in Code:**
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### 9.2 Custom Baselines

**Creating a Custom Baseline:**
```python
# logic/src/pipeline/reinforcement_learning/core/my_baseline.py
class MyBaseline:
    def __init__(self, decay=0.95):
        self.decay = decay
        self.running_mean = None

    def estimate(self, batch):
        if self.running_mean is None:
            return torch.zeros(batch['coords'].shape[0])
        return self.running_mean

    def update(self, rewards):
        if self.running_mean is None:
            self.running_mean = rewards.mean()
        else:
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * rewards.mean()
```

### 9.3 Distance Matrix Computation

**Pre-compute for Real-World Instances:**
```python
from logic.src.pipeline.simulator.network import Network

# Compute OSM distances
network = Network(area='riomaior', distance_type='osm')
network.compute_and_cache_all()

# Load cached matrix
dist_matrix = network.load_distance_matrix()
```

### 9.4 Checkpoint Management

**Save Checkpoints:**
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'baseline_state': baseline.state_dict(),
    'best_val_cost': best_val_cost,
}
torch.save(checkpoint, f'checkpoint-epoch-{epoch}.pt')
```

**Resume from Checkpoint:**
```bash
python main.py train --load_path checkpoint-epoch-50.pt --epoch_start 51
```

### 9.5 Logging and Monitoring

**WandB Integration:**
```python
import wandb

wandb.init(project='wsmart-route', config=opts)

# Log metrics
wandb.log({
    'train_loss': loss,
    'val_cost': val_cost,
    'epoch': epoch
})

# Log model
wandb.watch(model)
```

**TensorBoard:**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Cost/validation', val_cost, epoch)
```

---

## 10. Exhaustive Code Reference

### High-Level File Index

| Category | Key Files | Description |
|----------|-----------|-------------|
| **Neural Models** | `attention_model.py`, `gat_lstm_manager.py`, `meta_rnn.py` | Core neural architectures |
| **Encoders** | `gat_encoder.py`, `gcn_encoder.py`, `tgc_encoder.py` | Graph encoding layers |
| **Trainer** | `pipeline/train_lightning.py` | Central entry for `train`, `meta_train`, `hpo`. Manages device selection, data loading, epoch loops via Lightning. |
| **Policies** | `alns.py`, `bcp.py`, `hgs.py`, `look_ahead.py` | Classical solvers |
| **Problems** | `vrpp/`, `wcvrp/` | Environment definitions |
| **Simulator** | `simulation.py`, `bins.py`, `network.py` | Physics engine |
| **RL Pipeline** | `reinforce.py`, `epoch.py`, `trainers.py` | Training loops |
| **HPO** | `dehb.py`, `hpo.py` | Hyperparameter optimization |
| **GUI** | `main_window.py`, `chart_worker.py` | User interface |

---

## 11. Glossary of Terms

| Term | Definition |
|------|------------|
| **ALNS** | Adaptive Large Neighborhood Search (Metaheuristic) |
| **AM** | Attention Model |
| **BCP** | Branch-Cut-and-Price (Exact method) |
| **Baseline** | Reference value to reduce policy gradient variance |
| **CWC VRP** | Capacitated Waste Collection VRP |
| **Decoder** | Neural component that selects next node |
| **DEHB** | Differential Evolution Hyperband |
| **Encoder** | Neural component that processes node features |
| **GAT** | Graph Attention Network |
| **GCN** | Graph Convolutional Network |
| **HGS** | Hybrid Genetic Search |
| **HRL** | Hierarchical Reinforcement Learning |
| **LSTM** | Long Short-Term Memory (Recurrent network) |
| **Manager** | High-level agent in HRL (decides when to act) |
| **Masking** | Preventing invalid actions in decoder |
| **Meta-Learning** | Learning to learn across task distributions |
| **POMO** | Policy Optimization with Multiple Optima |
| **PPO** | Proximal Policy Optimization |
| **REINFORCE** | Policy gradient RL algorithm |
| **Rollout** | Greedy evaluation of policy for baseline |
| **TSP** | Traveling Salesman Problem |
| **VRP** | Vehicle Routing Problem |
| **VRPP** | Vehicle Routing Problem with Profits |
| **Worker** | Low-level agent in HRL (solves routing) |

---

## 12. Frequently Asked Questions

### Training

**Q: My model isn't learning. What should I check?**
A: Check these in order:
1. Learning rate (try 1e-4 to 1e-3)
2. Batch size (try 256-1024)
3. Baseline type (try 'rollout')
4. Gradient clipping (add `--max_grad_norm 1.0`)

**Q: How do I know when training has converged?**
A: Monitor validation cost. Training is likely converged when:
- Validation cost plateaus for 10+ epochs
- Training loss stabilizes
- Greedy vs. sampling gap is small

**Q: Should I use CPU or GPU?**
A: Always use GPU for training if available. For small inference tasks, CPU is often sufficient.

### Simulation

**Q: How do I compare multiple policies fairly?**
A: Use the same random seed, same number of samples, and same problem instances:
```bash
python main.py test_sim --policies am alns gurobi --seed 42 --n_samples 10
```

**Q: Why are my simulation results different each run?**
A: Set explicit random seeds:
```bash
python main.py test_sim --seed 42 --np_seed 42 --torch_seed 42
```

### Models

**Q: Which model should I use for my problem?**
A:
- Small instances (< 50 nodes): AttentionModel with GATEncoder
- Large instances (> 100 nodes): TransGCN or DeepDecoderAM
- Temporal problems: GATLSTManager + AttentionModel (HRL)
- Distribution shift: MetaRNN

**Q: How do I export a trained model for deployment?**
A:
```python
# Save model
torch.save(model.state_dict(), 'model.pt')

# Load for inference
model = AttentionModel(problem='vrpp')
model.load_state_dict(torch.load('model.pt'))
model.eval()
```

### Troubleshooting

**Q: I get CUDA out of memory errors.**
A: Reduce batch size or graph size. Clear cache between runs:
```python
torch.cuda.empty_cache()
```

**Q: My GUI won't launch.**
A: Check PySide6 installation:
```bash
uv pip install PySide6
python -c "from PySide6.QtWidgets import QApplication; print('OK')"
```

---

**This guide is the living foundation of WSmart+ Route. Happy Routing!**
