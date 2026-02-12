# Routing Policies & Global Optimization

**Module**: `logic/src/policies`
**Purpose**: Comprehensive collection of classical, metaheuristic, exact, and neural routing policies for Combinatorial Optimization.
**Files**: 157 Python modules across 14 major components
**Version**: 3.0
**Last Updated**: February 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Design Patterns](#architecture--design-patterns)
3. [Module Organization](#module-organization)
4. [Policy Adapters](#policy-adapters)
5. [Classical & Metaheuristic Policies](#classical--metaheuristic-policies)
6. [Exact Optimization](#exact-optimization)
7. [Neural Policies](#neural-policies)
8. [Operators Library](#operators-library)
9. [Must-Go Selection Strategies](#must-go-selection-strategies)
10. [Post-Processing & Refinement](#post-processing--refinement)
11. [Local Search](#local-search)
12. [Integration Examples](#integration-examples)
13. [Best Practices](#best-practices)
14. [Quick Reference](#quick-reference)

---

## 1. Overview

The `policies` module provides a comprehensive technical ecosystem of routing optimization algorithms designed to solve diverse variants of the Vehicle Routing Problem (VRP). It integrates **13 distinct policy implementations**, **26 specialized operators**, **6 must-go selection strategies**, and **5 post-processing methods** into a unified operational framework.

### Key Features

- **Unified Adapter Pattern**: Ensures a consistent execution interface across heterogeneous routing policies.
- **Modular Framework**: Facilitates the seamless integration of new solvers and neighborhood search operators.
- **Dynamic Policy Registry**: Automated discovery and instantiation of optimization engines by name.
- **Multi-Engine Integration**: Out-of-the-box support for Gurobi, OR-Tools, Hexaly, and PyVRP.
- **Operator Composability**: Mix and match atomic operators to construct complex metaheuristics.
- **Workflow Standardization**: Template-based execution logic that handles common preprocessing and mapping steps.
- **Integrated Must-Go Selection**: Advanced strategies for identifying and prioritizing mandatory collection points.
- **Post-Processing Pipeline**: Automated route refinement and local search improvement after main solver execution.

### Supported Problems

| Problem     | Description                         | Policies                      |
| ----------- | ----------------------------------- | ----------------------------- |
| **TSP**     | Traveling Salesman Problem          | LKH, fast_tsp, exact solvers  |
| **CVRP**    | Capacitated Vehicle Routing Problem | HGS, OR-Tools, Gurobi         |
| **VRPP**    | VRP with Profits                    | Gurobi, Hexaly                |
| **WCVRP**   | Waste Collection VRP                | ALNS, HGS, SANS, SISR, Neural |
| **SDWCVRP** | Stochastic Demand WCVRP             | Neural, Adaptive Heuristics   |

### Policy Categories

1. **Exact Methods** (BCP): Branch-Cut-and-Price via Gurobi/OR-Tools/VRPy
2. **Metaheuristics**:
   - ALNS (Adaptive Large Neighborhood Search)
   - HGS (Hybrid Genetic Search)
   - HGS-ALNS (Hybrid of HGS and ALNS)
   - ACO (Ant Colony Optimization - K-Sparse and Hyper-Heuristic variants)
   - SANS (Simulated Annealing Neighborhood Search)
   - SISR (Slack Induction by String Removal)
3. **Neural Policies**: Deep RL models (AM, TAM, DDAM)
4. **Classical Heuristics**: TSP, CVRP solvers (LKH, OR-Tools)
5. **Specialized**: VRPP exact optimizers

---

## 2. Architecture & Design Patterns

The `policies` module utilizes established design patterns to maintain scalability and decouple high-level optimization logic from specific solver implementations.

### Strategic Design Patterns

1.  **Adapter Pattern** (`adapters/`): Provides a unified interface that abstracts the complexities of disparate routing libraries (Gurobi, OR-Tools, etc.).
2.  **Factory Pattern** (`adapters/factory.py`): Centralizes the instantiation logic for all policy adapters.
3.  **Registry Pattern** (`adapters/registry.py`): Facilitates dynamic discovery and decoupled registration of new optimization strategies.
4.  **Template Method Pattern** (`adapters/base_routing_policy.py`): Defines a standardized skeleton for routing execution, ensuring consistent data preparation and result mapping.
5.  **Strategy Pattern** (`other/must_go/`): Enables pluggable logic for customer selection and prioritization.
6.  **Command Pattern** (`operators/`): Encapsulates atomic neighborhood search actions to support composable heuristics.

### Core Interface Definitions

The system relies on the `IPolicyAdapter` interface to ensure type safety and consistent interaction.

```python
from logic.src.interfaces.adapter import IPolicyAdapter

class IPolicyAdapter(ABC):
    """Unified interface for all routing optimization policies."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the optimization policy.

        Returns:
            Tuple containing (optimized_tour, total_cost, extra_metadata).
        """
        pass
```

### Standard Execution Flow

The internal workflow for a policy execution follow a strictly defined sequence to maintain data integrity:

1.  **Discovery**: The `PolicyFactory` identifies the requested adapter by key.
2.  **Initialization**: The adapter is instantiated and its specific hyperparameters are loaded.
3.  **Selection**: If configured, a "Must-Go" strategy filters the target nodes.
4.  **Transformation**: The global distance matrix is subsetted and normalized for the local solver.
5.  **Optimization**: The core engine (e.g., SISR solver, Gurobi) computes the solution.
6.  **Mapping**: Local solver indices are projected back to global system IDs.
7.  **Refinement**: Optional post-processing steps (e.g., path improvement) are applied.
8.  **Output**: The final tour, accurate cost, and diagnostic metadata are returned.

---

## 3. Module Organization

### Core Directory Structure

```
logic/src/policies/
├── __init__.py                              # Main exports
│
├── adapters/                                # Policy Adapter Pattern (13 adapters)
│   ├── __init__.py
│   ├── base_routing_policy.py              # Template base class
│   ├── factory.py                           # PolicyFactory
│   ├── registry.py                          # PolicyRegistry
│   ├── policy_alns.py                       # ALNS adapter
│   ├── policy_bcp.py                        # BCP adapter
│   ├── policy_cvrp.py                       # CVRP adapter
│   ├── policy_hgs.py                        # HGS adapter
│   ├── policy_hgs_alns.py                   # HGS-ALNS adapter
│   ├── policy_hh_aco.py                     # Hyper-Heuristic ACO adapter
│   ├── policy_ks_aco.py                     # K-Sparse ACO adapter
│   ├── policy_lkh.py                        # Lin-Kernighan-Helsgaun adapter
│   ├── policy_neural.py                     # Neural agent adapter
│   ├── policy_sans.py                       # SANS adapter
│   ├── policy_sisr.py                       # SISR adapter
│   ├── policy_tsp.py                        # TSP adapter
│   └── policy_vrpp.py                       # VRPP adapter
│
├── adaptive_large_neighborhood_search/      # ALNS Implementation
│   ├── __init__.py
│   ├── alns.py                              # Custom ALNS implementation
│   ├── alns_package.py                      # Third-party ALNS wrapper
│   ├── ortools_wrapper.py                   # OR-Tools ALNS wrapper
│   └── params.py                            # ALNS parameters
│
├── ant_colony_optimization/                 # ACO Implementations
│   ├── k_sparse_aco/                        # K-Sparse ACO variant
│   │   ├── construction.py                  # Solution construction
│   │   ├── pheromones.py                    # Pheromone management
│   │   ├── solver.py                        # Main solver
│   │   ├── runner.py                        # Execution wrapper
│   │   └── params.py
│   └── hyper_heuristic_aco/                 # Hyper-Heuristic ACO
│       ├── hyper_aco.py                     # Main algorithm
│       ├── hyper_operators.py               # Operator selection
│       ├── runner.py
│       └── params.py
│
├── branch_cut_and_price/                    # Exact Optimization
│   ├── dispatcher.py                        # Engine selection
│   ├── gurobi_engine.py                     # Gurobi solver
│   ├── ortools_engine.py                    # OR-Tools solver
│   └── vrpy_engine.py                       # VRPy solver
│
├── hybrid_genetic_search/                   # HGS Implementation
│   ├── hgs.py                               # Main HGS algorithm
│   ├── pyvrp_wrapper.py                     # PyVRP HGS wrapper
│   ├── evolution.py                         # Evolutionary operators
│   ├── individual.py                        # Individual representation
│   ├── split.py                             # Split algorithm
│   └── params.py
│
├── simulated_annealing_neighborhood_search/ # SANS Implementation
│   ├── heuristics/
│   │   ├── sans.py                          # Main SANS algorithm
│   │   ├── sans_state.py                    # State management
│   │   ├── sans_opt.py                      # Optimization logic
│   │   ├── sans_operators.py                # Operator application
│   │   ├── sans_neighborhoods.py            # Neighborhood structures
│   │   ├── sans_perturbations.py            # Perturbation mechanisms
│   │   └── anneal.py                        # Annealing schedule
│   ├── operators/                           # SANS-specific operators
│   ├── select/                              # Node selection strategies
│   ├── search/                              # Search strategies
│   ├── refinement/                          # Solution refinement
│   └── common/                              # Utilities (routes, distance, etc.)
│
├── slack_induction_by_string_removal/       # SISR Implementation
│   ├── sisr.py                              # Main SISR algorithm
│   ├── solver.py                            # SISR solver
│   └── params.py
│
├── vehicle_routing_problem_with_profits/    # VRPP Solvers
│   ├── interface.py                         # VRPP interface
│   ├── gurobi.py                            # Gurobi VRPP solver
│   └── hexaly.py                            # Hexaly VRPP solver
│
├── neural_agent/                            # Neural Policy Wrapper
│   ├── agent.py                             # NeuralAgent class
│   ├── batch.py                             # Batch processing
│   └── simulation.py                        # Simulation interface
│
├── operators/                               # Operator Library (26 operators)
│   ├── move/                                # Move operators
│   │   ├── relocate.py                      # Relocate single node
│   │   └── swap.py                          # Swap two nodes
│   ├── route/                               # Intra-route operators
│   │   ├── two_opt_intra.py                 # 2-opt within route
│   │   ├── two_opt_star.py                  # 2-opt* between routes
│   │   ├── three_opt_intra.py               # 3-opt within route
│   │   └── swap_star.py                     # Swap* between routes
│   ├── destroy/                             # Destroy operators (ALNS)
│   │   ├── random.py                        # Random removal
│   │   ├── worst.py                         # Worst removal
│   │   ├── cluster.py                       # Cluster removal
│   │   ├── shaw.py                          # Shaw removal
│   │   └── string.py                        # String removal (SISR)
│   ├── repair/                              # Repair operators (ALNS)
│   │   ├── greedy.py                        # Greedy insertion
│   │   ├── regret.py                        # Regret-k insertion
│   │   └── greedy_blink.py                  # Greedy with blink
│   ├── perturbation/                        # Perturbation operators
│   │   ├── perturb.py                       # General perturbation
│   │   └── kick.py                          # Kick operator
│   └── exchange/                            # Exchange operators
│       ├── or_opt.py                        # Or-opt
│       ├── cross.py                         # Cross-exchange
│       ├── ejection.py                      # Ejection chain
│       └── lambda_interchange.py            # λ-interchange
│
├── other/                                   # Auxiliary Components
│   ├── must_go/                             # Must-Go Selection (6 strategies)
│   │   ├── base/                            # Selection framework
│   │   │   ├── selection_context.py         # Context object
│   │   │   ├── selection_factory.py         # Factory
│   │   │   └── selection_registry.py        # Registry
│   │   ├── selection_last_minute.py         # Threshold-based selection
│   │   ├── selection_regular.py             # Fixed-frequency selection
│   │   ├── selection_lookahead.py           # Predictive selection
│   │   ├── selection_revenue.py             # Profit-based selection
│   │   ├── selection_service_level.py       # Statistical selection
│   │   └── selection_combined.py            # Combined strategies
│   └── post_processing/                     # Post-Processing (5 methods)
│       ├── factory.py                       # Factory
│       ├── registry.py                      # Registry
│       ├── fast_tsp.py                      # TSP improvement
│       ├── local_search.py                  # Classical LS
│       ├── random_ls.py                     # Randomized LS
│       ├── ils.py                           # Iterated Local Search
│       └── path.py                          # Path improvement
│
├── local_search/                            # Local Search Base
│   ├── local_search_base.py                 # Base class
│   ├── local_search_hgs.py                  # HGS-specific LS
│   └── local_search_aco.py                  # ACO-specific LS
│
├── cvrp.py                                  # CVRP solver entry
├── tsp.py                                   # TSP solver entry
├── hgs_alns.py                              # HGS-ALNS entry
└── lin_kernighan_helsgaun.py                # LKH entry
```

### Statistics

- **Total Files**: 157 Python modules
- **Policy Adapters**: 13 implementations
- **Operators**: 26 specialized operators
- **Must-Go Strategies**: 6 selection methods
- **Post-Processors**: 5 refinement methods
- **Local Search Variants**: 3 implementations

---

## 4. Policy Adapters

**Directory**: `adapters/`
**Purpose**: Unified interface for all routing policies using Adapter pattern

### 4.1 PolicyFactory

**File**: `adapters/factory.py`

The `PolicyFactory` serves as the centralized entry point for dynamically instantiating policy adapters. It abstracts the underlying engine selection from the high-level routing logic.

```python
from logic.src.policies.adapters import PolicyFactory

# Create policy by name
policy = PolicyFactory.get_adapter("hgs")

# Execute policy
tour, cost, metadata = policy.execute(
    must_go=[1, 5, 10, 15],
    bins=bins_state,
    distance_matrix=dist_matrix,
    area="riomaior",
    waste_type="plastic",
    config=config_dict
)
```

**Supported Policy Names**:

- `"alns"`, `"hgs"`, `"hgs_alns"`, `"bcp"`, `"gurobi"`, `"hexaly"`
- `"ks_aco"`, `"hh_aco"`, `"sisr"`, `"sans"`, `"lkh"`
- `"neural"`, `"am"`, `"ddam"`, `"tam"`, `"transgcn"`
- `"tsp"`, `"cvrp"`, `"vrpp"`

### 4.2 PolicyRegistry

**File**: `adapters/registry.py`

The `PolicyRegistry` maintains a decentralized map of all available optimization engines, enabling automated discovery and type-safe access through the registration decorator.

```python
from logic.src.policies.adapters import PolicyRegistry

# Register custom policy
@PolicyRegistry.register("my_custom_policy")
class MyCustomPolicy(IPolicyAdapter):
    def execute(self, **kwargs):
        # Implementation
        return tour, cost, metadata

# Get registered policy
policy_cls = PolicyRegistry.get("my_custom_policy")
policy = policy_cls()

# List all policies
all_policies = PolicyRegistry.list_policies()
print(all_policies)
# ['alns', 'hgs', 'bcp', 'neural', 'my_custom_policy', ...]
```

### 4.3 BaseRoutingPolicy

**File**: `adapters/base_routing_policy.py`

The `BaseRoutingPolicy` provides a standardized template method for the routing workflow. It encapsulates common logistical operations (validation, data subsetting, result projection) while delegating the core optimization to subclass-specific solver implementations.

```python
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy

class MyPolicy(BaseRoutingPolicy):
    """Custom policy implementation."""

    def _get_config_key(self) -> str:
        """Return config key for parameter loading."""
        return "my_policy"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_demands: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs
    ) -> Tuple[List[List[int]], float]:
        """Run the solver (subclass-specific)."""
        # 1. Initialize solution
        # 2. Run optimization algorithm
        # 3. Return routes and cost
        routes = [[1, 2, 3], [4, 5]]
        solver_cost = 123.45
        return routes, solver_cost
```

**Template Method Steps**:

1. **Validate** must-go inputs
2. **Extract** context (bins, distance matrix, config)
3. **Load** area-specific parameters (Q, R, C)
4. **Create** subset problem (distance matrix for must-go nodes)
5. **Run** solver (subclass-specific implementation)
6. **Map** local indices back to global bin IDs
7. **Compute** total tour cost

---

## 5. Classical & Metaheuristic Policies

### 5.1 ALNS (Adaptive Large Neighborhood Search)

**Directory**: `adaptive_large_neighborhood_search/`
**Adapters**: `policy_alns.py`

Destroy-and-repair metaheuristic with adaptive operator weights.

#### Architecture

```
ALNSPolicy
├── Engine Selection (custom, ortools, package)
├── Destroy Operators
│   ├── Random Removal
│   ├── Worst Removal
│   ├── Cluster Removal
│   ├── Shaw Removal
│   └── String Removal
├── Repair Operators
│   ├── Greedy Insertion
│   ├── Regret-2 Insertion
│   ├── Regret-k Insertion
│   └── Greedy Blink Insertion
└── Adaptive Weight Update (Simulated Annealing acceptance)
```

#### Usage Example

```python
from logic.src.policies import run_alns, ALNSParams

# Configure ALNS
params = ALNSParams(
    time_limit=60.0,
    max_iterations=5000,
    start_temp=100.0,
    cooling_rate=0.995,
    reaction_factor=0.1,
    min_removal=1,
    max_removal_pct=0.3
)

# Run ALNS
routes, cost = run_alns(
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0,
    params=params
)

# Via adapter
from logic.src.policies.adapters import PolicyFactory

alns_policy = PolicyFactory.get_adapter("alns")
tour, cost, _ = alns_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    area="riomaior",
    config={"alns": {"time_limit": 120.0}}
)
```

#### Key Parameters

| Parameter         | Default | Description                                    |
| ----------------- | ------- | ---------------------------------------------- |
| `time_limit`      | 60.0    | Maximum runtime in seconds                     |
| `max_iterations`  | 5000    | Maximum destroy/repair cycles                  |
| `start_temp`      | 100.0   | Initial temperature for SA acceptance          |
| `cooling_rate`    | 0.995   | Temperature decay rate per iteration           |
| `reaction_factor` | 0.1     | Speed of operator weight adaptation            |
| `min_removal`     | 1       | Minimum nodes to destroy                       |
| `max_removal_pct` | 0.3     | Maximum % of nodes to destroy                  |
| `engine`          | custom  | Implementation: `custom`, `ortools`, `package` |

### 5.2 HGS (Hybrid Genetic Search)

**Directory**: `hybrid_genetic_search/`
**Adapters**: `policy_hgs.py`, `policy_hgs_alns.py`

State-of-the-art genetic algorithm with local search and Split procedure.

#### Architecture

```
HGSPolicy
├── Population Management
│   ├── Feasible Solutions
│   └── Infeasible Solutions
├── Genetic Operators
│   ├── OX Crossover (Order Crossover)
│   ├── PMX Crossover (Partially Mapped)
│   └── Mutation (random swaps)
├── Local Search
│   ├── Intra-route: 2-opt, 3-opt
│   ├── Inter-route: relocate, swap, 2-opt*
│   └── Perturbation for diversity
├── Split Algorithm (giant tour → routes)
└── Survivor Selection (elite + diversity)
```

#### Usage Example

```python
from logic.src.policies import run_hgs

# Run HGS
routes, cost = run_hgs(
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0,
    time_limit=60.0,
    population_size=50,
    elite_size=10
)

# Via adapter
hgs_policy = PolicyFactory.get_adapter("hgs")
tour, cost, _ = hgs_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "hgs": {
            "time_limit": 120.0,
            "population_size": 100
        }
    }
)
```

#### 5.2.1 HGS-ALNS Variant

Combines HGS with ALNS operators for enhanced exploration.

```python
hgs_alns_policy = PolicyFactory.get_adapter("hgs_alns")
tour, cost, _ = hgs_alns_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "hgs_alns": {
            "time_limit": 180.0,
            "alns_education_iterations": 50  # ALNS improvement per individual
        }
    }
)
```

### 5.3 ACO (Ant Colony Optimization)

**Directory**: `ant_colony_optimization/`
**Adapters**: `policy_ks_aco.py`, `policy_hh_aco.py`

Bio-inspired algorithm with pheromone-based learning.

#### K-Sparse ACO

Efficient ACO variant using K-nearest neighbors graph.

```python
from logic.src.policies import run_k_sparse_aco

routes, cost = run_k_sparse_aco(
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0,
    n_ants=20,
    alpha=1.0,        # Pheromone importance
    beta=2.0,         # Heuristic importance
    rho=0.1,          # Evaporation rate
    q0=0.9,           # Exploitation vs exploration
    k_sparse=15       # K-nearest neighbors
)
```

#### Hyper-Heuristic ACO

ACO with dynamic operator selection.

```python
from logic.src.policies import run_hyper_heuristic_aco

routes, cost = run_hyper_heuristic_aco(
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0,
    n_ants=20,
    operator_pool=["2opt", "relocate", "swap", "or_opt"]
)
```

### 5.4 SANS (Simulated Annealing Neighborhood Search)

**Directory**: `simulated_annealing_neighborhood_search/`
**Adapters**: `policy_sans.py`

Comprehensive SA-based search with multiple neighborhoods.

```python
sans_policy = PolicyFactory.get_adapter("sans")
tour, cost, _ = sans_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "sans": {
            "initial_temp": 1000.0,
            "cooling_schedule": "geometric",
            "cooling_rate": 0.95,
            "max_iterations": 10000,
            "neighborhoods": ["swap", "relocate", "2opt"]
        }
    }
)
```

### 5.5 SISR (Slack Induction by String Removal)

**Directory**: `slack_induction_by_string_removal/`
**Adapters**: `policy_sisr.py`

Sophisticated destroy-repair with string removal patterns.

```python
from logic.src.policies import run_sisr

routes, cost = run_sisr(
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0,
    string_removal_max_len=10,
    max_removal_pct=0.2,
    blink_probability=0.01
)

# Via adapter
sisr_policy = PolicyFactory.get_adapter("sisr")
tour, cost, _ = sisr_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix
)
```

---

## 6. Exact Optimization

### 6.1 BCP (Branch-Cut-and-Price)

**Directory**: `branch_cut_and_price/`
**Adapters**: `policy_bcp.py`

Exact MIP solvers for optimal solutions.

#### Engine Selection

```python
from logic.src.policies import run_bcp

# Dispatcher selects best available engine
routes, cost = run_bcp(
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0,
    engine="auto"  # Auto-select: Gurobi > OR-Tools > VRPy
)

# Force specific engine
routes, cost = run_bcp(
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0,
    engine="gurobi",  # or "ortools", "vrpy"
    time_limit=300.0,
    mip_gap=0.01
)
```

**Available Engines**:

1. **Gurobi** (`gurobi_engine.py`): Commercial solver (fastest, requires license)
2. **OR-Tools** (`ortools_engine.py`): Google's open-source solver
3. **VRPy** (`vrpy_engine.py`): Python-based column generation

#### Usage via Adapter

```python
bcp_policy = PolicyFactory.get_adapter("bcp")
tour, cost, _ = bcp_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "bcp": {
            "engine": "gurobi",
            "time_limit": 600.0,
            "mip_gap": 0.001  # 0.1% optimality tolerance
        }
    }
)
```

### 6.2 VRPP Exact Solvers

**Directory**: `vehicle_routing_problem_with_profits/`
**Adapters**: `policy_vrpp.py`

Exact optimization for VRPP (profit-oriented VRP).

```python
from logic.src.policies import run_vrpp_optimizer

# Gurobi VRPP
routes, cost, selected_nodes = run_vrpp_optimizer(
    distance_matrix=dist_matrix,
    demands=demands_dict,
    prizes=prizes_dict,  # Profit per node
    capacity=200.0,
    max_length=100.0,
    engine="gurobi",
    Omega=1.0,      # Profit weight
    delta=0.5,      # Distance penalty
    psi=10.0        # Unvisited penalty
)

# Hexaly VRPP (high-performance local search)
routes, cost, selected_nodes = run_vrpp_optimizer(
    distance_matrix=dist_matrix,
    demands=demands_dict,
    prizes=prizes_dict,
    capacity=200.0,
    engine="hexaly",
    time_limit=60.0
)
```

---

## 7. Neural Policies

**Directory**: `neural_agent/`
**Adapters**: `policy_neural.py`

Deep Reinforcement Learning-based routing policies.

### 7.1 NeuralAgent

**File**: `neural_agent/agent.py`

Wrapper around trained neural models (AM, TAM, DDAM).

```python
from logic.src.policies import NeuralAgent
from logic.src.models import load_model

# Load trained model
model = load_model("assets/model_weights/am_wcvrp_100.pt")

# Create agent
agent = NeuralAgent(
    model=model,
    device="cuda",
    decode_type="greedy",  # or "sampling", "beam_search"
    temperature=1.0,
    beam_width=5
)

# Solve single instance
tour, cost = agent.solve(
    coordinates=coords,
    demands=demands,
    capacity=200.0
)

# Solve batch
tours, costs = agent.solve_batch(
    batch_coords=batch_coords,
    batch_demands=batch_demands,
    capacity=200.0
)
```

### 7.2 Via Adapter

```python
neural_policy = PolicyFactory.get_adapter("neural")

# Configure model path
tour, cost, _ = neural_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "neural": {
            "model_path": "assets/model_weights/best_model.pt",
            "decode_type": "beam_search",
            "beam_width": 10
        }
    }
)
```

### 7.3 Supported Models

| Model        | Architecture      | Best For                          |
| ------------ | ----------------- | --------------------------------- |
| **AM**       | Attention Model   | General VRP, fast inference       |
| **DDAM**     | Deep Decoder AM   | Large-scale problems (200+ nodes) |
| **TAM**      | Temporal AM       | Multi-day scenarios, WCVRP        |
| **TransGCN** | Transformer + GCN | Graph-structured problems         |

---

## 8. Operators Library

**Directory**: `operators/`
**Purpose**: 26 specialized operators for solution manipulation

### 8.1 Operator Categories

1. **Move Operators** (`move/`): Single-node relocations
2. **Route Operators** (`route/`): Intra-route improvements
3. **Destroy Operators** (`destroy/`): Solution deconstruction (ALNS)
4. **Repair Operators** (`repair/`): Solution reconstruction (ALNS)
5. **Perturbation Operators** (`perturbation/`): Diversification
6. **Exchange Operators** (`exchange/`): Multi-node swaps

### 8.1 Move Operators

**Directory**: `operators/move/`

#### Relocate

**File**: `relocate.py`

Move a single node from one route to another.

```python
from logic.src.policies.operators import move_relocate

# Relocate node 5 from route 0 position 2 to route 1 position 1
new_routes, delta_cost = move_relocate(
    routes=[[0, 3, 5, 7, 0], [0, 2, 4, 0]],
    distance_matrix=dist_matrix,
    from_route=0,
    from_pos=2,
    to_route=1,
    to_pos=1
)
# Result: [[0, 3, 7, 0], [0, 2, 5, 4, 0]]
```

#### Swap

**File**: `swap.py`

Exchange two nodes between or within routes.

```python
from logic.src.policies.operators import move_swap

# Swap node at route 0 position 1 with node at route 1 position 1
new_routes, delta_cost = move_swap(
    routes=[[0, 3, 5, 0], [0, 2, 4, 0]],
    distance_matrix=dist_matrix,
    route1=0,
    pos1=1,
    route2=1,
    pos2=1
)
# Result: [[0, 2, 5, 0], [0, 3, 4, 0]]
```

### 8.2 Route Operators

**Directory**: `operators/route/`

#### 2-Opt Intra-Route

**File**: `two_opt_intra.py`

Reverse a segment within a single route.

```python
from logic.src.policies.operators import move_2opt_intra

# 2-opt on route 0 between positions 1 and 3
new_routes, delta_cost = move_2opt_intra(
    routes=[[0, 3, 5, 7, 9, 0]],
    distance_matrix=dist_matrix,
    route_idx=0,
    i=1,
    j=3
)
# [0, 3, 5, 7, 9, 0] → [0, 3, 7, 5, 9, 0] (reversed segment 5-7)
```

#### 2-Opt\* (Inter-Route)

**File**: `two_opt_star.py`

Exchange tails between two routes.

```python
from logic.src.policies.operators import move_2opt_star

# Exchange tails after position 1 in both routes
new_routes, delta_cost = move_2opt_star(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    distance_matrix=dist_matrix,
    route1=0,
    route2=1,
    pos1=1,
    pos2=1
)
# Result: [[0, 1, 5, 6, 0], [0, 4, 2, 3, 0]]
```

#### 3-Opt Intra-Route

**File**: `three_opt_intra.py`

Advanced intra-route optimization with 3 edges.

```python
from logic.src.policies.operators import move_3opt_intra

new_routes, delta_cost = move_3opt_intra(
    routes=[[0, 1, 2, 3, 4, 5, 0]],
    distance_matrix=dist_matrix,
    route_idx=0,
    i=1,
    j=3,
    k=5
)
```

### 8.3 Destroy Operators (ALNS)

**Directory**: `operators/destroy/`

#### Random Removal

**File**: `random.py`

Remove random nodes from routes.

```python
from logic.src.policies.operators import random_removal

# Remove 5 random nodes
removed_nodes, modified_routes = random_removal(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    num_remove=5,
    rng=np.random.default_rng(42)
)
```

#### Worst Removal

**File**: `worst.py`

Remove nodes with highest cost contribution.

```python
from logic.src.policies.operators import worst_removal

# Remove 3 worst nodes
removed_nodes, modified_routes = worst_removal(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    distance_matrix=dist_matrix,
    num_remove=3,
    randomness=0.1  # Add some randomness
)
```

#### Cluster Removal

**File**: `cluster.py`

Remove spatially clustered nodes.

```python
from logic.src.policies.operators import cluster_removal

# Remove a cluster of 4 nodes
removed_nodes, modified_routes = cluster_removal(
    routes=[[0, 1, 2, 3, 4, 5, 0]],
    distance_matrix=dist_matrix,
    num_remove=4,
    rng=np.random.default_rng(42)
)
```

#### Shaw Removal

**File**: `shaw.py`

Remove similar nodes (by distance, demand, time).

```python
from logic.src.policies.operators import shaw_removal

# Remove 4 similar nodes
removed_nodes, modified_routes = shaw_removal(
    routes=[[0, 1, 2, 3, 4, 5, 0]],
    distance_matrix=dist_matrix,
    demands=demands_dict,
    num_remove=4,
    relatedness_weights=(0.5, 0.3, 0.2)  # distance, demand, time
)
```

#### String Removal (SISR)

**File**: `string.py`

Remove consecutive subsequences.

```python
from logic.src.policies.operators import string_removal

# Remove strings of length 2-4
removed_nodes, modified_routes = string_removal(
    routes=[[0, 1, 2, 3, 4, 5, 0]],
    min_string_len=2,
    max_string_len=4,
    num_strings=2
)
```

### 8.4 Repair Operators (ALNS)

**Directory**: `operators/repair/`

#### Greedy Insertion

**File**: `greedy.py`

Insert nodes at best positions.

```python
from logic.src.policies.operators import greedy_insertion

# Insert removed nodes greedily
new_routes = greedy_insertion(
    routes=[[0, 1, 0], [0, 3, 0]],
    unrouted_nodes=[2, 4, 5],
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0
)
```

#### Regret-k Insertion

**File**: `regret.py`

Insert nodes based on regret value (difference between best and k-th best position).

```python
from logic.src.policies.operators import regret_k_insertion

# Regret-2 insertion
new_routes = regret_k_insertion(
    routes=[[0, 1, 0]],
    unrouted_nodes=[2, 3, 4],
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0,
    k=2  # Regret-2
)
```

#### Greedy Blink Insertion (SISR)

**File**: `greedy_blink.py`

Greedy insertion with random "blinks" for exploration.

```python
from logic.src.policies.operators import greedy_insertion_with_blinks

# Insert with 1% blink probability
new_routes = greedy_insertion_with_blinks(
    routes=[[0, 1, 0]],
    unrouted_nodes=[2, 3, 4],
    distance_matrix=dist_matrix,
    demands=demands_dict,
    capacity=200.0,
    blink_probability=0.01
)
```

### 8.6 Exchange Operators

**Directory**: `operators/exchange/`

#### Or-Opt

**File**: `or_opt.py`

Move a chain of 1-3 consecutive nodes.

```python
from logic.src.policies.operators import move_or_opt

# Move chain of length 2
new_routes, delta_cost = move_or_opt(
    routes=[[0, 1, 2, 3, 4, 0]],
    distance_matrix=dist_matrix,
    route_idx=0,
    start_pos=1,
    chain_length=2,
    insert_pos=4
)
# [0, 1, 2, 3, 4, 0] → [0, 3, 4, 1, 2, 0]
```

#### Cross-Exchange

**File**: `cross.py`

Exchange segments between two routes.

```python
from logic.src.policies.operators import cross_exchange

# Exchange segments
new_routes, delta_cost = cross_exchange(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    distance_matrix=dist_matrix,
    route1=0,
    route2=1,
    start1=1,
    len1=2,
    start2=1,
    len2=2
)
```

#### Ejection Chain

**File**: `ejection.py`

Complex multi-route reconfiguration.

```python
from logic.src.policies.operators import ejection_chain

# Perform ejection chain
new_routes, delta_cost = ejection_chain(
    routes=[[0, 1, 2, 0], [0, 3, 4, 0], [0, 5, 6, 0]],
    distance_matrix=dist_matrix,
    chain_length=3
)
```

#### λ-Interchange

**File**: `lambda_interchange.py`

Exchange up to λ nodes between routes.

```python
from logic.src.policies.operators import lambda_interchange

# (1,1)-interchange
new_routes, delta_cost = lambda_interchange(
    routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]],
    distance_matrix=dist_matrix,
    lambda1=1,
    lambda2=1
)
```

---

## 9. Must-Go Selection Strategies

**Directory**: `other/must_go/`
**Purpose**: Pre-selection of mandatory collection nodes

### 9.1 Selection Framework

#### SelectionContext

**File**: `base/selection_context.py`

Context object for selection decisions.

```python
from logic.src.policies.other import SelectionContext

context = SelectionContext(
    bins=bins_state,
    distance_matrix=dist_matrix,
    area="riomaior",
    waste_type="plastic",
    day=5,
    history=historical_data
)
```

#### MustGoSelectionFactory

**File**: `base/selection_factory.py`

Factory for creating selectors.

```python
from logic.src.policies.other import MustGoSelectionFactory

# Create selector
selector = MustGoSelectionFactory.create_selector(
    strategy="last_minute",
    threshold=0.9
)

# Execute selection
must_go_bins = selector.select(context)
```

### 9.2 Selection Strategies

#### Last-Minute Selector

**File**: `selection_last_minute.py`

Select bins exceeding fill threshold.

```python
from logic.src.policies.other.must_go import LastMinuteSelector

selector = LastMinuteSelector(threshold=0.9)

# Select bins with fill ≥ 90%
must_go = selector.select(context)
```

#### Regular Selector

**File**: `selection_regular.py`

Select bins on fixed-frequency schedule.

```python
from logic.src.policies.other.must_go import RegularSelector

# Collect every 3 days
selector = RegularSelector(frequency=3)

must_go = selector.select(context)
```

#### Lookahead Selector

**File**: `selection_lookahead.py`

Select bins predicted to overflow within N days.

```python
from logic.src.policies.other.must_go import LookaheadSelector

# Predict 7 days ahead
selector = LookaheadSelector(
    lookahead_days=7,
    predictor=fill_predictor_model
)

must_go = selector.select(context)
```

#### Revenue Selector

**File**: `selection_revenue.py`

Select bins where profit exceeds cost.

```python
from logic.src.policies.other.must_go import RevenueSelector

selector = RevenueSelector(
    revenue_per_kg=1.0,
    cost_per_km=0.5
)

must_go = selector.select(context)
```

#### Service-Level Selector

**File**: `selection_service_level.py`

Statistical overflow prediction.

```python
from logic.src.policies.other.must_go import ServiceLevelSelector

# 95% service level
selector = ServiceLevelSelector(
    service_level=0.95,
    fill_std_dev=0.1
)

must_go = selector.select(context)
```

#### Combined Selector

**File**: `selection_combined.py`

Combine multiple strategies with AND/OR logic.

```python
from logic.src.policies.other.must_go import CombinedSelector

selector = CombinedSelector(
    logic="or",  # or "and"
    strategies=[
        {"strategy": "last_minute", "threshold": 0.9},
        {"strategy": "regular", "frequency": 7}
    ]
)

# Select if last_minute OR regular triggers
must_go = selector.select(context)
```

### 9.3 Usage in Simulation

```python
from logic.src.policies import create_policy
from logic.src.policies.other import MustGoSelectionFactory

# Create policy
policy = create_policy("hgs")

# Create must-go selector
selector = MustGoSelectionFactory.create_selector(
    strategy="combined",
    logic="or",
    strategies=[
        {"strategy": "last_minute", "threshold": 0.85},
        {"strategy": "regular", "frequency": 5}
    ]
)

# Execute selection
must_go = selector.select(context)

# Execute policy
tour, cost, _ = policy.execute(
    must_go=must_go,
    bins=context.bins,
    distance_matrix=context.distance_matrix
)
```

---

## 10. Post-Processing & Refinement

**Directory**: `other/post_processing/`
**Purpose**: Automatic route improvement after initial solution

### 10.1 PostProcessorFactory

**File**: `factory.py`

```python
from logic.src.policies.other import PostProcessorFactory

# Create post-processor
post_processor = PostProcessorFactory.create_processor("fast_tsp")

# Apply to tour
improved_tour = post_processor.process(
    tour=[0, 5, 3, 8, 2, 0],
    distance_matrix=dist_matrix
)
```

### 10.2 Post-Processing Methods

#### Fast TSP

**File**: `fast_tsp.py`

Fast TSP heuristic using `fast_tsp` library.

```python
from logic.src.policies.other.post_processing import FastTSPPostProcessor

processor = FastTSPPostProcessor()

# Improve single-route tour
improved_tour = processor.process(
    tour=[0, 5, 3, 8, 2, 7, 0],
    distance_matrix=dist_matrix
)
```

#### Classical Local Search

**File**: `local_search.py`

2-opt and 3-opt intra-route improvements.

```python
from logic.src.policies.other.post_processing import ClassicalLocalSearchPostProcessor

processor = ClassicalLocalSearchPostProcessor(
    operators=["2opt", "3opt"],
    max_iterations=100
)

improved_tour = processor.process(
    tour=[0, 1, 2, 3, 4, 5, 0],
    distance_matrix=dist_matrix
)
```

#### Randomized Local Search

**File**: `random_ls.py`

Randomized operator selection.

```python
from logic.src.policies.other.post_processing import RandomLocalSearchPostProcessor

processor = RandomLocalSearchPostProcessor(
    operator_probs={
        "2opt": 0.4,
        "swap": 0.3,
        "relocate": 0.3
    },
    max_iterations=50
)

improved_tour = processor.process(
    tour=[0, 1, 2, 3, 0],
    distance_matrix=dist_matrix
)
```

#### Iterated Local Search

**File**: `ils.py`

ILS with perturbation for escaping local minima.

```python
from logic.src.policies.other.post_processing import IteratedLocalSearchPostProcessor

processor = IteratedLocalSearchPostProcessor(
    n_restarts=5,
    ls_iterations=50,
    perturbation_strength=0.2
)

improved_tour = processor.process(
    tour=[0, 1, 2, 3, 4, 5, 0],
    distance_matrix=dist_matrix
)
```

#### Path Improvement

**File**: `path.py`

Specialized path optimization.

```python
from logic.src.policies.other.post_processing import PathPostProcessor

processor = PathPostProcessor()

improved_tour = processor.process(
    tour=[0, 1, 2, 3, 0],
    distance_matrix=dist_matrix
)
```

### 10.3 Pipeline Configuration

```python
from logic.src.policies.other import PostProcessorFactory

# Multi-stage refinement pipeline
processors = [
    PostProcessorFactory.create_processor("fast_tsp"),
    PostProcessorFactory.create_processor("classical_ls"),
    PostProcessorFactory.create_processor("ils")
]

tour = initial_tour
for processor in processors:
    tour = processor.process(tour, distance_matrix)

print(f"Final tour cost: {compute_cost(tour, distance_matrix)}")
```

---

## 11. Local Search

**Directory**: `local_search/`
**Purpose**: Base classes for local search implementations

### 11.1 LocalSearch Base

**File**: `local_search_base.py`

```python
from logic.src.policies.local_search import LocalSearch

class CustomLocalSearch(LocalSearch):
    """Custom local search implementation."""

    def search(self, solution, distance_matrix, **kwargs):
        """Perform local search."""
        improved = True
        while improved:
            improved = False
            for operator in self.operators:
                new_solution = operator.apply(solution)
                if self.is_better(new_solution, solution):
                    solution = new_solution
                    improved = True
        return solution
```

### 11.2 HGS Local Search

**File**: `local_search_hgs.py`

```python
from logic.src.policies.local_search import HGSLocalSearch

ls = HGSLocalSearch(
    operators=["relocate", "swap", "2opt", "2opt_star"],
    max_iterations=100
)

improved_routes = ls.search(
    routes=[[0, 1, 2, 0], [0, 3, 4, 0]],
    distance_matrix=dist_matrix
)
```

### 11.3 ACO Local Search

**File**: `local_search_aco.py`

```python
from logic.src.policies.local_search import ACOLocalSearch

ls = ACOLocalSearch(
    operators=["2opt", "3opt"],
    max_iterations=50
)

improved_routes = ls.search(
    routes=[[0, 1, 2, 3, 0]],
    distance_matrix=dist_matrix
)
```

---

## 12. Integration Examples

### 12.1 Complete Simulation Pipeline

```python
from logic.src.policies import create_policy
from logic.src.policies.other import (
    MustGoSelectionFactory,
    PostProcessorFactory,
    SelectionContext
)

# 1. Create must-go selector
selector = MustGoSelectionFactory.create_selector(
    strategy="combined",
    logic="or",
    strategies=[
        {"strategy": "last_minute", "threshold": 0.9},
        {"strategy": "regular", "frequency": 3}
    ]
)

# 2. Create policy
policy = create_policy("hgs")

# 3. Create post-processor
post_processor = PostProcessorFactory.create_processor("fast_tsp")

# 4. Execute simulation day
context = SelectionContext(
    bins=bins_state,
    distance_matrix=dist_matrix,
    area="riomaior",
    waste_type="plastic",
    day=5
)

# 5. Select must-go bins
must_go = selector.select(context)

# 6. Generate tour with policy
tour, cost, _ = policy.execute(
    must_go=must_go,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config=config_dict
)

# 7. Post-process tour
improved_tour = post_processor.process(tour, dist_matrix)
final_cost = compute_cost(improved_tour, dist_matrix)

print(f"Initial cost: {cost:.2f}")
print(f"Final cost: {final_cost:.2f}")
print(f"Improvement: {(cost - final_cost) / cost * 100:.1f}%")
```

### 12.2 Multi-Policy Comparison

```python
from logic.src.policies import create_policy

policies = ["hgs", "alns", "bcp", "neural"]
results = {}

for policy_name in policies:
    policy = create_policy(policy_name)

    tour, cost, metadata = policy.execute(
        must_go=must_go_bins,
        bins=bins_state,
        distance_matrix=dist_matrix,
        config=config
    )

    results[policy_name] = {
        "tour": tour,
        "cost": cost,
        "metadata": metadata
    }

# Find best
best_policy = min(results, key=lambda p: results[p]["cost"])
print(f"Best policy: {best_policy} with cost {results[best_policy]['cost']:.2f}")
```

### 12.3 Custom Policy Implementation

```python
from logic.src.policies.adapters import BaseRoutingPolicy, PolicyRegistry

@PolicyRegistry.register("my_custom_policy")
class MyCustomPolicy(BaseRoutingPolicy):
    """Custom routing policy implementation."""

    def _get_config_key(self) -> str:
        return "my_custom"

    def _run_solver(
        self,
        sub_dist_matrix,
        sub_demands,
        capacity,
        revenue,
        cost_unit,
        values,
        **kwargs
    ):
        """Implement custom solver logic."""
        # 1. Initialize solution
        routes = self._initialize_routes(sub_dist_matrix, sub_demands, capacity)

        # 2. Improve with local search
        for iteration in range(values.get("max_iterations", 100)):
            routes = self._improve_routes(routes, sub_dist_matrix)

        # 3. Compute cost
        cost = self._compute_routes_cost(routes, sub_dist_matrix)

        return routes, cost

    def _initialize_routes(self, dist_matrix, demands, capacity):
        """Create initial solution."""
        # Custom initialization logic
        return [[1, 2], [3, 4]]

    def _improve_routes(self, routes, dist_matrix):
        """Apply improvement operators."""
        # Custom improvement logic
        return routes

    def _compute_routes_cost(self, routes, dist_matrix):
        """Calculate total cost."""
        # Custom cost calculation
        return 0.0

# Usage
policy = create_policy("my_custom_policy")
tour, cost, _ = policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix
)
```

### 12.4 Operator Composition

```python
from logic.src.policies.operators import (
    random_removal,
    worst_removal,
    greedy_insertion,
    regret_k_insertion
)

def custom_alns_iteration(routes, dist_matrix, demands, capacity):
    """Custom ALNS iteration with specific operators."""
    # Destroy phase: alternate between random and worst removal
    if iteration % 2 == 0:
        removed, routes = random_removal(routes, num_remove=5)
    else:
        removed, routes = worst_removal(routes, dist_matrix, num_remove=5)

    # Repair phase: alternate between greedy and regret
    if len(removed) < 3:
        routes = greedy_insertion(routes, removed, dist_matrix, demands, capacity)
    else:
        routes = regret_k_insertion(routes, removed, dist_matrix, demands, capacity, k=2)

    return routes

# Use in optimization loop
routes = initial_routes
for iteration in range(100):
    new_routes = custom_alns_iteration(routes, dist_matrix, demands, capacity)
    if compute_cost(new_routes) < compute_cost(routes):
        routes = new_routes
```

---

## 13. Best Practices

### 13.1 Recommended Practices

**Use PolicyFactory for Instantiation**

```python
# ✅ GOOD: Dynamic policy creation
policy = PolicyFactory.get_adapter("hgs")

# ❌ BAD: Direct import and instantiation
from logic.src.policies.adapters.policy_hgs import HGSPolicy
policy = HGSPolicy()
```

**Register Custom Policies**

```python
# ✅ GOOD: Register for discoverability
@PolicyRegistry.register("my_policy")
class MyPolicy(IPolicyAdapter):
    ...

# ❌ BAD: Unregistered policies
class MyPolicy(IPolicyAdapter):
    ...
```

**Use Must-Go Selectors**

```python
# ✅ GOOD: Configurable selection strategy
selector = MustGoSelectionFactory.create_selector("last_minute", threshold=0.9)
must_go = selector.select(context)

# ❌ BAD: Hardcoded selection logic
must_go = [i for i, fill in enumerate(bins.c) if fill > 0.9]
```

**Apply Post-Processing**

```python
# ✅ GOOD: Refine solutions
tour, cost, _ = policy.execute(...)
improved_tour = post_processor.process(tour, dist_matrix)

# ❌ BAD: Use raw policy output
tour, cost, _ = policy.execute(...)
# No improvement applied
```

**Handle Engine Availability**

```python
# ✅ GOOD: Graceful fallback
try:
    policy = PolicyFactory.get_adapter("bcp")
    tour, cost, _ = policy.execute(..., config={"bcp": {"engine": "gurobi"}})
except ImportError:
    print("Gurobi not available, falling back to OR-Tools")
    tour, cost, _ = policy.execute(..., config={"bcp": {"engine": "ortools"}})

# ❌ BAD: Assume engine exists
policy = PolicyFactory.get_adapter("bcp")
tour, cost, _ = policy.execute(..., config={"bcp": {"engine": "gurobi"}})
# Crashes if Gurobi not installed
```

### 13.2 Anti-Patterns

**Don't Modify Operators In-Place**

```python
# ❌ BAD: Modifying input routes
def bad_operator(routes):
    routes[0].append(5)  # Mutates input!
    return routes

# ✅ GOOD: Return new routes
def good_operator(routes):
    new_routes = [route.copy() for route in routes]
    new_routes[0].append(5)
    return new_routes
```

**Don't Skip Validation**

```python
# ❌ BAD: No input validation
def solve(routes, dist_matrix):
    # Assumes routes is valid
    return optimize(routes)

# ✅ GOOD: Validate inputs
def solve(routes, dist_matrix):
    if not routes:
        return [0, 0], 0.0
    if len(dist_matrix) == 0:
        raise ValueError("Empty distance matrix")
    return optimize(routes)
```

**Don't Ignore Capacity Constraints**

```python
# ❌ BAD: Insert without capacity check
def insert_node(route, node, demand):
    route.insert(1, node)

# ✅ GOOD: Check capacity
def insert_node(route, node, demand, capacity, demands):
    current_load = sum(demands[n] for n in route if n != 0)
    if current_load + demand <= capacity:
        route.insert(1, node)
        return True
    return False
```

### 13.3 Performance Optimization

**Batch Processing**

```python
# ✅ GOOD: Batch process for neural policies
from logic.src.policies import NeuralAgent

agent = NeuralAgent(model)
tours, costs = agent.solve_batch(
    batch_coords=batch_coords,
    batch_demands=batch_demands
)

# ❌ BAD: Loop over instances
tours, costs = [], []
for coords, demands in zip(batch_coords, batch_demands):
    tour, cost = agent.solve(coords, demands)
    tours.append(tour)
    costs.append(cost)
```

**Operator Selection**

```python
# ✅ GOOD: Use fast operators for large problems
if num_nodes > 200:
    operators = ["relocate", "swap", "2opt"]  # O(n²)
else:
    operators = ["relocate", "swap", "2opt", "3opt", "or_opt"]  # Include O(n³)

# ❌ BAD: Always use expensive operators
operators = ["3opt", "ejection_chain"]  # Very slow on large instances
```

**Caching**

```python
# ✅ GOOD: Cache distance calculations
@lru_cache(maxsize=10000)
def get_distance(i, j, dist_matrix):
    return dist_matrix[i][j]

# ❌ BAD: Repeated lookups
for i in range(n):
    for j in range(n):
        d = dist_matrix[i][j]  # No caching
```

---

## 14. Quick Reference

### 14.1 Common Imports

```python
# Policy creation
from logic.src.policies.adapters import PolicyFactory, PolicyRegistry

# Main policies
from logic.src.policies import (
    run_alns, run_hgs, run_bcp, run_sisr,
    run_k_sparse_aco, run_hyper_heuristic_aco,
    NeuralAgent
)

# Operators
from logic.src.policies.operators import (
    # Move
    move_relocate, move_swap,
    # Route
    move_2opt_intra, move_2opt_star, move_3opt_intra,
    # Destroy
    random_removal, worst_removal, cluster_removal,
    # Repair
    greedy_insertion, regret_k_insertion
)

# Must-go selection
from logic.src.policies.other import (
    MustGoSelectionFactory, MustGoSelectionRegistry,
    SelectionContext
)

# Post-processing
from logic.src.policies.other import (
    PostProcessorFactory, PostProcessorRegistry
)

# TSP/CVRP
from logic.src.policies import find_route, find_routes
```

### 14.2 Policy Summary

| Policy       | Type          | Best For                             | Typical Runtime |
| ------------ | ------------- | ------------------------------------ | --------------- |
| **BCP**      | Exact         | Small instances (<50 nodes), optimal | Minutes-Hours   |
| **HGS**      | Metaheuristic | Medium-large instances, high quality | Seconds-Minutes |
| **ALNS**     | Metaheuristic | Diverse instances, robustness        | Seconds-Minutes |
| **HGS-ALNS** | Hybrid        | Large instances, best quality        | Minutes         |
| **ACO**      | Bio-inspired  | Sparse graphs, parallel execution    | Minutes         |
| **SANS**     | SA-based      | Complex constraints                  | Seconds-Minutes |
| **SISR**     | Specialized   | String patterns in routes            | Seconds         |
| **Neural**   | Deep RL       | Fast inference, generalization       | Milliseconds    |
| **LKH**      | TSP heuristic | TSP instances, excellent quality     | Seconds         |
| **VRPP**     | Exact         | Profit-oriented VRP                  | Minutes         |

### 14.3 Operator Complexity

| Operator         | Time Complexity | Space Complexity | Use Case                     |
| ---------------- | --------------- | ---------------- | ---------------------------- |
| Relocate         | O(n²)           | O(1)             | General improvement          |
| Swap             | O(n²)           | O(1)             | General improvement          |
| 2-Opt            | O(n²)           | O(1)             | Intra-route improvement      |
| 3-Opt            | O(n³)           | O(1)             | Deep intra-route improvement |
| 2-Opt\*          | O(n²)           | O(1)             | Inter-route improvement      |
| Or-Opt           | O(n²)           | O(1)             | Chain moves                  |
| Cross-Exchange   | O(n⁴)           | O(1)             | Complex inter-route          |
| Ejection Chain   | O(n^k)          | O(k)             | Deep exploration             |
| Random Removal   | O(n)            | O(n)             | ALNS destroy                 |
| Worst Removal    | O(n²)           | O(n)             | ALNS destroy (quality)       |
| Greedy Insertion | O(n²)           | O(n)             | ALNS repair                  |
| Regret-k         | O(kn²)          | O(kn)            | ALNS repair (quality)        |

### 14.4 Configuration Examples

**ALNS Configuration**

```yaml
alns:
  time_limit: 60.0
  max_iterations: 5000
  start_temp: 100.0
  cooling_rate: 0.995
  reaction_factor: 0.1
  min_removal: 1
  max_removal_pct: 0.3
  engine: custom
```

**HGS Configuration**

```yaml
hgs:
  time_limit: 120.0
  population_size: 50
  elite_size: 10
  mutation_rate: 0.2
  crossover_type: ox
  local_search_iterations: 100
```

**Neural Configuration**

```yaml
neural:
  model_path: assets/model_weights/am_wcvrp_100.pt
  decode_type: beam_search
  beam_width: 10
  temperature: 1.0
  device: cuda
```

### 14.5 File Locations

| Component         | Location                                      |
| ----------------- | --------------------------------------------- |
| Policy Adapters   | `adapters/policy_*.py` (13 files)             |
| Operators         | `operators/` (26 implementations)             |
| Must-Go Selectors | `other/must_go/selection_*.py` (6 strategies) |
| Post-Processors   | `other/post_processing/*.py` (5 methods)      |
| ALNS              | `adaptive_large_neighborhood_search/`         |
| HGS               | `hybrid_genetic_search/`                      |
| ACO               | `ant_colony_optimization/`                    |
| SANS              | `simulated_annealing_neighborhood_search/`    |
| SISR              | `slack_induction_by_string_removal/`          |
| BCP               | `branch_cut_and_price/`                       |
| Neural            | `neural_agent/`                               |

### 14.6 Related Documentation

- [CONFIGS_MODULE.md](CONFIGS_MODULE.md) - Policy configuration dataclasses
- [MODELS_MODULE.md](MODELS_MODULE.md) - Neural network architectures
- [ENVS_MODULE.md](ENVS_MODULE.md) - Problem environments
- [CLAUDE.md](../CLAUDE.md) - Coding standards and guidelines

---

**Last Updated**: January 2026
**Maintainer**: WSmart+ Route Development Team
**Status**: ✅ Active - Comprehensive routing policies library
