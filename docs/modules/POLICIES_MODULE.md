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
   - [Rigorous Meta-Heuristic Implementations](#rigorous-meta-heuristic-implementations)
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
| **SCWCVRP** | Stochastic Capacitated WCVRP        | Neural, Adaptive Heuristics   |

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
        sub_wastes: Dict[int, float],
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
    wastes=wastes_dict,
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
    wastes=wastes_dict,
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
    wastes=wastes_dict,
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
    wastes=wastes_dict,
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

### 5.5 RL-HVPL (Reinforcement Learning Hybrid Volleyball Premier League)

**Directory**: `reinforcement_learning_hybrid_volleyball_premier_league/`
**Adapters**: `policy_rl_hvpl.py`

#### Overview

The **Reinforcement Learning Hybrid Volleyball Premier League (RL-HVPL)** is a metaheuristic algorithm that combines:

1. **Enhanced ACO with Q-Learning** for intelligent solution construction
2. **Enhanced ALNS with SARSA** for adaptive destroy/repair operator selection
3. **Population-based VPL framework** for global search and diversity management

This implementation bridges the gap between:

- **HVPL** (basic ACO + ALNS in population framework)
- **RL-AHVPL** (advanced with HGS genetic operators, CMAB crossover selection, GLS, reactive tabu)

RL-HVPL provides the benefits of reinforcement learning-enhanced operators **without** the complexity of genetic evolution.

#### Architecture

##### Core Components

```text
RL-HVPL
├── ACO with Q-Learning (Construction)
│   ├── K-Sparse pheromone matrix
│   ├── Probabilistic ant construction
│   └── Q-Learning for local search operator selection
│
├── ALNS with SARSA (Coaching/Improvement)
│   ├── 9 Destroy operators (random, worst, cluster, shaw, string, unstring I-IV)
│   ├── 9 Repair operators (greedy, regret-k, blink, string I-IV)
│   ├── Perturbation operators (kick, random)
│   └── SARSA for adaptive operator selection
│
└── VPL Framework (Population Management)
    ├── Population (teams) of candidate solutions
    ├── Coaching phase (ALNS-SARSA improvement)
    ├── Competition (best solutions survive)
    ├── Substitution (weak solutions replaced)
    └── Pheromone updates (global guidance)
```

##### Algorithm Flow

```text
1. INITIALIZATION
   ├── Create population using ACO-Q (n_teams solutions)
   └── Track best solution globally

2. MAIN LOOP (max_iterations)
   ├── COACHING PHASE
   │   ├── Elite teams → Intensive ALNS-SARSA (elite_coaching_iterations)
   │   └── Regular teams → Light ALNS-SARSA (regular_coaching_iterations)
   │
   ├── COMPETITION
   │   └── Update global best if improved
   │
   ├── PHEROMONE UPDATE
   │   ├── Evaporate existing pheromones
   │   └── Deposit on best solution edges (profit-based or cost-based)
   │
   └── SUBSTITUTION
       └── Replace weakest teams with new ACO-Q solutions

3. RETURN best solution found
```

#### Key Features

##### 1. Reinforcement Learning Integration

- **Q-Learning in ACO**: Dynamically selects the most effective local search operators during construction
- **SARSA in ALNS**: Adaptively chooses destroy/repair operator pairs based on search state
- **State Representation**: Progress, stagnation, and diversity features discretized into states
- **Adaptive Exploration**: Epsilon-greedy with decay for exploration-exploitation balance

##### 2. Pheromone Update Strategies

**Profit-based (default):**

```python
delta = elitist_weight * profit_weight * profit / cost
```

- Reinforces high profit-to-cost ratio solutions
- Better for profit-maximization problems (VRPP, WCVRP)

**Cost-based (classic ACS):**

```python
delta = elitist_weight / cost
```

- Reinforces low-cost solutions
- Better for pure distance minimization (TSP, CVRP)

##### 3. Adaptive Coaching

Elite teams (top `elite_size`) receive **intensive coaching** with more ALNS-SARSA iterations, while regular teams get **lighter coaching** to balance computational resources.

#### Parameters

##### Core Parameters

| Parameter                     | Default | Description                               |
| ----------------------------- | ------- | ----------------------------------------- |
| `n_teams`                     | 10      | Population size                           |
| `max_iterations`              | 100     | Number of league seasons                  |
| `time_limit`                  | 60.0    | Time budget (seconds)                     |
| `sub_rate`                    | 0.2     | Fraction of teams replaced each iteration |
| `elite_size`                  | 3       | Number of teams receiving elite coaching  |
| `elite_coaching_iterations`   | 300     | ALNS iterations for elite teams           |
| `regular_coaching_iterations` | 100     | ALNS iterations for regular teams         |

##### Pheromone Strategy

| Parameter                   | Default    | Description                     |
| --------------------------- | ---------- | ------------------------------- |
| `pheromone_update_strategy` | `"profit"` | `"profit"` or `"cost"`          |
| `profit_weight`             | 1.0        | Weight for profit-based updates |

##### ACO Parameters

Configured via `aco_params` (ACOParams):

- `n_ants`: Number of ants per iteration (default: 10)
- `k_sparse`: Sparse pheromone neighbors (default: 10)
- `alpha`: Pheromone influence (default: 1.0)
- `beta`: Heuristic influence (default: 2.0)
- `rho`: Evaporation rate (default: 0.1)
- `q0`: Exploitation probability (default: 0.9)
- `local_search`: Enable local search (default: True)
- `local_search_iterations`: Local search budget (default: 50)

##### ALNS Parameters

Configured via `alns_params` (ALNSParams):

- `max_iterations`: ALNS iterations (default: 200)
- `start_temp`: Initial temperature (default: 100.0)
- `cooling_rate`: Temperature decay (default: 0.97)
- `max_removal_pct`: Max fraction of nodes to remove (default: 0.3)
- `perturb_k`: Perturbation strength (default: 3)

##### RL Parameters

Configured via `rl_config` (RLConfig):

**Q-Learning (ACO):**

- `alpha`: Learning rate (default: 0.1)
- `gamma`: Discount factor (default: 0.9)
- `epsilon`: Exploration rate (default: 0.2)
- `epsilon_decay`: Decay multiplier (default: 0.995)
- `epsilon_min`: Minimum epsilon (default: 0.01)

**SARSA (ALNS):**

- `alpha`: Learning rate (default: 0.1)
- `gamma`: Discount factor (default: 0.9)
- `epsilon`: Exploration rate (default: 0.3)
- `epsilon_decay`: Decay multiplier (default: 0.99)
- `epsilon_min`: Minimum epsilon (default: 0.05)

#### Usage

##### Basic Usage

```python
import numpy as np
from logic.src.policies.reinforcement_learning_hybrid_volleyball_premier_league import (
    RLHVPLSolver,
    RLHVPLParams,
)

# Problem instance
dist_matrix = np.array([[...]])  # Distance matrix (n+1 x n+1, depot at 0)
wastes = {1: 10.0, 2: 15.0, ...}  # Node waste amounts
capacity = 100.0  # Vehicle capacity
R = 10.0  # Revenue per unit waste
C = 1.0   # Cost per unit distance

# Configure solver
params = RLHVPLParams(
    n_teams=10,
    max_iterations=100,
    time_limit=60.0,
)

# Solve
solver = RLHVPLSolver(
    dist_matrix=dist_matrix,
    wastes=wastes,
    capacity=capacity,
    R=R,
    C=C,
    params=params,
    seed=42,
)

routes, profit, cost = solver.solve()

print(f"Profit: {profit:.2f}")
print(f"Cost: {cost:.2f}")
print(f"Routes: {routes}")
```

##### Advanced Configuration

```python
from logic.src.configs.policies.other import RLConfig
from logic.src.policies.ant_colony_optimization.k_sparse_aco.params import ACOParams
from logic.src.policies.adaptive_large_neighborhood_search.params import ALNSParams

# Custom RL configuration
rl_config = RLConfig()
rl_config.td_learning.alpha = 0.15
rl_config.td_learning.gamma = 0.95
rl_config.td_learning.epsilon = 0.25

# Custom ACO parameters
aco_params = ACOParams(
    n_ants=15,
    k_sparse=15,
    alpha=1.2,
    beta=2.5,
    rho=0.15,
    local_search_iterations=100,
)

# Custom ALNS parameters
alns_params = ALNSParams(
    max_iterations=300,
    start_temp=150.0,
    cooling_rate=0.98,
    max_removal_pct=0.4,
)

# Create solver with custom parameters
params = RLHVPLParams(
    n_teams=15,
    max_iterations=150,
    elite_size=5,
    elite_coaching_iterations=500,
    regular_coaching_iterations=150,
    pheromone_update_strategy="profit",
    profit_weight=1.5,
    rl_config=rl_config,
    aco_params=aco_params,
    alns_params=alns_params,
)

solver = RLHVPLSolver(dist_matrix, wastes, capacity, R, C, params, seed=42)
routes, profit, cost = solver.solve()
```

##### Adapter Usage

```python
from logic.src.policies.adapters import PolicyFactory

rl_hvpl_policy = PolicyFactory.get_adapter("rl_hvpl")
tour, cost, _ = rl_hvpl_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "rl_hvpl": {
            "n_teams": 10,
            "max_iterations": 100,
            "time_limit": 60.0,
            "elite_size": 3,
            "pheromone_update_strategy": "profit"
        }
    }
)
```

#### Performance Characteristics

##### Computational Complexity

- **Per iteration**: O(n_teams × (ACO_cost + ALNS_cost))
- **ACO_cost**: O(n_ants × n² × k_sparse)
- **ALNS_cost**: O(max_iterations × n²)

##### Memory Usage

- Pheromone matrix: O(n × k_sparse)
- Population: O(n_teams × n)
- RL Q-tables: O(n_states × n_actions)

##### Scalability

| Problem Size          | Recommended Settings           |
| --------------------- | ------------------------------ |
| Small (n < 50)        | n_teams=5, max_iterations=50   |
| Medium (50 ≤ n < 100) | n_teams=10, max_iterations=100 |
| Large (100 ≤ n < 200) | n_teams=15, max_iterations=150 |
| Very Large (n ≥ 200)  | n_teams=20, max_iterations=200 |

#### References

1. **Volleyball Premier League Algorithm**
   - Population-based metaheuristic framework

2. **Ant Colony Optimization**
   - Dorigo & Gambardella, "Ant Colony System", IEEE Trans., 1997

3. **Adaptive Large Neighborhood Search**
   - Pisinger & Ropke, "A general heuristic for VRP", Computers & OR, 2007

4. **Q-Learning**
   - Watkins & Dayan, "Q-Learning", Machine Learning, 1992

5. **SARSA**
   - Sutton & Barto, "Reinforcement Learning: An Introduction", 2018

#### File Structure

```text
reinforcement_learning_hybrid_volleyball_premier_league/
├── __init__.py          # Module exports
├── params.py            # RLHVPLParams configuration
└── rl_hvpl.py           # RLHVPLSolver main implementation
```

### 5.6 FILO (Fast Iterative Localized Optimization)

**Directory**: `fast_iterative_localized_optimization/`
**Adapters**: `policy_filo.py`

Fast Iterative Localized Optimization (FILO) is a scalable metaheuristic built specifically to solve large Capacitated Vehicle Routing Problems (CVRP). It introduces dynamic parameters to selectively evaluate the neighborhood space.

#### Architecture

```text
FILO
├── Ruin & Recreate (Shaking)
│   └── Node extraction proportional to `omega` intensity bounds
├── Fast Local Search
│   └── Sparse exploration restricted by `gamma` activation rates
└── Simulated Annealing
    └── Acceptance criterion with dynamic cooling
```

#### Key Features

- **Granular Shaking Intensity (`omega`)**: Each node dynamically calibrates its own degree of extraction during Ruin & Recreate, bounding the scope based on the current objective value and moving average route costs.
- **Node-Level Activation Probability (`gamma`)**: To radically reduce iteration times, FILO tracks consecutive non-improving evaluations for each node and applies an activation probability (gamma) that drops the likelihood of computing unpromising neighbors.
- **Dynamic Tuning**: At runtime, `shaking_lb` and `shaking_ub` intervals recalibrate themselves whenever a new global best is found.

#### Usage Example

```python
from logic.src.policies.adapters import PolicyFactory

filo_policy = PolicyFactory.get_adapter("filo")
tour, cost, _ = filo_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "filo": {
            "time_limit": 60.0,
            "max_iterations": 50000,
            "initial_temperature_factor": 10.0,
            "delta_gamma": 0.1
        }
    }
)
```

#### Performance Characteristics

Because FILO uses extreme sparsification, it is capable of performing tens of thousands of iterations in highly competitive time frames.

- **Computational Complexity**: O(n_nodes) per iteration dynamically masked via `gamma`.
- **Memory Usage**: O(N) where N is node dimension, meaning it completely out-scales full ant colony and path-relinking structures.

#### References

1. Accorsi & Vigo. "A Fast and Scalable Heuristic for the Solution of Large-Scale Capacitated Vehicle Routing Problems", _Transportation Science_, 2021.

### 5.7 HILS (Hybrid Iterated Local Search)

**Directory**: `hybrid_iterated_local_search/`
**Adapters**: `policy_hils.py`

Hybrid Iterated Local Search (HILS) combines the Iterated Local Search (ILS) metaheuristic with an exact algorithm approach, effectively marrying heuristics with Set Partitioning.

#### Architecture

```text
HILS
├── ILS Phase
│   ├── Perturbation (random node extraction & greedy insertion)
│   └── Randomized Variable Neighborhood Descent (RVND) Local Search
├── Route Pool
│   └── Global tracking of all active unique route structures
└── Set Partitioning (SP) Phase
    └── Gurobi MIP model resolving optimal combinations over the route pool
```

#### Key Features

- **Randomized Variable Neighborhood Descent**: Local search operators (e.g., Relocate, Swap, 2-Opt) are dynamically shuffled and applied until full local optimums are found, preventing static cyclic traps.
- **Route Pool Synergies**: Employs Iterated Local Search solely as a mass-route generator. The final solution is resolved purely mathematically through Set Partitioning, ensuring optimal exploitation of the explored sub-spaces.
- **Exact-Heuristic Hybridity**: Takes advantage of Gurobi's commercial performance to find absolute guarantees over sub-domains identified by agile heuristic techniques.

#### Usage Example

```python
from logic.src.policies.adapters import PolicyFactory

hils_policy = PolicyFactory.get_adapter("hils")
tour, cost, _ = hils_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    config={
        "hils": {
            "max_iterations": 100,
            "ils_iterations": 50,
            "perturbation_size": 2,
            "use_set_partitioning": True,
            "sp_time_limit": 60.0,
            "time_limit": 120.0
        }
    }
)
```

#### References

1. Subramanian et al. "A hybrid algorithm for a class of vehicle routing problems", _Computers & Operations Research_, 2013.

### 5.8 KGLS (Knowledge-Guided Local Search)

**Directory**: `knowledge_guided_local_search/`
**Adapters**: `policy_kgls.py`

Knowledge-Guided Local Search (KGLS) utilizes problem-specific domain logic—particularly spatial and geometric routing constraints—to perturb solutions dynamically, rather than purely mathematically or randomly tearing sub-graphs.

#### Architecture

```text
KGLS
├── Initial Construct
├── Local Search Descent
│   └── (Unpenalized baseline descent on valid configurations)
└── KGLS Execution Loop
    ├── Perturbation (Enable geometric evaluation matrices)
    │   ├── Evaluate edge badness (Width, Length, or both)
    │   ├── Inflate targeted true edge costs via Penalty Counters
    │   └── Trigger strict Fast Local Search radiating out of penalized targets
    └── Repair (Disable geometric matrices for absolute baseline descent)
```

#### Key Features

- **Geometric Perturbation**: Evaluates continuous local sub-routes via width properties computed dynamically against the main depot, extracting connections that weave haphazardly globally.
- **Inflated Cost Matrices**: Edge connections identified as geometrically suboptimal receive additive penalties (a fraction of network baseline scaled by their penalty count). Normal heuristic operators immediately snap/break these false edges simply by falling down the penalized gradient.
- **Cycle Criteria**: KGLS systematically rotates its penalty scoring mechanisms (`width`, `length`, `width_length`), preventing repetitive exploitation traps.

#### Usage Example

```python
from logic.src.policies.adapters import PolicyFactory

# Requires positional arguments via the environment to compute widths
kgls_policy = PolicyFactory.get_adapter("kgls")
tour, cost, _ = kgls_policy.execute(
    must_go=must_go_bins,
    bins=bins_state,
    distance_matrix=dist_matrix,
    data_nodes={"depot": [0,0], "locs": [...]},
    config={
        "kgls": {
            "time_limit": 60.0,
            "num_perturbations": 3,
            "neighborhood_size": 20,
            "penalization_cycle": ["width", "length", "width_length"]
        }
    }
)
```

#### References

1. Arnold & Sörensen. "What makes a VRP solution good? The generation of problem-specific knowledge for heuristics", _Computers & Operations Research_, 2019.

---

### 5.9 Rigorous Meta-Heuristic Implementations

This section describes the mathematically rigorous implementations that replace metaphor-based algorithms in the WSmart+ Route codebase.

#### Overview

The "metaphor controversy" in optimization research refers to algorithms that obscure standard mathematical operations with biological, physical, or social metaphors. We have replaced these with canonical implementations using proper Operations Research terminology.

---

#### Implementation Map

| Metaphor-Based Algorithm    | Rigorous Implementation                 | Mathematical Foundation                                    |
| --------------------------- | --------------------------------------- | ---------------------------------------------------------- |
| Harmony Search (HS)         | **(μ+λ) Evolution Strategy [with λ=1]** | Population-based search with recombination and mutation    |
| Firefly Algorithm (FA)      | **Distance-Based PSO**                  | Particle swarm with exponential distance decay             |
| Artificial Bee Colony (ABC) | **(μ,λ) Evolution Strategy**            | Multi-phase random search with restart mechanism           |
| HVPL, SLC (Sports)          | **Island Model Genetic Algorithm**      | Multi-population GA with migration and local search        |
| League Championship (LCA)   | **Stochastic Tournament GA**            | Pairwise tournament selection with sigmoid probability     |
| SCA (Sine Cosine)           | **Continuous Local Search**             | Gradient-free search with trigonometric perturbations      |
| **(μ,κ,λ) ES**              | **Age-Based Evolution Strategy**        | Metaheuristic with age-based selection and self-adaptation |

---

#### Detailed Implementations

##### 1. (μ+λ) Evolution Strategy

**Replaces:** Harmony Search (HS)

**Location:** `logic/src/policies/evolution_strategy_mu_plus_lambda/`

**Algorithm:**

```
1. Initialize population of μ solutions
2. For each iteration:
   a. Generate λ offspring:
      i. Select parent via fitness-proportional selection
      ii. Create offspring via recombination (using archive solutions)
      iii. Apply mutation operator (local perturbation)
   b. Combine parents (μ) and offspring (λ) into population of (μ+λ)
   c. Select best μ individuals to survive (elitist selection)
```

**Key Parameters:**

- `population_size` (μ): Number of parent solutions
- `offspring_size` (λ): Number of offspring generated per iteration
- `recombination_rate`: Probability of using archive (was "HMCR")
- `mutation_rate`: Probability of local mutation (was "PAR")

**Terminology Mapping:**

- "Harmony Memory" → Population/Archive
- "Improvisation" → Offspring generation
- "HMCR" → Recombination rate
- "Pitch Adjustment" → Mutation operator

**Complexity:**

- Time: O(T × λ × (n + n²)) where T = iterations
- Space: O((μ + λ) × n)

**Reference:**

> Rechenberg, I. (1973). "Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution."

---

##### 2. Distance-Based Particle Swarm Optimization

**Replaces:** Firefly Algorithm (FA)

**Location:** `logic/src/policies/particle_swarm_optimization_distance/`

**Algorithm:**

```
1. Initialize swarm of particles (solutions)
2. For each iteration:
   a. For each particle pair (i, j where fitness[j] > fitness[i]):
      - Compute Hamming distance d between solutions
      - With probability β(d) = β₀ × exp(-γ × d²), move particle i toward j
   b. With probability α, apply random walk exploration
   c. Update global best solution
```

**Key Parameters:**

- `initial_attraction` (β₀): Global best attraction coefficient
- `distance_decay` (γ): Exponential decay for distance-based attraction
- `exploration_rate` (α): Random walk probability

**Terminology Mapping:**

- "Fireflies" → Particles
- "Light intensity" → Objective function value (fitness)
- "Attractiveness" → Distance-weighted attraction weight
- "Random walk" → Exploration operator

**Mathematical Foundation:**

- Attraction weight: β(d) = β₀ × exp(-γ × d²)
- Hamming distance: d = |edges(A) ⊕ edges(B)|

**Complexity:**

- Time: O(T × N² × n²) where N = population size
- Space: O(N × n)

**Reference:**

> Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization." Proceedings of ICNN'95.

---

##### 3. (μ,λ) Evolution Strategy

**Replaces:** Artificial Bee Colony (ABC)

**Location:** `logic/src/policies/evolution_strategy_mu_comma_lambda/`

**Algorithm:**

```
1. Initialize population of μ solutions
2. For each iteration:
   a. Local Search Phase: Each parent generates offspring via mutation
   b. Selection Phase: Select offspring via fitness-proportional selection
   c. Update Phase: Replace parents with selected offspring (non-elitist)
   d. Restart Phase: Replace stagnant solutions with random restarts
```

**Key Parameters:**

- `population_size` (μ): Number of parent solutions
- `offspring_per_parent`: Offspring generation rate (λ/μ)
- `stagnation_limit`: Random restart threshold

**Terminology Mapping:**

- "Employed bees" → Local search agents (exploitation)
- "Onlooker bees" → Probabilistic selection mechanism
- "Scout bees" → Random restart mechanism
- "Food sources" → Parent solutions

**Complexity:**

- Time: O(T × λ × n²) where λ = offspring count
- Space: O(μ × n) + O(λ × n)

**Reference:**

> Schwefel, H.-P. (1981). "Numerical Optimization of Computer Models." John Wiley & Sons.

---

##### 4. Island Model Genetic Algorithm

**Replaces:** HVPL, LCA, SLC (Sports-Based Algorithms)

**Location:** `logic/src/policies/island_model_genetic_algorithm/`

**Algorithm:**

```
1. Initialize K islands with N individuals each
2. For each generation:
   a. Local Improvement: Apply ALNS to each individual in each island
   b. Fitness Evaluation: Compute objective values
   c. Tournament Selection: Replace weak individuals via k-tournament
   d. Migration: Periodically exchange best solutions between islands
   e. Global Update: Track global best across all islands
```

**Key Parameters:**

- `n_islands` (K): Number of sub-populations
- `island_size` (N): Population size per island
- `tournament_size`: Selection pressure
- `migration_interval`: Generations between migrations

**Terminology Mapping:**

- "Teams" → Islands (sub-populations)
- "Matches/Competition" → Fitness evaluations
- "Seasons" → Generations
- "Coaching" → Local search operator (ALNS)
- "Relegation/Promotion" → Tournament selection
- "League tables" → Fitness rankings

**Migration Topology:**

- Ring topology: Island i sends to island (i+1) mod K

**Complexity:**

- Time: O(T × K × N × ALNS_cost)
- Space: O(K × N × n)

**Reference:**

> Whitley, D., Rana, S., & Heckendorn, R. B. (1998). "The island model genetic algorithm."

---

##### 5. Stochastic Tournament Genetic Algorithm

**Replaces:** League Championship Algorithm (LCA)

**Location:** `logic/src/policies/stochastic_tournament_genetic_algorithm/`

**Algorithm:**

```
1. Initialize population of N chromosomes
2. For each generation:
   a. Fitness evaluation for all individuals
   b. Stochastic tournament selection:
      - Each individual competes against k random opponents
      - Win probability: P(i > j) = σ(β × (f_i - f_j))
      - Select winners for mating pool
   c. Crossover (recombination) of selected parents
   d. Mutation of offspring
   e. Elitist replacement (keep best individuals)
```

**Key Parameters:**

- `population_size` (N): Number of chromosomes
- `tournament_competitors` (k): Opponents per individual
- `selection_pressure` (β): Sigmoid coefficient for win probability
- `crossover_rate`: Probability of recombination
- `mutation_rate`: Probability of perturbation
- `elitism_count`: Top individuals preserved unchanged

**Terminology Mapping:**

- "League Schedule/Fixtures" → Pairwise fitness evaluation cycles
- "Playing Strength" → Objective function value (fitness)
- "Match Outcome (Win/Loss)" → Stochastic tournament result
- "Team Formation" → Crossover/recombination operator

**Mathematical Foundation:**

- **Stochastic Tournament Selection:** P(i defeats j) = σ(β × (f(i) - f(j)))
- **Sigmoid Function:** σ(x) = 1/(1 + exp(-x))
- **Selection Pressure:** Higher β → more deterministic selection

**Complexity:**

- Time: O(T × N × k × eval_cost) where T = generations
- Space: O(N × n)
- Selection: O(N × k) for tournament comparisons

**Reference:**

> Goldberg, D. E., & Deb, K. (1991). "A comparative analysis of selection schemes used in genetic algorithms." Foundations of Genetic Algorithms.

---

##### 6. Continuous Local Search

**Replaces:** Sine Cosine Algorithm (SCA)

**Location:** `logic/src/policies/continuous_local_search/`

**Algorithm:**

```
1. Initialize population in continuous space [-1, 1]^n
2. For each iteration:
   a. Update step size α = α_max × (1 - t/T) (linear decay)
   b. For each solution vector:
      - Compute perturbation direction toward global best
      - Apply trigonometric step: sin(θ) or cos(θ)
      - Update: x' = x + α × trig(θ) × |β × x_best - x|
      - Decode to discrete solution and evaluate fitness
   c. Track global best solution
```

**Key Parameters:**

- `max_step_size` (α_max): Initial perturbation step size
- `population_size`: Number of continuous solution vectors

**Terminology Mapping:**

- "Position vectors" → Continuous solution encoding
- "Destination point" → Best solution (global attractor)
- "Sine/Cosine update" → Directional perturbation operators
- "Parameter a" → Adaptive step size

**Mathematical Foundation:**

- Position update: x'[i] = x[i] + r₁ × sin(r₂) × |r₃ × x_best[i] - x[i]|
- or: x'[i] = x[i] + r₁ × cos(r₂) × |r₃ × x_best[i] - x[i]|
- where r₁ ∈ [0, α], r₂ ∈ [0, 2π], r₃ ∈ [0, 2], α decays linearly

**Decoding Strategy:**

1. Sigmoid binarization: b[j] = 1 if σ(x[j]) > 0.5
2. Largest Rank Value (LRV) ordering
3. Greedy insertion for route construction

**Complexity:**

- Time: O(T × N × n²)
- Space: O(N × n)

**Reference:**

> Mirjalili, S. (2016). "SCA: A Sine Cosine Algorithm for solving optimization problems." (Mathematical interpretation without metaphor)

---

##### 7. (μ,κ,λ) Evolution Strategy

**Location:** `logic/src/policies/evolution_strategy_mu_kappa_lambda/`

**Algorithm:**

The (μ,κ,λ)-ES implements a classical metaheuristic for continuous optimization with **age-based selection**. Selection occurs from μ parents who have not exceeded an age of κ and λ offspring individuals.

```text
1. Initialize P₀ with μ parent individuals
2. For each generation t:
    a. Recombine(Pₜ₋₁) → create λ offspring
    b. Mutate(offspring) → apply self-adaptive mutation
    c. Evaluate(offspring) → compute fitness
    d. Select(offspring ∪ eligible_parents) → choose μ best
       where eligible_parents are those with age ≤ κ
    e. Increment age of all surviving individuals
    f. Update best solution if improved
3. Return best solution found
```

**Mathematical Formulation:**

**Step-size mutation:**

```text
N_global ~ N(0,1)  (shared across all dimensions)
σ'ᵢ ← σᵢ · exp(τ_global · N_global + τ_local · Nᵢ(0,1))
```

**Decision variable mutation:**

```text
x'ᵢ ← xᵢ + σ'ᵢ · N(0,1)
```

**Learning rates:**

```text
τ_local  = 1/√(2d)
τ_global = 1/(2√d)
```

**Key Parameters:**

- `mu` (μ): Number of parent individuals.
- `kappa` (κ): Maximum age for parents before they are discarded (controls population turnover).
- `lambda_` (λ): Number of offspring generated per generation.
- `rho` (ρ): Number of parents involved in recombination (1 for discrete, μ for intermediate).
- `initial_sigma`: Initial step size for mutation (recommended ~5% of search domain).

**Terminology Mapping:**

- "Eligible Parents" → Stay in the selection pool if age ≤ κ
- "Age" → Counter for how many generations an individual has survived
- "Self-Adaptation" → Step sizes (σ) evolve alongside decision variables (x)

**Complexity:**

- Time: O(T × λ × d) where T = iterations, λ = offspring, d = dimensions
- Space: O((μ + λ) × d) for population + offspring

**Reference:**

> Emmerich, M., Shir, O. M., & Wang, H. (2015). "Evolution Strategies." In: Handbook of Natural Computing, Springer.

---

#### Usage Examples

##### Example 1: (μ+λ) Evolution Strategy

```python
from logic.src.policies import MuPlusLambdaESSolver, MuPlusLambdaESParams

# Configure parameters
params = MuPlusLambdaESParams(
    population_size=10,        # μ parameter
    offspring_size=5,          # λ parameter
    recombination_rate=0.95,   # Archive recombination probability
    mutation_rate=0.3,         # Local mutation probability
    max_iterations=500,
    time_limit=60.0
)

# Initialize solver
solver = MuPlusLambdaESSolver(
    dist_matrix=distance_matrix,
    wastes=waste_dict,
    capacity=100.0,
    R=1.0,
    C=1.0,
    params=params,
    mandatory_nodes=[1, 5, 10],
    seed=42
)

# Execute optimization
best_routes, best_profit, best_cost = solver.solve()
```

##### Example 2: Distance-Based PSO

```python
from logic.src.policies import DistancePSOSolver, DistancePSOParams

params = DistancePSOParams(
    population_size=20,
    initial_attraction=1.0,    # β₀
    distance_decay=0.01,       # γ
    exploration_rate=0.1,      # α
    max_iterations=500
)

solver = DistancePSOSolver(
    dist_matrix=distance_matrix,
    wastes=waste_dict,
    capacity=100.0,
    R=1.0,
    C=1.0,
    params=params,
    seed=42
)

best_routes, best_profit, best_cost = solver.solve()
```

##### Example 3: Island Model GA

```python
from logic.src.policies import IslandModelGASolver, IslandModelGAParams

params = IslandModelGAParams(
    n_islands=10,              # K sub-populations
    island_size=10,            # N individuals per island
    max_generations=50,
    migration_interval=5,
    tournament_size=3
)

solver = IslandModelGASolver(
    dist_matrix=distance_matrix,
    wastes=waste_dict,
    capacity=100.0,
    R=1.0,
    C=1.0,
    params=params,
    seed=42
)

best_routes, best_profit, best_cost = solver.solve()
```

##### Example 4: (μ,κ,λ) Evolution Strategy

```python
import numpy as np
from logic.src.policies.evolution_strategy_mu_kappa_lambda import (
    MuKappaLambdaESSolver,
    MuKappaLambdaESParams
)

# Define objective function
def sphere(x):
    return np.sum(x**2)

# Configure parameters
params = MuKappaLambdaESParams(
    mu=15,              # Number of parents
    kappa=7,            # Maximum age
    lambda_=100,        # Number of offspring
    rho=2,              # Recombination size
    initial_sigma=1.0,  # Initial step size
    max_iterations=100,
    bounds_min=-5.0,
    bounds_max=5.0,
)

# Create solver
solver = MuKappaLambdaESSolver(
    objective_function=sphere,
    dimension=10,
    params=params,
    seed=42,
    minimize=True,
)

# Solve
best_x, best_fitness = solver.solve()
print(f"Best fitness: {best_fitness:.6e}")
```

---

#### Performance Characteristics

| Algorithm                | Time Complexity     | Space Complexity | Best Use Case                                   |
| ------------------------ | ------------------- | ---------------- | ----------------------------------------------- |
| (μ+λ)-ES                 | O(T × λ × n²)       | O((μ+λ) × n)     | Small-medium instances, fast convergence        |
| Distance-Based PSO       | O(T × N² × n²)      | O(N × n)         | Medium instances, global exploration            |
| (μ,λ)-ES                 | O(T × λ × n²)       | O((μ+λ) × n)     | Large instances, multi-phase search             |
| Island Model GA          | O(T × K × N × ALNS) | O(K × N × n)     | Large instances, parallel execution             |
| Stochastic Tournament GA | O(T × N × k × n²)   | O(N × n)         | Medium instances, controlled selection pressure |
| Continuous Local Search  | O(T × N × n²)       | O(N × n)         | Continuous relaxations, gradient-free           |
| (μ,κ,λ)-ES               | O(T × λ × d)        | O((μ+λ) × d)     | Age-based elitism control, self-adaptation      |

---

#### Deprecation Notice

The following metaphor-based implementations are now superseded by rigorous alternatives:

- ❌ `harmony_search/` → Use `evolution_strategy_mu_plus_lambda/`
- ❌ `firefly_algorithm/` → Use `particle_swarm_optimization_distance/`
- ❌ `artificial_bee_colony/` → Use `evolution_strategy_mu_comma_lambda/`
- ❌ `hybrid_volleyball_premier_league/` → Use `island_model_genetic_algorithm/`
- ❌ `league_championship_algorithm/` → Use `stochastic_tournament_genetic_algorithm/`
- ❌ `soccer_league_competition/` → Use `island_model_genetic_algorithm/`
- ❌ `sine_cosine_algorithm/` → Use `continuous_local_search/`

The original implementations remain for backward compatibility but should not be used for new development.

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
    wastes=wastes_dict,
    capacity=200.0,
    engine="auto"  # Auto-select: Gurobi > OR-Tools > VRPy
)

# Force specific engine
routes, cost = run_bcp(
    distance_matrix=dist_matrix,
    wastes=wastes_dict,
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
    wastes=wastes_dict,  # Profit per node
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
    wastes=wastes_dict,
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
    wastes=wastes,
    capacity=200.0
)

# Solve batch
tours, costs = agent.solve_batch(
    batch_coords=batch_coords,
    batch_wastes=batch_wastes,
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

Remove similar nodes (by distance, waste, time).

```python
from logic.src.policies.operators import shaw_removal

# Remove 4 similar nodes
removed_nodes, modified_routes = shaw_removal(
    routes=[[0, 1, 2, 3, 4, 5, 0]],
    distance_matrix=dist_matrix,
    wastes=wastes_dict,
    num_remove=4,
    relatedness_weights=(0.5, 0.3, 0.2)  # distance, waste, time
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
    wastes=wastes_dict,
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
    wastes=wastes_dict,
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
    wastes=wastes_dict,
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
        sub_wastes,
        capacity,
        revenue,
        cost_unit,
        values,
        **kwargs
    ):
        """Implement custom solver logic."""
        # 1. Initialize solution
        routes = self._initialize_routes(sub_dist_matrix, sub_wastes, capacity)

        # 2. Improve with local search
        for iteration in range(values.get("max_iterations", 100)):
            routes = self._improve_routes(routes, sub_dist_matrix)

        # 3. Compute cost
        cost = self._compute_routes_cost(routes, sub_dist_matrix)

        return routes, cost

    def _initialize_routes(self, dist_matrix, wastes, capacity):
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

def custom_alns_iteration(routes, dist_matrix, wastes, capacity):
    """Custom ALNS iteration with specific operators."""
    # Destroy phase: alternate between random and worst removal
    if iteration % 2 == 0:
        removed, routes = random_removal(routes, num_remove=5)
    else:
        removed, routes = worst_removal(routes, dist_matrix, num_remove=5)

    # Repair phase: alternate between greedy and regret
    if len(removed) < 3:
        routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
    else:
        routes = regret_k_insertion(routes, removed, dist_matrix, wastes, capacity, k=2)

    return routes

# Use in optimization loop
routes = initial_routes
for iteration in range(100):
    new_routes = custom_alns_iteration(routes, dist_matrix, wastes, capacity)
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
def insert_node(route, node, waste):
    route.insert(1, node)

# ✅ GOOD: Check capacity
def insert_node(route, node, waste, capacity, wastes):
    current_load = sum(wastes[n] for n in route if n != 0)
    if current_load + waste <= capacity:
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
    batch_wastes=batch_wastes
)

# ❌ BAD: Loop over instances
tours, costs = [], []
for coords, wastes in zip(batch_coords, batch_wastes):
    tour, cost = agent.solve(coords, wastes)
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
    run_alns, run_hgs, run_bcp,
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
