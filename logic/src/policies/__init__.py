"""
Policies Package.

This package contains all routing policies (classical, heuristic, and neural)
used for solving the Waste Collection Vehicle Routing Problem (WCVRP) and
its variants.

Rigorous Implementations (Metaphor-Free):
    The following algorithms replace metaphor-based implementations with
    canonical mathematical foundations:
    - MuPlusLambdaESSolver: (μ+λ) Evolution Strategy (replaces Harmony Search)
    - DistancePSOSolver: Distance-Based PSO (replaces Firefly Algorithm)
    - MuCommaLambdaESSolver: (μ,λ) Evolution Strategy (replaces Artificial Bee Colony)
    - MemeticIslandModelGASolver: Memetic Island Model GA (replaces HVPL)
    - PureIslandModelGASolver: Pure Island Model GA (replaces SLC)
    - StochasticTournamentGASolver: Stochastic Tournament GA (replaces LCA)
    - ContinuousLocalSearchSolver: Continuous Local Search (replaces Sine Cosine Algorithm)

Attributes:
    ALNSParams (class): Parameters for ALNS.
    NeuralAgent (class): Neural policy wrapper.
    run_alns (function): Runs ALNS algorithm.
    run_hgs (function): Runs HGS algorithm.
    find_routes (function): Solves CVRP using classical heuristics.
    find_route (function): Solves TSP.

Example:
    >>> from logic.src.policies import find_routes
    >>> routes = find_routes(distance_matrix, wastes, capacity)
"""

from .adaptive_large_neighborhood_search.params import ALNSParams
from .adaptive_large_neighborhood_search.policy_alns import run_alns
from .capacitated_vehicle_routing_problem.cvrp import find_routes, find_routes_ortools
from .guided_indicators_hyper_heuristic.policy_gihh import run_gihh
from .hybrid_genetic_search.policy_hgs import run_hgs
from .hybrid_genetic_search_ruin_and_recreate.policy_hgsrr import run_hgsrr
from .neural_agent.policy_neural import NeuralAgent
from .travelling_salesman_problem.tsp import find_route

__all__ = [
    "ALNSParams",
    "run_alns",
    "find_routes",
    "find_routes_ortools",
    "run_gihh",
    "run_hgs",
    "run_hgsrr",
    "NeuralAgent",
    "find_route",
]
