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
    - DifferentialEvolutionSolver: Differential Evolution (replaces Artificial Bee Colony)
    - HybridMemeticSearchSolver: Hybrid Memetic Search (replaces HMS/HVPL)
    - MemeticAlgorithmIslandModelSolver: Memetic Algorithm with Island Model (replaces MA-IM/SLC)
    - MemeticAlgorithmToleranceBasedSelectionSolver: Memetic Algorithm with Tolerance-Based Selection (replaces MA-TB/LCA)
    - MemeticAlgorithmDualPopulationSolver: Memetic Algorithm with Dual Population (replaces MA-DP/VPL)
    - ParticleSwarmOptimizationSolver: Canonical PSO (replaces Sine Cosine Algorithm)

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
from .base import IPolicy, PolicyFactory, PolicyRegistry
from .capacitated_vehicle_routing_problem.cvrp import find_routes, find_routes_ortools
from .differential_evolution.params import DEParams
from .differential_evolution.solver import DESolver
from .evolution_strategy_mu_comma_lambda.solver import MuCommaLambdaESParams, MuCommaLambdaESSolver
from .evolution_strategy_mu_plus_lambda.solver import MuPlusLambdaESParams, MuPlusLambdaESSolver
from .guided_indicators_hyper_heuristic.policy_gihh import run_gihh
from .hybrid_genetic_search.policy_hgs import run_hgs
from .hybrid_genetic_search_ruin_and_recreate.policy_hgs_rr import run_hgs_rr
from .hybrid_memetic_search.params import HybridMemeticSearchParams
from .hybrid_memetic_search.solver import HybridMemeticSearchSolver
from .memetic_algorithm_dual_population.params import MemeticAlgorithmDualPopulationParams
from .memetic_algorithm_dual_population.solver import MemeticAlgorithmDualPopulationSolver
from .memetic_algorithm_island_model.params import MemeticAlgorithmIslandModelParams
from .memetic_algorithm_island_model.solver import MemeticAlgorithmIslandModelSolver
from .memetic_algorithm_tolerance_based_selection.params import MemeticAlgorithmToleranceBasedSelectionParams
from .memetic_algorithm_tolerance_based_selection.solver import MemeticAlgorithmToleranceBasedSelectionSolver
from .neural_agent.policy_neural import NeuralAgent
from .particle_swarm_optimization_distance_based_algorithm.solver import DistancePSOParams, DistancePSOSolver
from .travelling_salesman_problem.tsp import find_route

__all__ = [
    "ALNSParams",
    "run_alns",
    "IPolicy",
    "PolicyFactory",
    "PolicyRegistry",
    "find_routes",
    "find_routes_ortools",
    "run_gihh",
    "run_hgs",
    "run_hgs_rr",
    "NeuralAgent",
    "find_route",
    "MuPlusLambdaESSolver",
    "MuPlusLambdaESParams",
    "DistancePSOSolver",
    "DistancePSOParams",
    "DESolver",
    "DEParams",
    "MuCommaLambdaESSolver",
    "MuCommaLambdaESParams",
    "HybridMemeticSearchSolver",
    "HybridMemeticSearchParams",
    "MemeticAlgorithmIslandModelSolver",
    "MemeticAlgorithmIslandModelParams",
    "MemeticAlgorithmToleranceBasedSelectionSolver",
    "MemeticAlgorithmToleranceBasedSelectionParams",
    "MemeticAlgorithmDualPopulationSolver",
    "MemeticAlgorithmDualPopulationParams",
]
