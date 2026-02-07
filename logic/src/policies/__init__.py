"""
Policies package for WSmart-Route.

This package contains all routing policies (classical, heuristic, and neural)
used for solving the Waste Collection Vehicle Routing Problem.
"""

from .adapters.policy_vrpp import run_vrpp_optimizer
from .adaptive_large_neighborhood_search import (
    ALNSParams,
    run_alns,
    run_alns_ortools,
    run_alns_package,
)
from .branch_cut_and_price import run_bcp
from .hybrid_genetic_search import run_hgs
from .k_sparse_aco import ACOParams, run_aco
from .look_ahead_aux.common.routes import create_points
from .look_ahead_aux.refinement.route_search import find_solutions
from .multi_vehicle import find_routes, find_routes_ortools
from .neural_agent import NeuralAgent
from .single_vehicle import find_route, get_route_cost
from .slack_induction_by_string_removal import run_sisr

__all__ = [
    "ACOParams",
    "ALNSParams",
    "PolicyFactory",
    "NeuralAgent",
    "create_points",
    "create_policy",
    "find_route",
    "find_routes",
    "find_routes_ortools",
    "find_solutions",
    "get_route_cost",
    "run_alns",
    "run_alns_ortools",
    "run_alns_package",
    "run_aco",
    "run_bcp",
    "run_hgs",
    "run_sisr",
    "run_vrpp_optimizer",
]

from .adapters.factory import (
    PolicyFactory,
)

create_policy = PolicyFactory.get_adapter
