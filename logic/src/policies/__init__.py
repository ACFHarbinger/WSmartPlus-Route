"""
Policies package for WSmart-Route.

This package contains all routing policies (classical, heuristic, and neural)
used for solving the Waste Collection Vehicle Routing Problem.
"""

from .adaptive_large_neighborhood_search import (
    ALNSParams,
    run_alns,
    run_alns_ortools,
    run_alns_package,
)
from .branch_cut_and_price import run_bcp
from .hybrid_genetic_search import run_hgs
from .look_ahead_aux.routes import create_points
from .look_ahead_aux.solutions import find_solutions
from .multi_vehicle import find_routes, find_routes_ortools
from .neural_agent import NeuralAgent, NeuralPolicy
from .policy_lac import LACPolicy
from .policy_sans import SANSPolicy
from .policy_vrpp import VRPPPolicy, run_vrpp_optimizer
from .single_vehicle import find_route, get_route_cost, local_search_2opt

__all__ = [
    "ALNSParams",
    "PolicyFactory",
    "create_points",
    "create_policy",
    "find_route",
    "find_routes",
    "find_routes_ortools",
    "find_solutions",
    "get_route_cost",
    "local_search_2opt",
    "run_alns",
    "run_alns_ortools",
    "run_alns_package",
    "run_bcp",
    "run_hgs",
    "run_vrpp_optimizer",
]

from .adapters import (
    PolicyFactory,
)

# Backward compatibility aliases
NeuralPolicyAdapter = NeuralPolicy
VRPPPolicyAdapter = VRPPPolicy


# Alias for backward compatibility and testing
create_policy = PolicyFactory.get_adapter
