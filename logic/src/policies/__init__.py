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
from .last_minute import policy_last_minute, policy_last_minute_and_path
from .look_ahead import (
    policy_lookahead,
    policy_lookahead_alns,
    policy_lookahead_bcp,
    policy_lookahead_hgs,
    policy_lookahead_sans,
    policy_lookahead_vrpp,
)
from .look_ahead_aux.routes import create_points
from .look_ahead_aux.solutions import find_solutions
from .multi_vehicle import find_routes, find_routes_ortools
from .neural_agent import NeuralAgent
from .policy_vrpp import policy_vrpp
from .regular import policy_regular
from .single_vehicle import find_route, get_route_cost, local_search_2opt
from .vrpp_optimizer import run_vrpp_optimizer

__all__ = [
    "ALNSParams",
    "LastMinutePolicyAdapter",
    "LookAheadPolicyAdapter",
    "NeuralAgent",
    "NeuralPolicyAdapter",
    "PolicyAdapter",
    "PolicyFactory",
    "RegularPolicyAdapter",
    "VRPPPolicyAdapter",
    "create_points",
    "create_policy",
    "find_route",
    "find_routes",
    "find_routes_ortools",
    "find_solutions",
    "get_route_cost",
    "local_search_2opt",
    "policy_last_minute",
    "policy_last_minute_and_path",
    "policy_lookahead",
    "policy_lookahead_alns",
    "policy_lookahead_bcp",
    "policy_lookahead_hgs",
    "policy_lookahead_sans",
    "policy_lookahead_vrpp",
    "policy_regular",
    "policy_vrpp",
    "run_alns",
    "run_alns_ortools",
    "run_alns_package",
    "run_bcp",
    "run_hgs",
    "run_vrpp_optimizer",
]

from .adapters import (
    LastMinutePolicyAdapter,
    LookAheadPolicyAdapter,
    NeuralPolicyAdapter,
    PolicyAdapter,
    PolicyFactory,
    RegularPolicyAdapter,
    VRPPPolicyAdapter,
)

# Alias for backward compatibility and testing
create_policy = PolicyFactory.get_adapter
