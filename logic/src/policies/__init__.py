"""
Policies package for WSmart-Route.

This package contains all routing policies (classical, heuristic, and neural)
used for solving the Waste Collection Vehicle Routing Problem.

Policy Categories:
------------------
1. Classical Heuristics:
   - Regular: Fixed-schedule periodic collection
   - LastMinute: Reactive threshold-based collection

2. Optimization-Based:
   - VRPP: Vehicle Routing Problem with Profits (Gurobi/Hexaly)
   - LookAhead: Rolling-horizon optimization with multiple solvers

3. Metaheuristics:
   - ALNS: Adaptive Large Neighborhood Search
   - HGS: Hybrid Genetic Search
   - BCP: Branch-Cut-and-Price (OR-Tools, VRPy, Gurobi)

4. Neural Policies:
   - Neural Agent: Deep reinforcement learning models
   - Attention Models: Transformer-based routing

5. Routing Utilities:
   - Single-vehicle: TSP solving and local search
   - Multi-vehicle: VRP solving with capacity constraints

Usage:
------
Policies are typically accessed via the PolicyFactory and PolicyAdapter
pattern for unified execution interface across simulators.
"""
from .regular import policy_regular
from .policy_vrpp import policy_vrpp
from .last_minute import policy_last_minute, policy_last_minute_and_path, policy_profit_reactive
from .look_ahead import (
    policy_lookahead, policy_lookahead_vrpp, policy_lookahead_sans,
    policy_lookahead_hgs, policy_lookahead_alns, policy_lookahead_bcp
)

from .look_ahead_aux import create_points, find_solutions
from .single_vehicle import (
    find_route, get_route_cost,
    get_multi_tour, local_search_2opt
)