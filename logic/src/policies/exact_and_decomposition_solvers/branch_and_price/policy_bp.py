"""
Branch-and-Price (BP) Policy Adapter.

Adapts the core Branch-and-Price solver logic to the systems-agnostic policy
interface, handling parameter mapping, profit calculation, and environment
integration.

References:
    Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W. P.,
    & Vance, P. H. (1998). "Branch-and-price: Column Generation for Solving
    Huge Integer Programs". Operations Research, 46(3), 316-329.

    Baldacci, R., Mingozzi, A., & Roberti, R. (2011). "New Route Relaxation
    and Pricing Strategies for the Vehicle Routing Problem". Operations
    Research, 59(5), 1269-1283.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BPConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .bp import BranchAndPriceSolver
from .params import BPParams


@PolicyRegistry.register("bp")
class BranchAndPricePolicy(BaseRoutingPolicy):
    """
    Adapter for the Branch-and-Price routing solver.

    Implements a column generation method that handles large-scale optimisation
    by implicitly enumerating exponentially many variables (routes).

    The algorithm uses:

    1. Set Covering master problem — selects routes to cover all mandatory
       nodes while maximising profit.
    2. Pricing subproblem — generates profitable routes via RCSPP (exact DP
       or greedy heuristic).
    3. Column generation — iteratively adds routes with positive reduced cost.
    4. Branch-and-bound — optionally enforces integrality via edge branching
       or Ryan-Foster branching.

    Configuration
    -------------
    ``branching_strategy``  (``"edge"`` | ``"ryan_foster"``, default ``"edge"``)
        Selects the B&B branching scheme.

    ``use_exact_pricing``  (bool, default False)
        When True, uses the DP-based ``RCSPPSolver``; otherwise uses the
        greedy heuristic ``PricingSubproblem``.

    ``use_ng_routes``  (bool, default True)
        When ``use_exact_pricing=True``, enables the ng-route relaxation of
        Baldacci et al. (2011).  Dramatically reduces the label count on large
        instances while still producing near-elementary routes.  Set False to
        restore exact ESPPRC behaviour.

    ``ng_neighborhood_size``  (int, default 8)
        Size of each node's ng-neighborhood N_i (including the node itself).
        Larger values tighten the relaxation (approaching exact ESPPRC) at the
        cost of more labels per solve.
    """

    def __init__(self, config: Optional[Union[BPConfig, Dict[str, Any]]] = None) -> None:
        """
        Initialise the BP policy adapter.

        Args:
            config: A typed BPConfig dataclass, a raw dictionary for parsing,
                or None to use framework defaults.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration dataclass type for automatic parsing."""
        return BPConfig

    def _get_config_key(self) -> str:
        """Return the unique identification key for this policy's configuration."""
        return "bp"

    def execute(self, **kwargs: Any) -> Tuple[Union[List[int], List[List[int]]], float, Any]:
        """
        Execute Exact Branch-and-Price.
        If multi_day_mode is True, evaluates for a single stage of the multi-period
        problem without standard BaseRoutingPolicy subset extraction.
        """
        config_dict = kwargs.get("config", {}).get(self._get_config_key(), {})
        multi_day_mode = config_dict.get("multi_day_mode", False)

        if not multi_day_mode:
            # Standard single-day mode
            return super().execute(**kwargs)

        # Multi-day mode (Exact SDP inner deterministic optimizer)
        dist_matrix = kwargs["model_ls"][1]
        bins = kwargs["bins"]
        profit_vars = kwargs["model_ls"][2]

        R = profit_vars.get("revenue_kg", 1.0)
        C = profit_vars.get("cost_km", 1.0)
        num_vehicles = profit_vars.get("n_vehicles", 0)

        wastes = {i: float(bins.c[i - 1]) for i in range(1, len(bins.c) + 1)}
        capacity = float(profit_vars.get("bin_capacity", 100.0))
        must_go = kwargs.get("must_go", [])

        params = BPParams.from_config(config_dict)

        solver = BranchAndPriceSolver(
            n_nodes=len(dist_matrix) - 1,
            cost_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            revenue_per_kg=R,
            cost_per_km=C,
            mandatory_nodes=set(must_go),
            params=params,
            vehicle_limit=num_vehicles,
        )

        # solver.solve() returns the correctly delimited flat tour: [0, 1, 3, 0, 2, 0]
        flat_tour, _ip_obj, _stats = solver.solve()

        # Preserve the exact structure provided by the solver
        global_route = flat_tour if flat_tour else [0, 0]

        # Explicitly check for None before invoking methods on Optional[Any]
        model_env = kwargs.get("model_env")
        cost = model_env.compute_route_cost(global_route) if model_env is not None else 0.0

        return global_route, cost, {"policy_type": "bp", "multi_day_mode": True}

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run the Branch-and-Price solver.

        All nodes in ``mandatory_nodes`` are treated as must-visit.  In VRPP
        mode, additional nodes from ``sub_wastes`` may be collected if their
        inclusion is profitable.

        Parameter resolution
        --------------------
        Each parameter is looked up in ``values`` first (runtime override),
        then in ``self._config`` (YAML default), then falls back to a hard
        default.  This three-tier priority applies to all solver parameters,
        including the new ng-route parameters.

        ng-Route relaxation
        -------------------
        When ``use_exact_pricing=True``, the DP pricer can use the ng-route
        relaxation (Baldacci et al. 2011).  The two controlling parameters are:

        ``use_ng_routes`` (bool, default True)
            Enables or disables the ng-route state-space relaxation.
            Set to False to restore exact ESPPRC.

        ``ng_neighborhood_size`` (int, default 8)
            Size of each node's ng-neighborhood.  Higher values → tighter
            relaxation, more labels, longer solve time.

        Branching strategy
        ------------------
        ``branching_strategy`` (str) controls edge vs Ryan-Foster branching.
        The legacy ``use_ryan_foster_branching`` bool is still honoured as a
        fallback for backward compatibility.

        Args:
            sub_dist_matrix: Localised distance matrix for candidate nodes
                (shape (n+1) × (n+1), index 0 = depot).
            sub_wastes: Current fill levels for available customer nodes.
            capacity: Maximum vehicle payload.
            revenue: Expected revenue per unit of waste collected.
            cost_unit: Cost per unit of distance travelled.
            values: Merged configuration dictionary from simulation and YAML.
            mandatory_nodes: Indices of nodes that MUST be visited.
            **kwargs: Additional solver parameters (forwarded transparently).

        Returns:
            Tuple of ``(routes, total_profit, total_travel_cost)`` where
            ``routes`` is a list of node sequences (depot excluded) for each
            selected route.
        """
        n_nodes = len(sub_dist_matrix) - 1
        mandatory_set = set(mandatory_nodes)

        # ------------------------------------------------------------------
        # Build and run solver with standardized parameters
        # ------------------------------------------------------------------
        params = BPParams.from_config(values)

        solver = BranchAndPriceSolver(
            n_nodes=n_nodes,
            cost_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            revenue_per_kg=revenue,
            cost_per_km=cost_unit,
            mandatory_nodes=mandatory_set,
            params=params,
        )

        tour, profit, statistics = solver.solve()

        # ------------------------------------------------------------------
        # Convert flat tour to per-vehicle route lists (depot stripped)
        # ------------------------------------------------------------------
        if tour and len(tour) > 2:
            route = [node for node in tour if node != 0]
            routes = [route] if route else []
        else:
            routes = []

        # ------------------------------------------------------------------
        # Calculate actual distance cost
        # ------------------------------------------------------------------
        dist_cost = 0.0
        if tour:
            prev = 0
            for node in tour:
                if node != 0:
                    dist_cost += sub_dist_matrix[prev, node]
                    prev = node
                elif prev != 0:
                    dist_cost += sub_dist_matrix[prev, 0]
                    prev = 0
            dist_cost *= cost_unit

        return routes, profit, dist_cost
