"""
LKH-3 Policy Adapter.

Adapts the Lin-Kernighan-Helsgaun 3 heuristic to the agnostic routing
policy interface used by the WSmart+ Route simulator.

Reference:
    Helsgaun, K. (2017). An extension of the LKH-TSP solver for constrained
    traveling salesman and vehicle routing problems.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.lkh3 import LKH3Config
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.lin_kernighan_helsgaun_three.lin_kernighan_helsgaun import (
    solve_lkh,
)


@PolicyRegistry.register("lkh3")
class LKH3Policy(BaseRoutingPolicy):
    """LKH-3 policy class.

    Solves the CVRP/TSP sub-problem via the Lin-Kernighan-Helsgaun 3
    iterated local-search heuristic with POPMUSIC candidate generation,
    k-opt moves (k = 2..5), and IP-based tour merging.
    """

    def __init__(self, config: Optional[Union[LKH3Config, Dict[str, Any]]] = None):
        """Initialize LKH-3 policy with optional config.

        Args:
            config: LKH3Config dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return LKH3Config

    def _get_config_key(self) -> str:
        """Return config key for LKH-3."""
        return "lkh3"

    def _tour_to_routes(self, tour: List[int]) -> List[List[int]]:
        """Split a flat closed tour at depot (0) into individual route lists.

        Args:
            tour: Flat tour with depot visits (e.g. ``[0, 3, 1, 0, 2, 4, 0]``).

        Returns:
            List of route lists, each without leading/trailing depot.
            Example: ``[[3, 1], [2, 4]]``.
        """
        routes: List[List[int]] = []
        current: List[int] = []
        for node in tour:
            if node == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(node)
        if current:
            routes.append(current)
        return routes

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
        """Main LKH-3 entry point for the simulator.

        Extracts LKH-3 hyperparameters from the ``values`` dict, builds the
        demand array, invokes :func:`solve_lkh`, and reformats the result.

        Args:
            sub_dist_matrix: (N × N) distance matrix (local indices).
            sub_wastes: ``{local_idx: fill_level}`` for nodes 1..M.
            capacity: Vehicle capacity.
            revenue: Revenue per unit (scaled).
            cost_unit: Cost per distance unit.
            values: Merged config dict from :meth:`BaseRoutingPolicy._load_area_params`.
            mandatory_nodes: Local indices that **must** be visited.
            **kwargs: Additional keyword arguments.

        Returns:
            ``(routes, profit, cost)`` — routes as list-of-lists (local indices),
            total profit, and total routing cost.
        """
        n = len(sub_dist_matrix)

        # --- Build demand array (0 = depot, 1..M = customer nodes) ---
        waste_arr: Optional[np.ndarray] = None
        cap: Optional[float] = None
        if sub_wastes and capacity > 0:
            waste_arr = np.zeros(n, dtype=float)
            for idx, fill in sub_wastes.items():
                if 0 <= idx < n:
                    waste_arr[idx] = fill
            cap = capacity

        # --- Extract LKH-3 hyperparameters ---
        max_trials = int(values.get("max_trials", 1000))
        runs = int(values.get("runs", 10))
        popmusic_subpath_size = int(values.get("popmusic_subpath_size", 50))
        popmusic_trials = int(values.get("popmusic_trials", 50))
        max_k_opt = int(values.get("max_k_opt", 5))
        use_ip_merging = bool(values.get("use_ip_merging", True))
        seed = values.get("seed", 42)
        vrpp = values.get("vrpp", True)
        profit_aware_operators = values.get("profit_aware_operators", False)

        np_rng = np.random.default_rng(seed if seed is not None else 42)

        # --- Run the LKH-3 engine ---
        best_tour: Optional[List[int]] = None
        best_cost = float("inf")
        for _ in range(runs):
            tour, cost = solve_lkh(
                distance_matrix=sub_dist_matrix,
                initial_tour=None,
                max_iterations=max_trials,
                waste=waste_arr,
                capacity=cap,
                recorder=self._viz,
                np_rng=np_rng,
                popmusic_subpath_size=popmusic_subpath_size,
                popmusic_trials=popmusic_trials,
                max_k_opt=max_k_opt,
                use_ip_merging=use_ip_merging,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )
            if cost < best_cost:
                best_cost = cost
                best_tour = tour

        if best_tour is None:
            # Fallback: depot-only tour
            return [[]], 0.0, 0.0

        # --- Convert flat tour → routes ---
        routes = self._tour_to_routes(best_tour)

        # --- Mandatory-node enforcement + optional-node filtering ---
        mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
        if mandatory_set:
            routes = self._enforce_mandatory_and_filter(
                routes,
                mandatory_set,
                sub_dist_matrix,
                sub_wastes,
                capacity,
                revenue,
                cost_unit,
            )

        # --- Profit calculation (VRPP: revenue * collected − routing cost) ---
        collected_nodes = {node for route in routes for node in route}
        routing_cost = self._route_cost(routes, sub_dist_matrix)
        total_fill = sum(sub_wastes.get(node, 0.0) for node in collected_nodes)
        profit = revenue * total_fill - cost_unit * routing_cost
        return routes, profit, routing_cost

    # ------------------------------------------------------------------
    # Mandatory-node helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _route_cost(routes: List[List[int]], dist: np.ndarray) -> float:
        """Total routing distance for a set of routes (depot = 0).

        Args:
            routes: List of route lists (no depot).
            dist: Distance matrix.

        Returns:
            Total travel distance.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dist[0, route[0]]
            for k in range(len(route) - 1):
                total += dist[route[k], route[k + 1]]
            total += dist[route[-1], 0]
        return total

    def _enforce_mandatory_and_filter(
        self,
        routes: List[List[int]],
        mandatory_set: set,
        dist: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
    ) -> List[List[int]]:
        """Enforce mandatory nodes and filter unprofitable optional nodes.

        1. Removes optional (non-mandatory) nodes whose marginal profit
           contribution is negative — i.e., the cost of the detour to
           collect them exceeds the revenue from their fill level.
        2. Ensures every mandatory node appears in at least one route;
           any missing mandatory node is appended as a single-node route.

        Args:
            routes: Current routes (from the LKH-3 tour split).
            mandatory_set: Set of local node indices that must be visited.
            dist: (N × N) distance matrix.
            wastes: ``{local_idx: fill_level}``.
            capacity: Vehicle capacity.
            revenue: Revenue per unit collected.
            cost_unit: Cost per distance unit.

        Returns:
            Filtered routes with mandatory nodes guaranteed.
        """
        filtered_routes: List[List[int]] = []
        for route in routes:
            if not route:
                continue

            # Keep mandatory nodes unconditionally; filter optional ones
            kept: List[int] = []
            for node in route:
                if node in mandatory_set:
                    kept.append(node)
                else:
                    # Check marginal profit of visiting this optional node
                    node_fill = wastes.get(node, 0.0)
                    node_revenue = revenue * node_fill

                    # Estimate detour cost: cost of inserting this node
                    # between its predecessor and successor
                    prev = kept[-1] if kept else 0
                    # Approximate successor as depot (worst case)
                    detour_cost = cost_unit * (dist[prev, node] + dist[node, 0] - dist[prev, 0])

                    if node_revenue > detour_cost:
                        kept.append(node)

            if kept:
                # Re-check capacity feasibility
                route_load = sum(wastes.get(n, 0.0) for n in kept)
                if route_load <= capacity + 1e-6:
                    filtered_routes.append(kept)
                else:
                    # Split into capacity-feasible sub-routes
                    filtered_routes.extend(self._split_by_capacity(kept, wastes, capacity))

        # Ensure every mandatory node is visited
        visited = {node for route in filtered_routes for node in route}
        missing = mandatory_set - visited
        for node in sorted(missing):
            filtered_routes.append([node])

        return filtered_routes

    @staticmethod
    def _split_by_capacity(
        nodes: List[int],
        wastes: Dict[int, float],
        capacity: float,
    ) -> List[List[int]]:
        """Split a node sequence into capacity-feasible sub-routes.

        Args:
            nodes: Ordered node sequence.
            wastes: Node fill levels.
            capacity: Vehicle capacity.

        Returns:
            List of sub-routes, each within capacity.
        """
        routes: List[List[int]] = []
        current: List[int] = []
        load = 0.0
        for node in nodes:
            w = wastes.get(node, 0.0)
            if load + w > capacity + 1e-6 and current:
                routes.append(current)
                current = []
                load = 0.0
            current.append(node)
            load += w
        if current:
            routes.append(current)
        return routes
