"""
Local Search Operators:
    Exchange:
        - λ-interchange (cross-exchange)
        - OR-opt (relocate chains)
        - Ejection chains

    Route:
        - 2-opt* (inter-route tail exchange)
        - SWAP* (inter-route node swap)
        - 2-opt (intra-route reversal)
        - 3-opt (intra-route reconnection)

    Move:
        - Relocate single nodes
        - Swap within route

    Heuristics:
        - Lin-Kernighan-Helsgaun (LKH) for TSP sub-problems
"""

import random
from typing import Dict, List, Optional

import numpy as np

# Import operators from their respective modules
from ..other.operators.heuristics import solve_lkh
from ..other.operators.inter_route import (
    cross_exchange,
    ejection_chain,
    lambda_interchange,
    move_2opt_star,
    move_swap_star,
)
from ..other.operators.intra_route import move_2opt_intra, move_3opt_intra, move_or_opt, move_relocate, move_swap


class LocalSearchManager:
    """
    Manages local search operators for ACO route improvement.

    This class provides a unified interface to all local search operators
    from the operators module, compatible with Q-Learning operator selection.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        improvement_threshold: float,
        seed: Optional[int] = None,
    ):
        """Initialize local search manager."""
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.improvement_threshold = improvement_threshold
        self.rng = np.random.default_rng(seed)
        self.random_std = random.Random(seed)

        # Routes interface for operators
        self.routes: List[List[int]] = []

        # Load cache for efficiency
        self._load_cache: Dict[int, float] = {}

    def set_routes(self, routes: List[List[int]]) -> None:
        """
        Update the current working set of routes.

        Args:
            routes (List[List[int]]): New set of routes to optimize.
        """
        self.routes = [r[:] for r in routes]
        self._invalidate_cache()

    def get_routes(self) -> List[List[int]]:
        """
        Get a copy of the current routes.

        Returns:
            List[List[int]]: Deep copy of the current routes.
        """
        return [r[:] for r in self.routes]

    def _calc_load_fresh(self, route: List[int]) -> float:
        """
        Calculate total waste load of a specific route.

        Args:
            route (List[int]): The sequence of nodes in the route.

        Returns:
            float: Sum of wastes for all nodes in the route.
        """
        return sum(self.waste.get(n, 0) for n in route)

    def _get_load_cached(self, r_idx: int) -> float:
        """
        Retrieve the waste load for a route, using a cache for efficiency.

        Args:
            r_idx (int): Global index of the route.

        Returns:
            float: Total waste load.
        """
        if r_idx not in self._load_cache:
            self._load_cache[r_idx] = self._calc_load_fresh(self.routes[r_idx])
        return self._load_cache[r_idx]

    def _update_map(self, route_indices: set) -> None:
        """
        Synchronize the load cache and internal maps after route modifications.

        Args:
            route_indices (set): Set of indices for routes that were changed.
        """
        for r_idx in route_indices:
            if r_idx < len(self.routes):
                self._load_cache[r_idx] = self._calc_load_fresh(self.routes[r_idx])

    def _invalidate_cache(self) -> None:
        """Clear the entire load cache."""
        self._load_cache.clear()

    # ===== Exchange Operators =====

    def or_opt(self, chain_len: int = 2) -> bool:
        """
        OR-opt: relocate chain of nodes using the imported operator.

        Tries all possible OR-opt moves and applies the first improving one.
        """
        if not self.routes:
            return False

        for r_idx, route in enumerate(self.routes):
            if len(route) < chain_len:
                continue

            for pos in range(len(route)):
                if pos + chain_len > len(route):
                    continue

                node = route[pos]
                if move_or_opt(self, node, chain_len, r_idx, pos):
                    return True

        return False

    def cross_exchange_op(self, max_seg_len: int = 2) -> bool:
        """
        Cross-exchange: swap segments between routes.

        Systematically tries cross-exchange moves between route pairs.
        """
        if len(self.routes) < 2:
            return False

        for r_a in range(len(self.routes)):
            for r_b in range(r_a + 1, len(self.routes)):
                for seg_a_len in range(max_seg_len + 1):
                    for seg_b_len in range(max_seg_len + 1):
                        if seg_a_len == 0 and seg_b_len == 0:
                            continue

                        for seg_a_start in range(len(self.routes[r_a]) - seg_a_len + 1):
                            for seg_b_start in range(len(self.routes[r_b]) - seg_b_len + 1):
                                if cross_exchange(self, r_a, seg_a_start, seg_a_len, r_b, seg_b_start, seg_b_len):
                                    return True
        return False

    def lambda_interchange_op(self, lambda_max: int = 2) -> bool:
        """
        λ-interchange: generalized cross-exchange with segment lengths up to λ.
        """
        return lambda_interchange(self, lambda_max)

    def ejection_chain_op(self, max_depth: int = 3) -> bool:
        """
        Ejection chain: attempt to empty routes for fleet minimization.
        """
        if len(self.routes) < 2:
            return False

        # Try to eject from smallest routes first
        route_sizes = [(i, len(r)) for i, r in enumerate(self.routes) if r]
        route_sizes.sort(key=lambda x: x[1])

        # Try ejection on up to 3 smallest routes
        return any(ejection_chain(self, r_idx, max_depth) for r_idx, _ in route_sizes[:3])

    # ===== Route Operators =====

    def two_opt_star(self) -> bool:
        """2-opt* inter-route operator using imported implementation."""
        if len(self.routes) < 2:
            return False

        for r_u in range(len(self.routes)):
            for r_v in range(r_u + 1, len(self.routes)):
                route_u = self.routes[r_u]
                route_v = self.routes[r_v]

                if not route_u or not route_v:
                    continue

                for p_u in range(len(route_u)):
                    u = route_u[p_u]
                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_2opt_star(self, u, v, r_u, p_u, r_v, p_v):
                            return True

        return False

    def swap_star(self) -> bool:
        """SWAP* inter-route operator using imported implementation."""
        if len(self.routes) < 2:
            return False

        for r_u in range(len(self.routes)):
            for r_v in range(r_u + 1, len(self.routes)):
                route_u = self.routes[r_u]
                route_v = self.routes[r_v]

                if not route_u or not route_v:
                    continue

                for p_u in range(len(route_u)):
                    u = route_u[p_u]
                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_swap_star(self, u, v, r_u, p_u, r_v, p_v):
                            return True

        return False

    def two_opt_intra(self) -> bool:
        """2-opt intra-route: reverse segments within routes."""
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 3:
                continue

            for p_u in range(len(route) - 1):
                u = route[p_u]
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_2opt_intra(self, u, v, r_u, p_u, r_u, p_v):
                        return True
        return False

    def three_opt_intra(self) -> bool:
        """3-opt intra-route: reconnect three segments within routes."""
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 4:
                continue

            for p_u in range(len(route) - 2):
                u = route[p_u]
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_3opt_intra(self, u, v, r_u, p_u, r_u, p_v, self.random_std):
                        return True
        return False

    # ===== Move Operators =====

    def relocate(self) -> bool:
        """Relocate: move single nodes to better positions."""
        for r_u in range(len(self.routes)):
            route_u = self.routes[r_u]
            if not route_u:
                continue

            for p_u in range(len(route_u)):
                u = route_u[p_u]

                # Try relocating to all routes
                for r_v in range(len(self.routes)):
                    route_v = self.routes[r_v]

                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_relocate(self, u, v, r_u, p_u, r_v, p_v):
                            return True
        return False

    def swap(self) -> bool:
        """Swap: exchange positions of two nodes."""
        for r_u in range(len(self.routes)):
            route_u = self.routes[r_u]
            if not route_u:
                continue

            for p_u in range(len(route_u)):
                u = route_u[p_u]

                # Try swapping with nodes in all routes
                for r_v in range(r_u, len(self.routes)):
                    route_v = self.routes[r_v]
                    start_p_v = p_u + 1 if r_v == r_u else 0

                    for p_v in range(start_p_v, len(route_v)):
                        v = route_v[p_v]
                        if move_swap(self, u, v, r_u, p_u, r_v, p_v):
                            return True
        return False

    # ===== Heuristic Operators =====

    def lkh_refinement(self) -> bool:
        """
        Apply Lin-Kernighan-Helsgaun heuristic to refine individual routes.

        Uses LKH to optimize each route as a TSP subproblem.
        """
        improved = False
        waste_array = np.zeros(len(self.d))
        for node, w in self.waste.items():
            if node < len(waste_array):
                waste_array[node] = w

        for r_idx, route in enumerate(self.routes):
            if len(route) < 3:
                continue

            # Build TSP instance for this route
            nodes = [0] + route + [0]
            initial_cost = sum(self.d[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1))

            # Apply LKH
            improved_tour, improved_cost = solve_lkh(
                self.d, initial_tour=nodes, max_iterations=20, waste=waste_array, capacity=self.Q, np_rng=self.rng
            )

            if improved_cost < initial_cost - 1e-6:
                # Extract route from tour (remove depot duplicates)
                new_route = [n for n in improved_tour if n != 0]
                self.routes[r_idx] = new_route
                self._update_map({r_idx})
                improved = True

        return improved
