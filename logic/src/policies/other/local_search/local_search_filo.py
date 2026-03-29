"""
FILO Local Search Module.

This module provides the local search implementation specifically for
Fast Iterative Localized Optimization (FILO). It extends the base local
search with active node localization for O(1) scaling per iteration.

References:
    Accorsi, L., & Vigo, D. "A fast and scalable heuristic for the solution
    of large-scale capacitated vehicle routing problems", 2021.

Attributes:
    None

Example:
    >>> from logic.src.policies.fast_iterative_localized_optimization.local_search import FILOLocalSearch
    >>> ls = FILOLocalSearch(dist_matrix, waste, capacity, R, C, params)
    >>> active_nodes = {1, 5, 12, 23}  # Recently ruined + stagnant nodes
    >>> optimized_routes = ls.optimize(routes, active_nodes=active_nodes)
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np

from logic.src.policies.other.local_search.local_search_base import LocalSearch


class FILOLocalSearch(LocalSearch):
    """
    Local Search module for FILO with active node localization.

    This class implements the localized neighborhood search described in
    Accorsi & Vigo (2021), where the search is restricted to moves involving
    nodes in the active set. This achieves O(1) complexity per iteration
    instead of O(n²).

    Key Features:
        - Active node filtering for localized search
        - Supports full search when active_nodes=None (standard behavior)
        - Compatible with all standard move operators (2-opt, swap, relocate, etc.)
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        neighbors: Optional[Dict[int, List[int]]] = None,
    ):
        """
        Initialize FILO Local Search.

        Args:
            dist_matrix: The distance matrix between nodes.
            waste: A dictionary mapping node IDs to their waste amounts.
            capacity: The maximum capacity of a vehicle.
            R: Revenue multiplier per kg.
            C: Cost multiplier per km.
            params: An object containing additional parameters for the local search,
                    such as time_limit and local_search_iterations.
            neighbors: Optional pre-computed neighbor map to avoid redundant sorting.
        """
        super().__init__(dist_matrix, waste, capacity, R, C, params, neighbors)

    def optimize(
        self,
        solution: List[List[int]],
        active_nodes: Optional[Set[int]] = None,
        target_neighborhood: Optional[str] = None,
    ) -> List[List[int]]:
        """
        Apply local search operators to the solution with optional localization.

        This method implements the FILO localization mechanism. When active_nodes
        is provided, the search is restricted to moves involving at least one
        active node, achieving O(1) scaling per iteration.

        Args:
            solution: List of routes, where each route is a list of node indices.
            active_nodes: If provided, restricts search to moves involving at least
                        one active node. This enables O(1) localization (FILO).
                        If None, performs standard exhaustive local search.
            target_neighborhood: If provided, only applies the specified neighborhood operator.
                               If None or "all", applies all available operators.
                               Valid values: "intra_relocate", "intra_swap", "intra_2opt",
                               "intra_3opt", "intra_or_opt", "inter_relocate", "inter_swap",
                               "inter_2opt_star", "inter_swap_star", "cross_exchange",
                               "improved_cross_exchange", "lambda_interchange", "cyclic_transfer",
                               "exchange_chains", "ejection_chains", "relocate_chain", "three_permutation",
                               "unrouted_insert", "all"

        Returns:
            List of optimized routes.

        Example:
            >>> # FILO mode: Localized search (O(1) per iteration)
            >>> active_nodes = {1, 5, 12, 23}
            >>> routes = ls.optimize(routes, active_nodes=active_nodes)
            >>>
            >>> # Standard mode: Exhaustive search (O(n²) per iteration)
            >>> routes = ls.optimize(routes, active_nodes=None)
        """
        # Create a deep copy of routes to modify
        self.routes = [r[:] for r in solution]

        # Run optimization with optional active node filtering and neighborhood filter
        self._optimize_internal(target_neighborhood=target_neighborhood, active_nodes=active_nodes)

        return self.routes
