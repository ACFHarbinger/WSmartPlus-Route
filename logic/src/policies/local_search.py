"""
Local Search heuristics for Hybrid Genetic Search (HGS).

This module provides various local search operators (Relocate, Swap, 2-opt)
to improve individual solutions within the HGS population.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

import numpy as np

from .operators import (
    move_2opt_intra,
    move_2opt_star,
    move_3opt_intra,
    move_relocate,
    move_swap,
    move_swap_star,
)

if TYPE_CHECKING:
    from .hybrid_genetic_search.types import HGSParams, Individual


class LocalSearch(ABC):
    """
    Abstract base class for Local Search algorithms.
    Provides common infrastructure for neighbor lists and move operators.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
    ):
        self.d = np.array(dist_matrix)
        self.demands = demands
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params

        # Common initialization for neighbors (used by all LS)
        n_nodes = len(dist_matrix)
        self.neighbors = {}
        for i in range(1, n_nodes):
            row = self.d[i]
            order = np.argsort(row)
            cands = []
            for c in order:
                if c != i and c != 0:
                    cands.append(c)
                    if len(cands) >= 10:
                        break
            self.neighbors[i] = cands

        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.route_loads: List[float] = []
        self.routes: List[List[int]] = []

    @abstractmethod
    def optimize(self, solution: Any) -> Any:
        """
        Optimize the given solution.
        """
        pass

    def _optimize_internal(self):
        """
        Core local search loop. Assumes self.routes is populated.
        """
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]

        # Initialize node map
        self.node_map.clear()
        for ri, r in enumerate(self.routes):
            for pi, node in enumerate(r):
                self.node_map[node] = (ri, pi)

        improved = True
        limit = 500  # Safety cap
        it = 0
        t_start = time.time()

        while improved and it < limit:
            improved = False
            it += 1
            if it % 50 == 0 and (time.time() - t_start > self.params.time_limit):
                break

            nodes = [n for n in self.neighbors.keys() if n in self.node_map]
            random.shuffle(nodes)

            for u in nodes:
                if self._process_node(u):
                    improved = True
                    break

    def _calc_load_fresh(self, r: List[int]) -> float:
        return sum(self.demands.get(x, 0) for x in r)

    def _process_node(self, u: int) -> bool:
        u_loc = self.node_map.get(u)
        if not u_loc:
            return False
        r_u, p_u = u_loc

        for v in self.neighbors[u]:
            v_loc = self.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc

            if self._move_relocate(u, v, r_u, p_u, r_v, p_v):
                return True
            if self._move_swap(u, v, r_u, p_u, r_v, p_v):
                return True
            if r_u != r_v:
                if self._move_2opt_star(u, v, r_u, p_u, r_v, p_v):
                    return True
                if self._move_swap_star(u, v, r_u, p_u, r_v, p_v):
                    return True
            else:
                if self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v):
                    return True
                if self._move_3opt_intra(u, v, r_u, p_u, r_v, p_v):
                    return True

        return False

    def _update_map(self, affected_indices: Set[int]):
        for ri in affected_indices:
            for pi, node in enumerate(self.routes[ri]):
                self.node_map[node] = (ri, pi)
            self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])

    def _get_load_cached(self, ri: int) -> float:
        return self.route_loads[ri]

    def _move_relocate(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_relocate(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_swap(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_swap_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_3opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_3opt_intra(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_intra(self, u, v, r_u, p_u, r_v, p_v)


class HGSLocalSearch(LocalSearch):
    """
    Local Search module for HGS.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: "HGSParams",
    ):
        super().__init__(dist_matrix, demands, capacity, R, C, params)

    def optimize(self, solution: "Individual") -> "Individual":
        """
        Iteratively improve an individual using local search operators.
        """
        if not solution.routes:
            return solution

        self.routes = [r[:] for r in solution.routes]

        # Run optimization
        self._optimize_internal()

        solution.routes = self.routes
        gt = []
        for r in self.routes:
            gt.extend(r)
        solution.giant_tour = gt
        return solution


class ACOLocalSearch(LocalSearch):
    """
    Local Search module for K-Sparse ACO.
    Implements 2-opt local search refinement.
    """

    def optimize(self, solution: List[List[int]]) -> List[List[int]]:
        """
        Apply full set of local search operators to the solution.

        Args:
            solution: List of routes, where each route is a list of node indices.

        Returns:
            List of optimized routes.
        """
        # Create a deep copy of routes to modify
        self.routes = [r[:] for r in solution]

        # Run optimization
        self._optimize_internal()

        return self.routes
