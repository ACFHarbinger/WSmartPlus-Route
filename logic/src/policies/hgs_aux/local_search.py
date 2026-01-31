"""
Local Search heuristics for Hybrid Genetic Search (HGS).

This module provides various local search operators (Relocate, Swap, 2-opt)
to improve individual solutions within the HGS population.
"""

import random
import time
from typing import Dict, List, Set, Tuple

import numpy as np

from .operators import (
    move_2opt_intra,
    move_2opt_star,
    move_3opt_intra,
    move_relocate,
    move_swap,
    move_swap_star,
)
from .types import HGSParams, Individual


class LocalSearch:
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
        params: HGSParams,
    ):
        """
        Initialize the Local Search optimizer.

        Args:
            dist_matrix: NxN distance matrix.
            demands: Dictionary of node demands.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: HGS configuration parameters.
        """
        self.d = np.array(dist_matrix)
        self.demands = demands
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params

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

    def optimize(self, individual: Individual) -> Individual:
        """
        Iteratively improve an individual using local search operators.

        Args:
            individual: The HGS individual to be optimized.

        Returns:
            Individual: The improved individual.
        """
        if not individual.routes:
            return individual

        self.routes = [r[:] for r in individual.routes]
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]

        improved = True
        limit = 500  # Safety cap
        it = 0
        t_start = time.time()

        self.node_map.clear()
        for ri, r in enumerate(self.routes):
            for pi, node in enumerate(r):
                self.node_map[node] = (ri, pi)

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

        individual.routes = self.routes
        gt = []
        for r in self.routes:
            gt.extend(r)
        individual.giant_tour = gt
        return individual

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
