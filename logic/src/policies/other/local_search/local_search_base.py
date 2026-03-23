"""
Local Search Base Module.

This module defines the abstract base class for local search algorithms.
It handles common initialization, neighbor lists, and provides wrappers
for atomic move operators.

Attributes:
    None

Example:
    >>> # Cannot be instantiated directly
    >>> class MyLS(LocalSearch): ...
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..operators.inter_route import (
    move_2opt_star,
    move_swap_star,
)
from ..operators.inter_route.cross_exchange import cross_exchange, improved_cross_exchange, lambda_interchange
from ..operators.inter_route.cyclic_transfer import cyclic_transfer
from ..operators.inter_route.ejection_chain import ejection_chain
from ..operators.inter_route.exchange_chain import exchange_2_0, exchange_2_1
from ..operators.intra_route import (
    move_2opt_intra,
    move_3opt_intra,
    move_or_opt,
    move_relocate,
    move_swap,
    relocate_chain,
)
from ..operators.intra_route.k_permutation import three_permutation


class LocalSearch(ABC):
    """
    Abstract base class for Local Search algorithms.
    Provides common infrastructure for neighbor lists and move operators.
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
        Initialize Local Search base class.

        Args:
            dist_matrix: The distance matrix between nodes.
            waste: A dictionary mapping node IDs to their waste amounts.
            capacity: The maximum capacity of a vehicle.
            R: A parameter, likely related to route cost or penalty.
            C: A parameter, likely related to route cost or penalty.
            params: An object containing additional parameters for the local search,
                    such as time_limit.
            neighbors: Optional pre-computed neighbor map to avoid redundant sorting.
        """
        self.d = np.array(dist_matrix)
        self.waste = waste
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.random = random.Random(params.seed) if params.seed is not None else random.Random(42)

        if neighbors is not None:
            self.neighbors = neighbors
        else:
            # Common initialization for neighbors (used by all LS)
            n_nodes = len(dist_matrix)
            self.neighbors = {}
            for i in range(1, n_nodes):
                row = self.d[i]
                order = np.argsort(row)
                cands = []
                for c in order:
                    if c not in (i, 0):
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
        it = 0
        t_start = time.process_time()
        while improved and it < self.params.local_search_iterations:
            if self.params.time_limit > 0 and time.process_time() - t_start > self.params.time_limit:
                break

            improved = False
            it += 1

            nodes = [n for n in self.neighbors if n in self.node_map]
            self.random.shuffle(nodes)

            for u in nodes:
                if self._process_node(u):
                    improved = True
                    break

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                n_routes=len(self.routes),
                improved=int(improved),
            )

    def _calc_load_fresh(self, r: List[int]) -> float:
        return sum(self.waste.get(x, 0) for x in r)

    def _process_node(self, u: int) -> bool:  # noqa: C901
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
            if getattr(self.params, "use_relocate_chain", False) and self._move_relocate_chain(u, r_u, p_u, r_v, p_v):
                return True
            if r_u != r_v:
                if self._move_2opt_star(u, v, r_u, p_u, r_v, p_v):
                    return True
                if self._move_swap_star(u, v, r_u, p_u, r_v, p_v):
                    return True

                # Advanced inter-route operators
                if getattr(self.params, "use_cross_exchange", False) and self._try_cross_exchange(r_u, p_u, r_v, p_v):
                    return True

                if getattr(self.params, "use_improved_cross_exchange", False) and self._try_improved_cross_exchange(
                    r_u, p_u, r_v, p_v
                ):
                    return True

                if getattr(self.params, "use_lambda_interchange", False) and self._try_lambda_interchange(r_u, r_v):
                    return True

                if getattr(self.params, "use_cyclic_transfer", False) and self._try_cyclic_transfer(r_u, p_u, r_v, p_v):
                    return True

                if getattr(self.params, "use_exchange_chains", False) and self._try_exchange_chains(r_u, p_u, r_v, p_v):
                    return True

                if getattr(self.params, "use_ejection_chains", False) and self._try_ejection_chain(r_u):
                    return True
            else:
                if self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v):
                    return True
                if self._move_3opt_intra(u, v, r_u, p_u, r_v, p_v, self.random):
                    return True
                if self._move_or_opt(r_u, p_u, self.random.choice([1, 2, 3])):
                    return True
                if getattr(self.params, "use_three_permutation", False) and self._move_three_permutation(u, r_u, p_u):
                    return True

        return False

    def _update_map(self, affected_indices: Set[int]):
        for ri in sorted(affected_indices):
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

    def _move_3opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: random.Random) -> bool:
        return move_3opt_intra(self, u, v, r_u, p_u, r_v, p_v, rng)

    def _move_2opt_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_intra(self, u, v, r_u, p_u, r_v, p_v)

    def _move_or_opt(self, r_idx: int, pos: int, chain_len: int) -> bool:
        return move_or_opt(self, r_idx, pos, chain_len)

    def _try_cross_exchange(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return cross_exchange(self, r_u, p_u, 1, r_v, p_v, 1)

    def _try_improved_cross_exchange(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return improved_cross_exchange(self, r_u, p_u, 1, r_v, p_v, 1)

    def _try_lambda_interchange(self, r_u: int, r_v: int) -> bool:
        return lambda_interchange(self, getattr(self.params, "lambda_max", 2))

    def _try_cyclic_transfer(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if len(self.routes) < 3:
            return False
        candidates = [r for r in range(len(self.routes)) if r != r_u and r != r_v and self.routes[r]]
        if not candidates:
            return False
        r_w = self.random.choice(candidates)
        p_w = self.random.randint(0, len(self.routes[r_w]) - 1)
        return cyclic_transfer(self, [(r_u, p_u), (r_v, p_v), (r_w, p_w)])

    def _try_exchange_chains(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if exchange_2_0(self, r_u, p_u, r_v, p_v):
            return True
        return bool(exchange_2_1(self, r_u, p_u, r_v, p_v))

    def _try_ejection_chain(self, r_u: int) -> bool:
        return ejection_chain(self, r_u)

    def _move_relocate_chain(self, u: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if relocate_chain(self, r_u, p_u, r_v, p_v, chain_len=2):
            return True
        return bool(relocate_chain(self, r_u, p_u, r_v, p_v, chain_len=3))

    def _move_three_permutation(self, u: int, r_u: int, p_u: int) -> bool:
        return three_permutation(self, r_u, p_u)
