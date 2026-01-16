import random
import time
from typing import Dict, List, Set

import numpy as np

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

        self.node_map = {}
        self.route_loads = []

    def optimize(self, individual: Individual) -> Individual:
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
            else:
                if self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v):
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
        if r_u == r_v and (p_u == p_v + 1):
            return False
        dem_u = self.demands.get(u, 0)

        if r_u != r_v:
            if self._get_load_cached(r_v) + dem_u > self.Q:
                return False

        route_u = self.routes[r_u]
        route_v = self.routes[r_v]
        prev_u = route_u[p_u - 1] if p_u > 0 else 0
        next_u = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
        v_next = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0

        delta = -self.d[prev_u, u] - self.d[u, next_u] + self.d[prev_u, next_u]
        delta -= self.d[v, v_next]
        delta += self.d[v, u] + self.d[u, v_next]

        if delta * self.C < -1e-4:
            self.routes[r_u].pop(p_u)
            if r_u == r_v and p_u < p_v:
                p_v -= 1
            self.routes[r_v].insert(p_v + 1, u)
            self._update_map({r_u, r_v})
            return True
        return False

    def _move_swap(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if r_u == r_v and abs(p_u - p_v) <= 1:
            return False

        dem_u = self.demands.get(u, 0)
        dem_v = self.demands.get(v, 0)

        if r_u != r_v:
            if self._get_load_cached(r_u) - dem_u + dem_v > self.Q:
                return False
            if self._get_load_cached(r_v) - dem_v + dem_u > self.Q:
                return False

        route_u = self.routes[r_u]
        route_v = self.routes[r_v]

        prev_u = route_u[p_u - 1] if p_u > 0 else 0
        next_u = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
        prev_v = route_v[p_v - 1] if p_v > 0 else 0
        next_v = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0

        delta = -self.d[prev_u, u] - self.d[u, next_u] - self.d[prev_v, v] - self.d[v, next_v]
        delta += self.d[prev_u, v] + self.d[v, next_u] + self.d[prev_v, u] + self.d[u, next_v]

        if delta * self.C < -1e-4:
            self.routes[r_u][p_u] = v
            self.routes[r_v][p_v] = u
            self._update_map({r_u, r_v})
            return True
        return False

    def _move_2opt_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        route_u = self.routes[r_u]
        route_v = self.routes[r_v]

        tail_u = route_u[p_u + 1 :]
        tail_v = route_v[p_v + 1 :]

        l_head_u = self._calc_load_fresh(route_u[: p_u + 1])
        l_head_v = self._calc_load_fresh(route_v[: p_v + 1])
        l_tail_u = self._get_load_cached(r_u) - l_head_u
        l_tail_v = self._get_load_cached(r_v) - l_head_v

        if l_head_u + l_tail_v > self.Q or l_head_v + l_tail_u > self.Q:
            return False

        u_next = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
        v_next = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0

        delta = -self.d[u, u_next] - self.d[v, v_next] + self.d[u, v_next] + self.d[v, u_next]

        if delta * self.C < -1e-4:
            new_ru = route_u[: p_u + 1] + tail_v
            new_rv = route_v[: p_v + 1] + tail_u
            self.routes[r_u] = new_ru
            self.routes[r_v] = new_rv
            self._update_map({r_u, r_v})
            return True
        return False

    def _move_2opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if p_u >= p_v:
            return False
        if p_u + 1 == p_v:
            return False

        route = self.routes[r_u]
        u_next = route[p_u + 1]
        v_next = route[p_v + 1] if p_v < len(route) - 1 else 0

        delta = -self.d[u, u_next] - self.d[v, v_next] + self.d[u, v] + self.d[u_next, v_next]

        if delta * self.C < -1e-4:
            segment = route[p_u + 1 : p_v + 1]
            route[p_u + 1 : p_v + 1] = segment[::-1]
            self._update_map({r_u})
            return True
        return False
