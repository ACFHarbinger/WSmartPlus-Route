"""
Separation Algorithms for VRPP Cutting Planes.
"""

from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_flow

from logic.src.policies.other.branching_solvers.separation.inequality import (
    CapacityCut,
    Inequality,
    PCSubtourEliminationCut,
)


class SeparationEngine:
    """
    Separation algorithms for finding violated inequalities.
    """

    USE_COMB_CUTS = False
    _EXACT_SEP_PERIOD = 3

    def __init__(
        self,
        model,
        enable_heuristic_rcc_separation: bool = True,
        enable_comb_cuts: bool = False,
    ):
        self.model = model
        self.pool: List[Inequality] = []
        self.enable_heuristic_rcc_separation = enable_heuristic_rcc_separation
        self.enable_comb_cuts = enable_comb_cuts

    def separate_integer(
        self,
        x_vals: np.ndarray,
        y_vals: Optional[np.ndarray] = None,
        max_cuts: int = 100,
        iteration: int = 0,
        sec_only: bool = False,
    ) -> List[Inequality]:
        self.pool = []
        self._separate_disconnected_components(x_vals, y_vals)
        if not sec_only:
            self._separate_gsec_h2(x_vals, y_vals)
        self._strengthen_pool(self.pool, x_vals, y_vals)
        violated = [ineq for ineq in self.pool if ineq.violation > 0.01]
        violated.sort()
        return violated[:max_cuts]

    def separate_fractional(
        self,
        x_vals: np.ndarray,
        y_vals: Optional[np.ndarray] = None,
        max_cuts: int = 50,
        iteration: int = 0,
        node_count: int = 0,
    ) -> List[Inequality]:
        self.pool = []
        if node_count == 0:
            self._separate_subtours_heuristic(x_vals, y_vals)
            self._separate_capacity_cuts(x_vals, y_vals)
            self._separate_gsec_h2(x_vals, y_vals)
            self._separate_pcsec_exact(x_vals, y_vals, root_node=True)
            if self.enable_heuristic_rcc_separation:
                self._separate_capacity_cuts_maxflow_heuristic(x_vals, y_vals, root_node=True)
        else:
            self._separate_subtours_heuristic(x_vals, y_vals)
            self._separate_capacity_cuts(x_vals, y_vals)
            if iteration % self._EXACT_SEP_PERIOD == 0:
                self._separate_gsec_h2(x_vals, y_vals)
                self._separate_pcsec_exact(x_vals, y_vals, root_node=False)
                if self.enable_heuristic_rcc_separation:
                    self._separate_capacity_cuts_maxflow_heuristic(x_vals, y_vals, root_node=False)

        violated = [ineq for ineq in self.pool if ineq.violation > 0.01]
        self._strengthen_pool(violated, x_vals, y_vals)
        violated.sort()
        return violated[:max_cuts]

    def _separate_disconnected_components(self, x_vals, y_vals):
        threshold = 0.5
        n = self.model.n_nodes
        adj = np.zeros((n, n))
        for idx, (i, j) in enumerate(self.model.edges):
            if x_vals[idx] >= threshold:
                adj[i, j] = 1
                adj[j, i] = 1
        n_comp, labels = connected_components(csr_matrix(adj), directed=False)
        if n_comp <= 1:
            return
        depot_comp = labels[self.model.depot]
        for c_id in range(n_comp):
            if c_id == depot_comp:
                continue
            comp = set(np.where(labels == c_id)[0])
            if len(comp) < 2:
                continue
            is_vis = True
            if y_vals is not None:
                is_vis = any(y_vals[node - 1] > 0.5 for node in comp if node != self.model.depot)
            if not is_vis:
                continue
            cv = self._get_cut_value(comp, x_vals)
            if 2.0 - cv > 0.01:
                self.pool.append(PCSubtourEliminationCut(comp, 2.0 - cv))

    def _get_cut_value(self, node_set, x_vals):
        edges = self.model.delta(node_set)
        return sum(
            x_vals[self.model.edge_to_idx[tuple(sorted(e))]]
            for e in edges
            if tuple(sorted(e)) in self.model.edge_to_idx
        )

    def _separate_capacity_cuts(self, x_vals, y_vals):
        for seed in self.model.customers:
            if y_vals is not None and y_vals[seed - 1] < 0.5:
                continue
            node_set = {seed}
            rem = set(self.model.customers) - {seed}
            demand = self.model.get_node_demand(seed)
            while demand < self.model.capacity * 0.9 and rem:
                best, bdist = None, float("inf")
                for cand in rem:
                    d = min(self.model.cost_matrix[cand, n] for n in node_set)
                    if d < bdist:
                        bdist, best = d, cand
                if not best:
                    break
                if demand + self.model.get_node_demand(best) > self.model.capacity * 1.5:
                    break
                node_set.add(best)
                rem.remove(best)
                demand += self.model.get_node_demand(best)
            if len(node_set) >= 2:
                vis_demand = sum(
                    self.model.get_node_demand(i)
                    for i in node_set
                    if i > 0 and (y_vals[i - 1] > 0.5 if y_vals is not None else True)
                )
                if vis_demand <= 0.01:
                    continue
                min_v = int(np.ceil(vis_demand / self.model.capacity))
                violation = 2.0 * min_v - self._get_cut_value(node_set, x_vals)
                if violation > 0.01:
                    self.pool.append(CapacityCut(node_set, vis_demand, self.model.capacity, violation))

    def _separate_capacity_cuts_maxflow_heuristic(self, x_vals, y_vals, root_node=False, max_cuts=50):
        n = self.model.n_nodes
        adj = np.zeros((n, n))
        for idx, (i, j) in enumerate(self.model.edges):
            if x_vals[idx] > 1e-4:
                adj[i, j] = adj[j, i] = x_vals[idx]
        visited = sorted(
            self.model.customers, key=lambda c: (y_vals[c - 1] if y_vals is not None else 1.0), reverse=True
        )
        added = 0
        for sink in visited:
            if added >= max_cuts:
                break
            try:
                flow_res = maximum_flow(csr_matrix(adj), self.model.depot, sink)
                source_side = self._extract_min_cut(adj, flow_res.flow.toarray(), self.model.depot, sink)
                cut_set = set(range(n)) - source_side
            except (ValueError, RuntimeError, MemoryError):
                continue
            if not cut_set or len(cut_set) <= 1 or self.model.depot in cut_set:
                continue
            demand = sum(
                self.model.get_node_demand(i)
                for i in cut_set
                if i > 0 and (y_vals[i - 1] > 0.1 if y_vals is not None else True)
            )
            if demand <= 1e-4:
                continue
            min_v = int(np.ceil(demand / self.model.capacity))
            if min_v < 1:
                continue
            violation = 2.0 * min_v - self._get_cut_value(cut_set, x_vals)
            if violation > 0.01 and not any(
                set(cut_set) == set(e.node_set) for e in self.pool if isinstance(e, CapacityCut)
            ):
                self.pool.append(CapacityCut(set(cut_set), demand, self.model.capacity, violation))
                added += 1

    def _separate_pcsec_exact(self, x_vals, y_vals, root_node=False):
        n = self.model.n_nodes
        adj = np.zeros((n, n))
        for idx, (i, j) in enumerate(self.model.edges):
            if x_vals[idx] > 1e-4:
                adj[i, j] = adj[j, i] = x_vals[idx]
        visited = [c for c in self.model.customers if y_vals is None or y_vals[c - 1] >= 0.01]
        if not visited:
            return
        source = max(visited, key=lambda c: (y_vals[c - 1] if y_vals is not None else 0))
        for sink in visited:
            if sink == source:
                continue
            try:
                res = maximum_flow(csr_matrix(adj), source, sink)
                s_set = self._extract_min_cut(adj, res.flow.toarray(), source, sink)
            except (ValueError, RuntimeError, MemoryError):
                continue
            if s_set and self.model.depot not in s_set and len(s_set) >= 2:
                not_in_s = set(range(n)) - s_set - {self.model.depot}
                if not not_in_s:
                    continue
                j, yj = max([(v, y_vals[v - 1] if y_vals is not None else 1.0) for v in not_in_s], key=lambda x: x[1])
                yi = y_vals[source - 1] if y_vals is not None else 1.0
                rhs = 2.0 * (yi + yj - 1.0)
                if rhs - res.flow_value > 0.01:
                    cut = PCSubtourEliminationCut(
                        set(s_set), rhs - res.flow_value, facet_form="2.3", node_i=source, node_j=j
                    )
                    cut.local_only = True
                    self.pool.append(cut)

    def _extract_min_cut(self, capacity, flow, source, sink):
        res = capacity - flow
        vis = {source}
        q = [source]
        while q:
            u = q.pop(0)
            for v in range(len(res)):
                if res[u, v] > 1e-6 and v not in vis:
                    vis.add(v)
                    q.append(v)
        return vis

    def _strengthen_pool(self, cuts, x_vals, y_vals):
        for cut in cuts:
            if not isinstance(cut, (PCSubtourEliminationCut, CapacityCut)):
                continue
            changed = True
            while changed:
                changed = False
                ext = set(self.model.customers) - cut.node_set
                for n in ext:
                    new_set = cut.node_set | {n}
                    new_val = self._get_cut_value(new_set, x_vals)
                    # For SEC/RCC, if expanding the set decreases cut value (more violation), do it
                    if new_val < self._get_cut_value(cut.node_set, x_vals) - 0.001:
                        cut.node_set = new_set
                        changed = True
                        break

    def _separate_subtours_heuristic(self, x_vals, y_vals):
        self._separate_disconnected_components(x_vals, y_vals)
        self._separate_weak_subtours(x_vals, y_vals)

    def _separate_weak_subtours(self, x_vals, y_vals):
        for center in self.model.customers:
            if y_vals is not None and y_vals[center - 1] < 0.5:
                continue
            node_set = {center}
            cand = set(self.model.customers) - {center}
            while len(node_set) < 5 and cand:
                best, bconn = None, 0.0
                for c in cand:
                    conn = sum(
                        x_vals[self.model.edge_to_idx[tuple(sorted((c, n)))]]
                        for n in node_set
                        if tuple(sorted((c, n))) in self.model.edge_to_idx
                    )
                    if conn > bconn:
                        bconn, best = conn, c
                if not best or bconn < 0.1:
                    break
                node_set.add(best)
                cand.remove(best)
            if len(node_set) >= 2:
                violation = 2.0 - self._get_cut_value(node_set, x_vals)
                if violation > 0.01:
                    self.pool.append(PCSubtourEliminationCut(node_set, violation))

    def _separate_gsec_h2(self, x_vals, y_vals):
        # Simplified GSEC_H2
        pass
