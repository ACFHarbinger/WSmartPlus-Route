"""Iterated Local Search (ILS) for VRPP.

ILS alternates between a local search descent phase and a perturbation phase.

Attributes:
    ILSSolver: Main solver class for the Iterated Local Search.

Example:
    >>> solver = ILSSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    build_greedy_routes,
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    worst_profit_removal,
    worst_removal,
)

from .params import ILSParams


class ILSSolver:
    """Iterated Local Search solver for VRPP.

    Attributes:
        dist_matrix: Symmetric distance matrix.
        wastes: Mapping of bin IDs to waste quantities.
        capacity: Maximum vehicle collection capacity.
        R: Revenue per kg of waste.
        C: Cost per kg traveled.
        params: Algorithm-specific parameters.
        mandatory_nodes: Nodes that must be visited.
        n_nodes: Number of customer nodes.
        nodes: List of node indices.
        random: Random number generator.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ILSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initializes the ILS solver.

        Args:
            dist_matrix: Distance matrix between nodes.
            wastes: Dictionary mapping node indices to waste amounts.
            capacity: Vehicle capacity.
            R: Revenue per unit collected.
            C: Cost per unit distance.
            params: ILS parameters.
            mandatory_nodes: Optional list of nodes that must be visited.

        Returns:
            None.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        self._llh_pool = [
            self._llh0,
            self._llh1,
            self._llh2,
            self._llh3,
            self._llh4,
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Run Iterated Local Search.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.perf_counter()

        # Initial solution
        routes = self._build_initial_solution()
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        # Setup modular acceptance criterion
        self.params.acceptance_criterion.setup(profit)

        for restart in range(self.params.n_restarts):
            if self.params.time_limit > 0 and time.perf_counter() - start > self.params.time_limit:
                break

            # The inner loop is the Descent phase
            inner_routes = copy.deepcopy(routes)
            inner_profit = profit
            improved = True
            inner_count = 0
            while improved and inner_count < self.params.inner_iterations:
                if self.params.time_limit > 0 and time.perf_counter() - start > self.params.time_limit:
                    break
                improved = False
                inner_count += 1

                llh_idx = self.random.randint(0, self.params.n_llh - 1)
                llh = self._llh_pool[llh_idx]

                try:
                    cand_routes = llh(copy.deepcopy(inner_routes), self.params.n_removal)
                    cand_profit = self._evaluate(cand_routes)
                except Exception:
                    continue

                # Inner descent is strictly greedy
                if cand_profit > inner_profit:
                    inner_routes = cand_routes
                    inner_profit = cand_profit
                    improved = True

            # === Acceptance Phase (Transition between local optima) ===
            # Decide whether to move from incumbent 'routes' to 'inner_routes'
            is_accepted, _ = self.params.acceptance_criterion.accept(
                current_obj=profit,
                candidate_obj=inner_profit,
                iteration=restart,
                max_iterations=self.params.n_restarts,
            )

            if is_accepted:
                routes = inner_routes
                profit = inner_profit
                if profit > best_profit:
                    best_routes = copy.deepcopy(routes)
                    best_profit = profit

            # Step criterion
            self.params.acceptance_criterion.step(
                current_obj=profit,
                candidate_obj=inner_profit,
                accepted=is_accepted,
                iteration=restart,
            )

            # === Perturbation phase (Diversification for next restart) ===
            # Perturb the (potentially updated) routes
            perturbed_routes = self._perturb(copy.deepcopy(routes))
            routes = perturbed_routes
            profit = self._evaluate(routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=restart,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """Apply strong perturbation to escape local optimum.

        Args:
            routes: Current routes.

        Returns:
            Perturbed routes.
        """
        flat = [n for r in routes for n in r]
        if len(flat) < 4:
            return routes

        n_remove = max(2, int(len(flat) * self.params.perturbation_strength))

        # Random removal of a large chunk
        partial, removed = random_removal(routes, n_remove, self.random)

        # Reinsert removed nodes
        if self.params.profit_aware_operators:
            return greedy_profit_insertion(
                routes=partial,
                removed_nodes=removed,
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )
        else:
            return greedy_insertion(
                routes=partial,
                removed_nodes=removed,
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Random removal + Greedy insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        partial, removed = random_removal(routes, n, self.random)
        if self.params.profit_aware_operators:
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Worst removal + Regret-2 insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        if self.params.profit_aware_operators:
            partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return regret_2_insertion(
                partial, removed, self.dist_matrix, self.wastes, self.capacity, self.mandatory_nodes, self.params.vrpp
            )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Cluster removal + Greedy insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes, self.random)
        if self.params.profit_aware_operators:
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Worst removal + Greedy insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        if self.params.profit_aware_operators:
            partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Random removal + Regret-2 insertion.

        Args:
            routes: Current routes.
            n: Number of nodes to remove.

        Returns:
            Repaired routes.
        """
        partial, removed = random_removal(routes, n, self.random)
        if self.params.profit_aware_operators:
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.mandatory_nodes,
                self.params.vrpp,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_solution(self) -> List[List[int]]:
        """Construct initial solution.

        Returns:
            Set of routes.
        """
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes.

        Args:
            routes: Routing sequences.

        Returns:
            Net profit.
        """
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance.

        Args:
            routes: Routing sequences.

        Returns:
            Total distance.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
