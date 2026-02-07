"""
K-Sparse Ant Colony Optimization (ACO) for Vehicle Routing.

This module implements a K-Sparse variant of Ant Colony System (ACS)
optimized for CVRP and VRPP problems.

Key Features:
- Sparse pheromone matrix: Only k-best edges per node are stored
- Pseudo-random proportional rule for state transition
- Local pheromone update for exploration
- Global pheromone update on best-so-far solution
- Optional 2-opt local search refinement

Reference:
    Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: a cooperative
    learning approach to the traveling salesman problem. IEEE Transactions on
    Evolutionary Computation, 1(1), 53-66.
"""

import random
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from ..local_search import ACOLocalSearch
from .params import ACOParams
from .pheromones import SparsePheromoneTau


class KSparseACOSolver:
    """
    K-Sparse Ant Colony System solver for CVRP/VRPP.

    Implements ACS with sparse pheromone storage for memory efficiency
    and fast computation on large problem instances.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ACOParams,
    ):
        """
        Initialize the K-Sparse ACO solver.

        Args:
            dist_matrix: NxN distance matrix.
            demands: Dictionary of node demands {node_idx: demand}.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: ACO hyperparameters.
        """
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params

        self.n_nodes = len(dist_matrix)
        self.nodes = list(range(1, self.n_nodes))  # Exclude depot (0)

        # Precompute heuristic values (eta = 1/distance)
        self.eta = np.zeros_like(dist_matrix, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0.0)

        # Compute initial pheromone based on nearest neighbor heuristic
        if params.tau_0 is None:
            nn_cost = self._nearest_neighbor_cost()
            self.tau_0 = 1.0 / (self.n_nodes * nn_cost) if nn_cost > 0 else params.tau_max
        else:
            self.tau_0 = params.tau_0

        # Initialize sparse pheromone matrix
        self.pheromone = SparsePheromoneTau(
            self.n_nodes,
            params.k_sparse,
            self.tau_0,
            params.tau_min,
            params.tau_max,
        )

        # Initialize sparse pheromone matrix
        self.pheromone = SparsePheromoneTau(
            self.n_nodes,
            params.k_sparse,
            self.tau_0,
            params.tau_min,
            params.tau_max,
        )

        # Initialize Local Search
        self.ls = ACOLocalSearch(dist_matrix, demands, capacity, R, C, params)

        # Build candidate lists (k-nearest neighbors for each node)
        self.candidate_lists = self._build_candidate_lists()

    def _nearest_neighbor_cost(self) -> float:
        """Compute cost of nearest neighbor tour for tau_0 initialization."""
        visited = set([0])
        current = 0
        cost = 0.0
        for _ in range(len(self.nodes)):
            best_next = None
            best_dist = float("inf")
            for node in self.nodes:
                if node not in visited:
                    d = self.dist_matrix[current][node]
                    if d < best_dist:
                        best_dist = d
                        best_next = node
            if best_next is not None:
                cost += best_dist
                visited.add(best_next)
                current = best_next
        cost += self.dist_matrix[current][0]  # Return to depot
        return cost

    def _build_candidate_lists(self) -> Dict[int, List[int]]:
        """Build k-nearest neighbor candidate lists for each node."""
        candidates: Dict[int, List[int]] = {}
        k = min(self.params.k_sparse, len(self.nodes))

        for i in range(self.n_nodes):
            # Get distances to all other nodes
            distances = [(self.dist_matrix[i][j], j) for j in range(self.n_nodes) if j != i]
            distances.sort()
            candidates[i] = [j for _, j in distances[:k]]

        return candidates

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the K-Sparse ACO algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: (best_routes, profit, cost)
        """
        best_routes: List[List[int]] = []
        best_cost = float("inf")
        start_time = time.time()

        for iteration in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            iteration_best_routes: List[List[int]] = []
            iteration_best_cost = float("inf")

            # Each ant constructs a solution
            for _ in range(self.params.n_ants):
                routes = self._construct_solution()

                # Optional local search
                if self.params.local_search:
                    routes = self.ls.optimize(routes)

                cost = self._calculate_cost(routes)

                if cost < iteration_best_cost:
                    iteration_best_cost = cost
                    iteration_best_routes = routes

            # Update global best
            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_routes = iteration_best_routes

            # Global pheromone update
            self._global_pheromone_update(best_routes, best_cost)

        # Calculate profit
        collected_revenue = sum(self.demands.get(node, 0) * self.R for route in best_routes for node in route)
        profit = collected_revenue - best_cost * self.C

        return best_routes, profit, best_cost

    def _construct_solution(self) -> List[List[int]]:
        """
        Construct a solution using the ACS state transition rule.

        Returns:
            List of routes, each a list of node indices.
        """
        unvisited: Set[int] = set(self.nodes)
        routes: List[List[int]] = []

        while unvisited:
            route: List[int] = []
            load = 0.0
            current = 0  # Start from depot

            while unvisited:
                # Find feasible next nodes
                feasible = [j for j in unvisited if load + self.demands.get(j, 0) <= self.capacity]

                if not feasible:
                    break

                # Select next node
                next_node = self._select_next_node(current, feasible)

                # Local pheromone update (ACS rule)
                self._local_pheromone_update(current, next_node)

                route.append(next_node)
                load += self.demands.get(next_node, 0)
                unvisited.remove(next_node)
                current = next_node

            if route:
                routes.append(route)

        return routes

    def _select_next_node(self, current: int, feasible: List[int]) -> int:
        """
        Select next node using pseudo-random proportional rule.

        With probability q0, select best node (exploitation).
        Otherwise, use roulette wheel selection (exploration).
        """
        if random.random() < self.params.q0:
            # Exploitation: select best
            best_node = max(
                feasible,
                key=lambda j: (
                    self.pheromone.get(current, j) ** self.params.alpha * self.eta[current][j] ** self.params.beta
                ),
            )
            return best_node
        else:
            # Exploration: proportional selection
            # Prefer candidates if available
            candidates_in_feasible = [j for j in self.candidate_lists.get(current, []) if j in feasible]
            selection_pool = candidates_in_feasible if candidates_in_feasible else feasible

            probs = []
            for j in selection_pool:
                tau = self.pheromone.get(current, j)
                eta = self.eta[current][j]
                probs.append((tau**self.params.alpha) * (eta**self.params.beta))

            total = sum(probs)
            if total <= 0:
                return random.choice(selection_pool)

            r = random.uniform(0, total)
            cumsum = 0.0
            for idx, p in enumerate(probs):
                cumsum += p
                if cumsum >= r:
                    return selection_pool[idx]

            return selection_pool[-1]

    def _local_pheromone_update(self, i: int, j: int) -> None:
        """
        Apply ACS local pheromone update rule.

        tau(i,j) = (1 - rho) * tau(i,j) + rho * tau_0
        """
        rho = self.params.rho
        current = self.pheromone.get(i, j)
        new_value = (1 - rho) * current + rho * self.tau_0
        self.pheromone.set(i, j, new_value)

    def _global_pheromone_update(self, best_routes: List[List[int]], best_cost: float) -> None:
        """
        Apply ACS global pheromone update on best-so-far solution.

        Only edges in the best solution receive pheromone deposit.
        """
        if not best_routes or best_cost <= 0:
            return

        # Evaporate all pheromones
        self.pheromone.evaporate_all(self.params.rho)

        # Deposit on best solution edges
        delta = self.params.elitist_weight / best_cost

        for route in best_routes:
            if not route:
                continue

            # Depot to first node
            self.pheromone.update_edge(0, route[0], delta, evaporate=False)

            # Route edges
            for k in range(len(route) - 1):
                self.pheromone.update_edge(route[k], route[k + 1], delta, evaporate=False)

            # Last node back to depot
            self.pheromone.update_edge(route[-1], 0, delta, evaporate=False)

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total


def run_aco(
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    *args: Any,
) -> Tuple[List[List[int]], float, float]:
    """
    Main entry point for K-Sparse ACO solver.

    Args:
        dist_matrix: Distance matrix.
        demands: Node demands dictionary.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Configuration dictionary with ACO parameters.
        *args: Additional arguments (ignored).

    Returns:
        Tuple[List[List[int]], float, float]: (routes, profit, cost)
    """
    params = ACOParams(
        n_ants=values.get("n_ants", 10),
        k_sparse=values.get("k_sparse", 15),
        alpha=values.get("alpha", 1.0),
        beta=values.get("beta", 2.0),
        rho=values.get("rho", 0.1),
        q0=values.get("q0", 0.9),
        tau_0=values.get("tau_0"),
        tau_min=values.get("tau_min", 0.001),
        tau_max=values.get("tau_max", 10.0),
        max_iterations=values.get("max_iterations", 100),
        time_limit=values.get("time_limit", 30.0),
        local_search=values.get("local_search", True),
        elitist_weight=values.get("elitist_weight", 1.0),
    )

    solver = KSparseACOSolver(dist_matrix, demands, capacity, R, C, params)
    return solver.solve()
