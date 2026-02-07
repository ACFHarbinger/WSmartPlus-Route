import random
from typing import Dict, List, Set

import numpy as np

from .params import ACOParams
from .pheromones import SparsePheromoneTau


class SolutionConstructor:
    """
    Constructs a single solution (route set) for an ant using the
    k-sparse pheromone matrix and heuristic values.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
        pheromone: SparsePheromoneTau,
        eta: np.ndarray,
        candidate_lists: Dict[int, List[int]],
        nodes: List[int],
        params: ACOParams,
        tau_0: float,
    ):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.pheromone = pheromone
        self.eta = eta
        self.candidate_lists = candidate_lists
        self.nodes = nodes
        self.params = params
        self.tau_0 = tau_0

    def construct(self) -> List[List[int]]:
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
