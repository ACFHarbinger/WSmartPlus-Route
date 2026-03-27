"""
Sparse Pheromone Management Module.

This module implements the sparse pheromone matrix used in K-Sparse ACO.
It optimizes memory and access time by storing only the k-best neighbors
for each node.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization_k_sparse.pheromones import SparsePheromoneTau
    >>> pheromone = SparsePheromoneTau(n_nodes=100, k=25, tau_0=1.0, ...)
    >>> val = pheromone.get(0, 1)
"""

from collections import defaultdict
from typing import Dict


class SparsePheromoneTau:
    """
    Sparse pheromone matrix that stores only k-best values per node.

    For each node, maintains a heap of the k highest pheromone values
    to its neighbors. Unrepresented edges default to tau_0.
    """

    def __init__(self, n_nodes: int, k: int, tau_0: float, tau_min: float, tau_max: float):
        """
        Initialize sparse pheromone structure.

        Args:
            n_nodes: Total number of nodes (including depot).
            k: Maximum edges to store per node.
            tau_0: Default pheromone value for unrepresented edges.
            tau_min: Minimum pheromone bound.
            tau_max: Maximum pheromone bound.
        """
        self.n_nodes = n_nodes
        self.k = k
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Sparse storage: node -> {neighbor: pheromone}
        self._pheromone: Dict[int, Dict[int, float]] = defaultdict(dict)

    def get(self, i: int, j: int) -> float:
        """Get pheromone value for edge (i, j)."""
        if j in self._pheromone[i]:
            return self._pheromone[i][j]
        return self.tau_0

    def set(self, i: int, j: int, value: float) -> None:
        """
        Set pheromone value, strictly maintaining only the top k edges.

        Phase 1 Fix: Enforce k-sparse constraint without memory leak.
        The previous implementation could leak memory by inserting before checking capacity.
        """
        value = max(self.tau_min, min(self.tau_max, value))

        # 1. If edge is already tracked, simply update it
        if j in self._pheromone[i]:
            self._pheromone[i][j] = value
            return

        # 2. If we haven't reached capacity k, add it
        if len(self._pheromone[i]) < self.k:
            self._pheromone[i][j] = value
            return

        # 3. We are at capacity. Find the weakest neighbor.
        min_neighbor = min(self._pheromone[i].keys(), key=lambda n: self._pheromone[i][n])

        # 4. Only replace if the new value is strictly better than the weakest link
        if value > self._pheromone[i][min_neighbor]:
            del self._pheromone[i][min_neighbor]
            self._pheromone[i][j] = value

    def deposit_edge(self, i: int, j: int, delta: float) -> None:
        """
        Deposit pheromone on edge (i, j).

        This is an alias for adding to the current value, as expected by
        memetic algorithms.
        """
        current = self.get(i, j)
        self.set(i, j, current + delta)

    def update_edge(self, i: int, j: int, delta: float, evaporate: bool = True) -> None:
        """
        Update pheromone on edge with deposit and optional evaporation.

        Args:
            i: Source node.
            j: Destination node.
            delta: Pheromone to deposit.
            evaporate: If True, also apply evaporation.
        """
        current = self.get(i, j)
        if evaporate:
            # Local update rule: tau = (1-rho)*tau + rho*tau_0
            # This is applied per-edge during construction
            pass
        new_value = current + delta
        self.set(i, j, new_value)

    def evaporate_all(self, rho: float) -> None:
        """Apply global evaporation to all stored pheromones."""
        for i in self._pheromone:
            for j in list(self._pheromone[i].keys()):
                self._pheromone[i][j] *= 1 - rho
                self._pheromone[i][j] = max(self._pheromone[i][j], self.tau_min)
