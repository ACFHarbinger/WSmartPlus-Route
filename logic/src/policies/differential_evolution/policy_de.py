"""
Policy adapter for Differential Evolution (DE/rand/1/bin).

Provides the interface between the DE solver and the policy factory system.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.base.registry import PolicyRegistry
from logic.src.interfaces import IPolicyAdapter

from .params import DEParams
from .solver import DESolver


@PolicyRegistry.register("de")
class DEPolicyAdapter(IPolicyAdapter):
    """
    Policy adapter for Differential Evolution with rigorous DE/rand/1/bin mechanics.

    Replaces Artificial Bee Colony (ABC), which is mathematically equivalent to
    Differential Evolution with fitness-proportionate selection instead of greedy
    selection. This implementation uses proper DE mechanics:

    - Greedy one-to-one selection (not fitness-proportionate)
    - Explicit crossover operator with CR parameter
    - Differential mutation: v = x_r1 + F × (x_r2 - x_r3)
    - No metaphor (employed/onlooker/scout bees)
    - No trial counter or abandonment mechanism

    Mathematical Foundation:
        1. Mutation: v_i = x_r1 + F × (x_r2 - x_r3)
        2. Crossover: u_ij = v_ij if rand() < CR else x_ij
        3. Selection: x_i(t+1) = u_i if f(u_i) ≥ f(x_i) else x_i

    Reference:
        Storn, R., & Price, K. (1997). "Differential Evolution – A Simple and
        Efficient Heuristic for Global Optimization over Continuous Spaces."
        Journal of Global Optimization, 11(4), 341-359.
    """

    def __init__(self, params: Optional[DEParams] = None):
        """
        Initialize DE policy adapter.

        Args:
            params: DE configuration parameters. If None, uses defaults.
        """
        self.params = params or DEParams()

    def solve(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Solve VRPP instance using Differential Evolution.

        Args:
            dist_matrix: Distance matrix (n+1, n+1), index 0 is depot
            wastes: Mapping of node IDs to waste quantities
            capacity: Maximum vehicle capacity
            R: Revenue per unit waste
            C: Cost per unit distance
            mandatory_nodes: Nodes that must be visited
            seed: Random seed for reproducibility

        Returns:
            Tuple of (best_routes, best_profit, best_cost)
        """
        solver = DESolver(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=self.params,
            mandatory_nodes=mandatory_nodes,
            seed=seed,
        )

        return solver.solve()
