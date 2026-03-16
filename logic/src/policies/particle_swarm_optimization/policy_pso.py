"""
Policy adapter for Particle Swarm Optimization (PSO).

**Replaces SCA** - Proper PSO with velocity momentum.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.interfaces.adapter import IPolicyAdapter
from logic.src.policies.base.registry import PolicyRegistry

from .params import PSOParams
from .solver import PSOSolver


@PolicyRegistry.register("pso")
class PSOPolicyAdapter(IPolicyAdapter):
    """
    Policy adapter for Particle Swarm Optimization with velocity momentum.

    **TRUE PSO IMPLEMENTATION** (Kennedy & Eberhart 1995).
    Replaces the Sine Cosine Algorithm (SCA) which is mathematically
    equivalent to PSO without velocity momentum and with expensive
    trigonometric operations.

    Mathematical Superiority over SCA:
        PSO: v' = w*v + c₁*r₁*(pbest - x) + c₂*r₂*(gbest - x)
        SCA: x' = x + r₁·sin(r₂)·|r₃·gbest - x|

        Where SCA's sin(r₂) is just a random weight in [-1,1] with
        expensive transcendental computation and no periodicity exploitation.
    """

    def __init__(self, **config: Any):
        """
        Initialize PSO policy adapter.

        Args:
            **config: Configuration parameters matching PSOParams fields.
        """
        self.params = PSOParams(**config)

    def __call__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Execute PSO to solve the routing problem.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            mandatory_nodes: Nodes that must be visited.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        solver = PSOSolver(
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

    def get_name(self) -> str:
        """Return policy name."""
        return "Particle Swarm Optimization (PSO)"

    def get_acronym(self) -> str:
        """Return policy acronym."""
        return "PSO"
