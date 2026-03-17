"""
Policy adapter for Particle Swarm Optimization (PSO).

**Replaces SCA** - Proper PSO with velocity momentum.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.pso import PSOConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import PSOParams
from .solver import PSOSolver


@PolicyRegistry.register("pso")
class PSOPolicyAdapter(BaseRoutingPolicy):
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

    def __init__(self, config: Optional[Union[PSOConfig, Dict[str, Any]]] = None):
        """
        Initialize PSO policy adapter.

        Args:
            config: Configuration parameters matching PSOParams fields.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return PSOConfig

    def _get_config_key(self) -> str:
        return "pso"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Execute PSO to solve the routing problem.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        cfg = self._parse_config(values, PSOConfig)
        params = PSOParams(
            pop_size=cfg.pop_size,
            inertia_weight_start=cfg.inertia_weight_start,
            inertia_weight_end=cfg.inertia_weight_end,
            cognitive_coef=cfg.cognitive_coef,
            social_coef=cfg.social_coef,
            position_min=cfg.position_min,
            position_max=cfg.position_max,
            velocity_max=cfg.velocity_max,
            max_iterations=cfg.max_iterations,
            time_limit=cfg.time_limit,
        )

        solver = PSOSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
            seed=cfg.seed if cfg.seed is not None else kwargs.get("seed"),
        )
        return solver.solve()

    def get_name(self) -> str:
        """Return policy name."""
        return "Particle Swarm Optimization (PSO)"

    def get_acronym(self) -> str:
        """Return policy acronym."""
        return "PSO"
