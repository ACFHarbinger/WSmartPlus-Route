"""
Policy adapter for Particle Swarm Optimization (PSO).

**Replaces SCA** - Proper PSO with velocity momentum.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.pso import PSOConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import PSOParams
from .solver import PSOSolver


@RouteConstructorRegistry.register("pso")
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
        Execute the Particle Swarm Optimization (PSO) solver logic.

        PSO is a computational method that optimizes a problem by iteratively
        trying to improve a candidate solution with regard to a given measure
        of quality. It solves a problem by having a population of candidate
        solutions, here dubbed particles, and moving these particles around in
        the search-space according to simple mathematical formulae over the
        particle's position and velocity. Each particle's movement is
        influenced by its local best known position ("pbest") but is also
        guided toward the best known positions in the search-space ("gbest"),
        which are updated as better positions are found by other particles.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                PSO parameters (pop_size, cognitive_coef, social_coef, inertia_weight).
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = PSOParams(
            pop_size=values.get("pop_size", 30),
            inertia_weight_start=values.get("inertia_weight_start", 0.9),
            inertia_weight_end=values.get("inertia_weight_end", 0.4),
            cognitive_coef=values.get("cognitive_coef", 2.0),
            social_coef=values.get("social_coef", 2.0),
            position_min=values.get("position_min", -1.0),
            position_max=values.get("position_max", 1.0),
            velocity_max=values.get("velocity_max", 0.5),
            max_iterations=values.get("max_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = PSOSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )
        return solver.solve()

    def get_name(self) -> str:
        """Return policy name."""
        return "Particle Swarm Optimization (PSO)"

    def get_acronym(self) -> str:
        """Return policy acronym."""
        return "PSO"
