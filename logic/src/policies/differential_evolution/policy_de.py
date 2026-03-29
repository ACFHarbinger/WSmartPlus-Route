"""
Policy adapter for Differential Evolution (DE/rand/1/bin).

Provides the interface between the DE solver and the policy factory system.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.de import DEConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import DEParams
from .solver import DESolver


@PolicyRegistry.register("de")
class DEPolicyAdapter(BaseRoutingPolicy):
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

    def __init__(self, config: Optional[Union[DEConfig, Dict[str, Any]]] = None):
        """
        Initialize DE policy adapter.

        Args:
            config: DE configuration parameters. If None, uses defaults.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return DEConfig

    def _get_config_key(self) -> str:
        return "de"

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
        Solve VRPP instance using Differential Evolution.

        Returns:
            Tuple of (best_routes, best_profit, best_cost)
        """
        params = DEParams(
            pop_size=values.get("pop_size", 50),
            mutation_factor=values.get("mutation_factor", 0.8),
            crossover_rate=values.get("crossover_rate", 0.9),
            n_removal=values.get("n_removal", 3),
            max_iterations=values.get("max_iterations", 1000),
            local_search_iterations=values.get("local_search_iterations", 100),
            evolution_strategy=values.get("evolution_strategy", "lamarckian"),
            time_limit=values.get("time_limit", 60.0),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = DESolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        return solver.solve()
