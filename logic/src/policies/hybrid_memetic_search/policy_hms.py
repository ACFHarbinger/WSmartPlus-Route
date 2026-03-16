"""
Hybrid Memetic Search (HMS) Policy Adapter.

Adapts the rigorous Hybrid Memetic Search (replaces HVPL).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HybridMemeticSearchConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import HybridMemeticSearchParams
from .solver import HybridMemeticSearchSolver


@PolicyRegistry.register("hms")
class HybridMemeticSearchPolicy(BaseRoutingPolicy):
    """
    Hybrid Memetic Search policy class.

    Multi-phase hybrid solver (ACO + GA + ALNS). Replaces HVPL.
    """

    def __init__(self, config: Optional[Union[HybridMemeticSearchConfig, Dict[str, Any]]] = None):
        """Initialize HMS policy with optional config.

        Args:
            config: HybridMemeticSearchConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HybridMemeticSearchConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "hms"

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
        Run Memetic Island Model GA solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        # Import params classes
        from logic.src.policies.adaptive_large_neighborhood_search import ALNSParams
        from logic.src.policies.ant_colony_optimization_k_sparse.params import KSACOParams

        # Create nested params
        aco_params = KSACOParams(
            n_ants=20,
            k_sparse=10,
            max_iterations=1,
            time_limit=60,
            local_search=False,
        )

        alns_params = ALNSParams(
            max_iterations=values.get("alns_iterations", 100),
            start_temp=100.0,
            cooling_rate=0.95,
            time_limit=60,
        )

        params = HybridMemeticSearchParams(
            population_size=values.get("population_size", 30),
            max_generations=values.get("max_generations", 50),
            substitution_rate=values.get("substitution_rate", 0.2),
            crossover_rate=values.get("crossover_rate", 0.8),
            mutation_rate=values.get("mutation_rate", 0.1),
            elitism_count=values.get("elitism_count", 3),
            aco_init_iterations=values.get("aco_init_iterations", 50),
            alns_iterations=values.get("alns_iterations", 500),
            time_limit=values.get("time_limit", 300.0),
            aco_params=aco_params,
            alns_params=alns_params,
        )

        solver = HybridMemeticSearchSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
            seed=values.get("seed"),
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
