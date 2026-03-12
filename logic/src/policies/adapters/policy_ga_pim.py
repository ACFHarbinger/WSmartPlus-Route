"""
Pure Island Model GA Policy Adapter.

Adapts the rigorous Pure Island Model GA (replaces SLC).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import PureIslandModelGAConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.pure_island_model_genetic_algorithm import PureIslandModelGAParams, PureIslandModelGASolver

from .factory import PolicyRegistry


@PolicyRegistry.register("pure_island_ga")
class PureIslandModelGAPolicy(BaseRoutingPolicy):
    """
    Pure Island Model Genetic Algorithm policy class.

    Multi-population GA with genetic operators only. Replaces SLC.
    """

    def __init__(self, config: Optional[Union[PureIslandModelGAConfig, Dict[str, Any]]] = None):
        """Initialize Pure Island Model GA policy with optional config.

        Args:
            config: PureIslandModelGAConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return PureIslandModelGAConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "pure_island_ga"

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
        Run Pure Island Model GA solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = PureIslandModelGAParams(
            n_islands=values.get("n_islands", 10),
            island_size=values.get("island_size", 20),
            max_generations=values.get("max_generations", 100),
            crossover_rate=values.get("crossover_rate", 0.8),
            mutation_rate=values.get("mutation_rate", 0.2),
            tournament_size=values.get("tournament_size", 3),
            elitism_count=values.get("elitism_count", 2),
            migration_interval=values.get("migration_interval", 10),
            migration_size=values.get("migration_size", 2),
            time_limit=values.get("time_limit", 60.0),
        )

        solver = PureIslandModelGASolver(
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
