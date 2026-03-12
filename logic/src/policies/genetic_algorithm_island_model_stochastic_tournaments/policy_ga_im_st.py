"""
Island Model Genetic Algorithm with Stochastic Tournaments (IMGA-ST) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ga_im_st import IslandModelSTGAConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import IslandModelSTGAParams
from .solver import IslandModelSTGASolver


@PolicyRegistry.register("ga_im_st")
class IslandModelSTGAPolicy(BaseRoutingPolicy):
    """
    Island Model STGA policy class.
    """

    def __init__(self, config: Optional[Union[IslandModelSTGAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return IslandModelSTGAConfig

    def _get_config_key(self) -> str:
        return "ga_im_st"

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
        params = IslandModelSTGAParams(
            n_islands=values.get("n_islands", 4),
            island_size=values.get("island_size", 10),
            max_generations=values.get("max_generations", 100),
            migration_interval=values.get("migration_interval", 10),
            migration_size=values.get("migration_size", 2),
            tournament_size=values.get("tournament_size", 2),
            selection_pressure=values.get("selection_pressure", 0.5),
            crossover_rate=values.get("crossover_rate", 0.8),
            mutation_rate=values.get("mutation_rate", 0.2),
            alns_iterations=values.get("alns_iterations", 50),
            elitism_size=values.get("elitism_size", 2),
            time_limit=values.get("time_limit", 300.0),
        )

        solver = IslandModelSTGASolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
            seed=values.get("seed"),
        )

        return solver.solve()
