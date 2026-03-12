"""
Memetic Island Model GA Policy Adapter.

Adapts the rigorous Memetic Island Model GA (replaces HVPL).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MemeticIslandModelGAConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import MemeticIslandModelGAParams
from .solver import MemeticIslandModelGASolver


@PolicyRegistry.register("ga_mim")
class MemeticIslandModelGAPolicy(BaseRoutingPolicy):
    """
    Memetic Island Model Genetic Algorithm policy class.

    Multi-population GA with ALNS local search. Replaces HVPL.
    """

    def __init__(self, config: Optional[Union[MemeticIslandModelGAConfig, Dict[str, Any]]] = None):
        """Initialize Memetic Island Model GA policy with optional config.

        Args:
            config: MemeticIslandModelGAConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MemeticIslandModelGAConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "ga_mim"

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
        from logic.src.policies.ant_colony_optimization.k_sparse_aco.params import ACOParams

        # Create nested params
        aco_params = ACOParams(
            n_ants=values.get("aco_n_ants", 10),
            k_sparse=values.get("aco_k_sparse", 10),
            max_iterations=1,
            time_limit=30,
            local_search=False,
        )

        alns_params = ALNSParams(
            max_iterations=values.get("alns_max_iterations", 100),
            start_temp=values.get("alns_start_temp", 100.0),
            cooling_rate=values.get("alns_cooling_rate", 0.95),
            time_limit=30,
        )

        params = MemeticIslandModelGAParams(
            n_islands=values.get("n_islands", 10),
            island_size=values.get("island_size", 10),
            max_generations=values.get("max_generations", 50),
            time_limit=values.get("time_limit", 60.0),
            replacement_rate=values.get("replacement_rate", 0.2),
            tournament_size=values.get("tournament_size", 3),
            migration_interval=values.get("migration_interval", 5),
            migration_size=values.get("migration_size", 1),
            aco_params=aco_params,
            alns_params=alns_params,
        )

        solver = MemeticIslandModelGASolver(
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
