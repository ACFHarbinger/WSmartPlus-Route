"""
HGS-ALNS Hybrid Policy Adapter.

Adapts the Hybrid HGS-ALNS solver to the common simulator policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HGSALNSConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search import (
    HGSALNSParams,
    HGSALNSSolver,
)

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..hybrid_genetic_search.params import HGSParams


@PolicyRegistry.register("hgs_alns")
class HGSALNSPolicy(BaseRoutingPolicy):
    """
    Hybrid HGS-ALNS policy class for the simulator.

    Uses ALNS for the intensive education phase of HGS.
    """

    def __init__(self, config: Optional[Union[HGSALNSConfig, Dict[str, Any]]] = None):
        """Initialize HGS-ALNS policy with optional config.

        Args:
            config: HGSALNSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HGSALNSConfig

    def _get_config_key(self) -> str:
        """Return config key for HGS-ALNS hybrid."""
        return "hgs_alns"

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
        Run HGS-ALNS hybrid solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        cfg = self._config

        # Build Params from nested config
        alns_params = ALNSParams(
            max_iterations=values.get("alns_iterations", 100),
            start_temp=values.get("alns_start_temp", 100.0),
            cooling_rate=values.get("alns_cooling_rate", 0.95),
            reaction_factor=values.get("alns_reaction_factor", 0.1),
            min_removal=values.get("alns_min_removal", 1),
            max_removal_pct=values.get("alns_max_removal_pct", 0.2),
            time_limit=values.get("alns_time_limit", values.get("time_limit", 60.0)),
        )

        hgs_params = HGSParams(
            time_limit=values.get("hgs_time_limit", values.get("time_limit", 60.0)),
            mu=values.get("hgs_population_size", 50),
            nb_elite=values.get("hgs_elite_size", 5),
            mutation_rate=values.get("hgs_mutation_rate", 0.2),
            crossover_rate=values.get("hgs_crossover_rate", 0.7),
            n_offspring=values.get("hgs_n_generations", 100),  # Mapping generations to offspring for this adapter
            alpha_diversity=values.get("hgs_alpha_diversity", 0.1),
            min_diversity=values.get("hgs_min_diversity", 0.2),
            diversity_change_rate=values.get("hgs_diversity_change_rate", 0.05),
            n_iterations_no_improvement=values.get("hgs_no_improvement_threshold", 20),
            nb_granular=values.get("hgs_neighbor_list_size", 10),
            local_search_iterations=values.get("hgs_local_search_iterations", 100),
            max_vehicles=values.get("hgs_max_vehicles", 0),
        )

        # Create HGSALNSParams
        params = HGSALNSParams(
            alns_education_iterations=cfg.alns_education_iterations,
            hgs_max_iter=cfg.hgs_max_iter,
            time_limit=values.get("time_limit", 60.0),
            hgs_params=hgs_params,
            alns_params=alns_params,
        )

        solver = HGSALNSSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
            seed=values.get("seed"),
        )

        routes, profit, solver_cost = solver.solve()
        return routes, profit, solver_cost
