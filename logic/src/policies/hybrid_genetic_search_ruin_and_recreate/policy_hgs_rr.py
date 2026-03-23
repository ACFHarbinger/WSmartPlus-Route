"""
HGS-RR Policy Adapter.

Adapts the Hybrid Genetic Search with Ruin-and-Recreate (HGS-RR) logic
to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HGSRRConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.hybrid_genetic_search_ruin_and_recreate.hgs_rr import run_hgs_rr

from .params import HGSRRParams


@PolicyRegistry.register("hgs_rr")
class HGSRRPolicy(BaseRoutingPolicy):
    """
    Hybrid Genetic Search with Ruin-and-Recreate policy class.

    Combines evolutionary optimization with adaptive destroy/repair operators
    for solving VRPP.
    """

    def __init__(self, config: Optional[Union[HGSRRConfig, Dict[str, Any]]] = None):
        """Initialize HGS-RR policy with optional config.

        Args:
            config: HGSRRConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HGSRRConfig

    def _get_config_key(self) -> str:
        """Return config key for HGS-RR."""
        return "hgs_rr"

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
        Run HGS-RR solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = HGSRRParams(
            time_limit=values.get("time_limit", 10),
            population_size=values.get("population_size", 50),
            elite_size=values.get("elite_size", 10),
            mutation_rate=values.get("mutation_rate", 0.3),
            n_generations=values.get("n_generations", 100),
            alpha_diversity=values.get("alpha_diversity", 0.5),
            min_diversity=values.get("min_diversity", 0.2),
            diversity_change_rate=values.get("diversity_change_rate", 0.05),
            no_improvement_threshold=values.get("no_improvement_threshold", 20),
            survivor_threshold=values.get("survivor_threshold", 2.0),
            max_vehicles=values.get("max_vehicles", 0),
            crossover_rate=values.get("crossover_rate", 0.7),
            neighbor_list_size=values.get("neighbor_list_size", 10),
            # Ruin-recreate specific
            min_removal_pct=values.get("min_removal_pct", 0.1),
            max_removal_pct=values.get("max_removal_pct", 0.4),
            noise_factor=values.get("noise_factor", 0.015),
            reaction_factor=values.get("reaction_factor", 0.1),
            decay_parameter=values.get("decay_parameter", 0.95),
            operator_decay_rate=values.get("operator_decay_rate", 0.95),
            destroy_operators=values.get(
                "destroy_operators",
                ["random_removal", "worst_removal", "cluster_removal", "shaw_removal", "string_removal"],
            ),
            repair_operators=values.get(
                "repair_operators",
                ["greedy_insertion", "regret_2_insertion", "regret_k_insertion", "greedy_insertion_with_blinks"],
            ),
            score_sigma_1=values.get("score_sigma_1", 33.0),
            score_sigma_2=values.get("score_sigma_2", 9.0),
            score_sigma_3=values.get("score_sigma_3", 3.0),
            seed=values.get("seed"),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        routes, profit, solver_cost = run_hgs_rr(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes=mandatory_nodes,
        )
        return routes, profit, solver_cost
