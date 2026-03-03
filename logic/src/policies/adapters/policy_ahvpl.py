"""
AHVPL Policy Adapter.

Adapts the Augmented Hybrid Volleyball Premier League (AHVPL) logic
to the agnostic policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ahvpl import AHVPLConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.augmented_hybrid_volleyball_premier_league.ahvpl import (
    AHVPLSolver,
)
from logic.src.policies.augmented_hybrid_volleyball_premier_league.params import (
    AHVPLParams,
)

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams
from ..hybrid_genetic_search.params import HGSParams
from .factory import PolicyRegistry


@PolicyRegistry.register("ahvpl")
class AHVPLPolicy(BaseRoutingPolicy):
    """
    AHVPL policy class.

    Visits pre-selected 'must_go' bins using the augmented HVPL metaheuristic
    combining ACO, VPL, HGS, and ALNS.
    """

    def __init__(self, config: Optional[Union[AHVPLConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return AHVPLConfig

    def _get_config_key(self) -> str:
        return "ahvpl"

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
        Run AHVPL solver.

        Returns:
            Tuple of (routes, profit, solver_cost).
        """
        aco_params = ACOParams(
            n_ants=values.get("aco_n_ants", 10),
            k_sparse=values.get("aco_k_sparse", 10),
            alpha=values.get("aco_alpha", 1.0),
            beta=values.get("aco_beta", 2.0),
            rho=values.get("aco_rho", 0.1),
            q0=values.get("aco_q0", 0.9),
            tau_0=values.get("aco_tau_0"),
            tau_min=values.get("aco_tau_min", 0.001),
            tau_max=values.get("aco_tau_max", 10.0),
            max_iterations=values.get("aco_iterations", 1),
            local_search=values.get("aco_local_search", False),
            local_search_iterations=values.get("aco_local_search_iterations", 0),
            elitist_weight=values.get("aco_elitist_weight", 1.0),
            time_limit=values.get("aco_time_limit", values.get("time_limit", 60.0)),
        )

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
            population_size=values.get("hgs_population_size", 50),
            elite_size=values.get("hgs_elite_size", 5),
            mutation_rate=values.get("hgs_mutation_rate", 0.2),
            crossover_rate=values.get("hgs_crossover_rate", 0.7),
            n_generations=values.get("hgs_n_generations", 100),
            alpha_diversity=values.get("hgs_alpha_diversity", 0.1),
            local_search_iterations=values.get("hgs_local_search_iterations", 100),
            max_vehicles=values.get("hgs_max_vehicles", 0),
        )

        params = AHVPLParams(
            n_teams=values.get("n_teams", 10),
            max_iterations=values.get("max_iterations", 50),
            sub_rate=values.get("sub_rate", 0.2),
            time_limit=values.get("time_limit", 60.0),
            elite_alns_iterations=values.get("alns_elite_iterations", 500),
            not_coached_alns_iterations=values.get("alns_not_coached_iterations", 100),
            hgs_params=hgs_params,
            aco_params=aco_params,
            alns_params=alns_params,
        )

        solver = AHVPLSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
            seed=values.get("seed"),
        )

        routes, profit, solver_cost = solver.solve()
        return routes, profit, solver_cost
