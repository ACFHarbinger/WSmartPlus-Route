"""
HVPL Policy Adapter.

Adapts the Hybrid Volleyball Premier League (HVPL) logic to the agnostic interface.

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023). "Volleyball premier league
    algorithm with ACO and ALNS for simultaneous pickup–delivery location
    routing problem."
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hvpl import HVPLConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.hybrid_volleyball_premier_league.params import HVPLParams
from logic.src.policies.hybrid_volleyball_premier_league.solver import HVPLSolver

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization_k_sparse.params import KSACOParams


@PolicyRegistry.register("hvpl")
class HVPLPolicy(BaseRoutingPolicy):
    """
    HVPL policy class.

    Visits pre-selected 'must_go' bins using the population-based HVPL metaheuristic.
    """

    def __init__(self, config: Optional[Union[HVPLConfig, Dict[str, Any]]] = None):
        """Initialize HVPL policy with optional config.

        Args:
            config: HVPLConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HVPLConfig

    def _get_config_key(self) -> str:
        """Return config key for HVPL."""
        return "hvpl"

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
        Run HVPL solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        # Extract sub-params for ACO and ALNS
        # 'values' contains the flattened config, with nested 'aco' and 'alns' dicts

        # Get ACO config (nested or flattened)
        aco_config = values.get("aco", {})
        if isinstance(aco_config, dict):
            aco_params = KSACOParams(
                n_ants=aco_config.get("n_ants", 20),
                k_sparse=aco_config.get("k_sparse", 10),
                alpha=aco_config.get("alpha", 1.0),
                beta=aco_config.get("beta", 2.0),
                rho=aco_config.get("rho", 0.1),
                q0=aco_config.get("q0", 0.9),
                tau_0=aco_config.get("tau_0", 1.0),
                tau_min=aco_config.get("tau_min", 0.001),
                tau_max=aco_config.get("tau_max", 10.0),
                max_iterations=aco_config.get("max_iterations", 1),
                time_limit=aco_config.get("time_limit", 60.0),
                local_search=aco_config.get("local_search", False),
                local_search_iterations=aco_config.get("local_search_iterations", 0),
                elitist_weight=aco_config.get("elitist_weight", 1.0),
            )
        else:
            # Fallback to flattened parameters for backward compatibility
            aco_params = KSACOParams(
                n_ants=values.get("aco_n_ants", 20),
                k_sparse=values.get("aco_k_sparse", 10),
                alpha=values.get("aco_alpha", 1.0),
                beta=values.get("aco_beta", 2.0),
                rho=values.get("aco_rho", 0.1),
                q0=values.get("aco_q0", 0.9),
                tau_0=values.get("aco_tau_0", 1.0),
                tau_min=values.get("aco_tau_min", 0.001),
                tau_max=values.get("aco_tau_max", 10.0),
                max_iterations=values.get("aco_iterations", 1),
                time_limit=values.get("aco_time_limit", 60.0),
                local_search=values.get("aco_local_search", False),
                local_search_iterations=values.get("aco_local_search_iterations", 0),
                elitist_weight=values.get("aco_elitist_weight", 1.0),
            )

        # Get ALNS config (nested or flattened)
        alns_config = values.get("alns", {})
        if isinstance(alns_config, dict):
            alns_params = ALNSParams(
                max_iterations=alns_config.get("max_iterations", 500),
                start_temp=alns_config.get("start_temp", 100.0),
                cooling_rate=alns_config.get("cooling_rate", 0.95),
                reaction_factor=alns_config.get("reaction_factor", 0.1),
                min_removal=alns_config.get("min_removal", 1),
                max_removal_pct=alns_config.get("max_removal_pct", 0.3),
                time_limit=alns_config.get("time_limit", 60.0),
            )
        else:
            # Fallback to flattened parameters for backward compatibility
            alns_params = ALNSParams(
                max_iterations=values.get("alns_iterations", 500),
                start_temp=values.get("alns_start_temp", 100.0),
                cooling_rate=values.get("alns_cooling_rate", 0.95),
                reaction_factor=values.get("alns_reaction_factor", 0.1),
                min_removal=values.get("alns_min_removal", 1),
                max_removal_pct=values.get("alns_max_removal_pct", 0.3),
                time_limit=values.get("alns_time_limit", 60.0),
            )

        params = HVPLParams(
            n_teams=values.get("n_teams", 30),
            max_iterations=values.get("max_iterations", 100),
            substitution_rate=values.get("substitution_rate", 0.2),
            crossover_rate=values.get("crossover_rate", 0.8),
            mutation_rate=values.get("mutation_rate", 0.1),
            elite_size=values.get("elite_size", 3),
            aco_init_iterations=values.get("aco_init_iterations", 50),
            alns_iterations=values.get("alns_iterations", 500),
            time_limit=values.get("time_limit", 300.0),
            aco_params=aco_params,
            alns_params=alns_params,
        )

        solver = HVPLSolver(
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
