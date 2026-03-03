"""
HVPL Policy Adapter.

Adapts the Hybrid Volleyball Premier League (HVPL) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hvpl import HVPLConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.hybrid_volleyball_premier_league.hvpl import HVPLSolver
from logic.src.policies.hybrid_volleyball_premier_league.params import HVPLParams

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams
from .factory import PolicyRegistry


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
        # 'values' contains the flattened config

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
            time_limit=values.get("aco_time_limit", values.get("time_limit", 60.0)),
            local_search=values.get("aco_local_search", False),
            local_search_iterations=values.get("aco_local_search_iterations", 0),
            elitist_weight=values.get("aco_elitist_weight", 1.0),
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

        params = HVPLParams(
            n_teams=values.get("n_teams", 10),
            max_iterations=values.get("max_iterations", 50),
            sub_rate=values.get("sub_rate", 0.2),
            time_limit=values.get("time_limit", 60.0),
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
