"""
ILS-RVND-SP Policy Adapter.

Adapts the Iterated Local Search - Randomized Variable Neighborhood Descent - Set Partitioning (ILS-RVND-SP) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ILSRVNDSPConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp import (
    ILSRVNDSPSolver,
)
from logic.src.policies.route_construction.matheuristics.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.params import (
    ILSRVNDSPParams,
)


@RouteConstructorRegistry.register("ils_rvnd_sp")
class ILSRVNDSPPolicy(BaseRoutingPolicy):
    """
    ILS-RVND-SP policy class.

    Visits pre-selected 'mandatory' bins using Iterated Local Search, Randomized Variable Neighborhood Descent, and Set Partitioning.
    """

    def __init__(self, config: Optional[Union[ILSRVNDSPConfig, Dict[str, Any]]] = None):
        """Initialize ILS-RVND-SP policy with optional config.

        Args:
            config: ILSRVNDSPConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ILSRVNDSPConfig

    def _get_config_key(self) -> str:
        """Return config key for ILS-RVND-SP."""
        return "ils_rvnd_sp"

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
        Run ILS-RVND-SP solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = ILSRVNDSPParams(
            max_restarts=int(values.get("max_restarts", 10)),
            max_iter_ils=int(values.get("max_iter_ils", 100)),
            perturbation_strength=int(values.get("perturbation_strength", 2)),
            use_set_partitioning=bool(values.get("use_set_partitioning", True)),
            mip_time_limit=float(values.get("mip_time_limit", 60.0)),
            sp_mip_gap=float(values.get("sp_mip_gap", 0.01)),
            N=int(values.get("N", 150)),
            A=float(values.get("A", 11.0)),
            MaxIter_a=int(values.get("MaxIter_a", 50)),
            MaxIter_b=int(values.get("MaxIter_b", 100)),
            MaxIterILS_b=int(values.get("MaxIterILS_b", 2000)),
            TDev_a=float(values.get("TDev_a", 0.05)),
            TDev_b=float(values.get("TDev_b", 0.005)),
            max_vehicles=int(values.get("max_vehicles", 9999)),
            time_limit=float(values.get("time_limit", 300.0)),
            seed=int(values.get("seed", 42)),
            vrpp=bool(values.get("vrpp", True)),
            profit_aware_operators=bool(values.get("profit_aware_operators", False)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
        )

        solver = ILSRVNDSPSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, best_profit, best_cost = solver.solve()

        return routes, best_profit, best_cost
