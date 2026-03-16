"""
ILS-RVND-SP Policy Adapter.

Adapts the Iterated Local Search - Randomized Variable Neighborhood Descent - Set Partitioning (ILS-RVND-SP) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ILSRVNDSPConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp import (
    ILSRVNDSPSolver,
)
from logic.src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.params import (
    ILSRVNDSPParams,
)


@PolicyRegistry.register("ils_rvnd_sp")
class ILSRVNDSPPolicy(BaseRoutingPolicy):
    """
    ILS-RVND-SP policy class.

    Visits pre-selected 'must_go' bins using Iterated Local Search, Randomized Variable Neighborhood Descent, and Set Partitioning.
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
        from dataclasses import fields

        valid_fields = {f.name for f in fields(ILSRVNDSPConfig)}
        filtered_values = {k: v for k, v in values.items() if k in valid_fields}
        ils_rvnd_sp_config = ILSRVNDSPConfig(**filtered_values)
        params = ILSRVNDSPParams.from_config(ils_rvnd_sp_config)

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
