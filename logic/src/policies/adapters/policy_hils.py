"""
HILS Policy Adapter.

Adapts the Hybrid Iterated Local Search (HILS) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HILSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.hybrid_iterated_local_search.hils import HILSSolver
from logic.src.policies.hybrid_iterated_local_search.params import HILSParams

from .factory import PolicyRegistry


@PolicyRegistry.register("hils")
class HILSPolicy(BaseRoutingPolicy):
    """
    HILS policy class.

    Visits pre-selected 'must_go' bins using Hybrid Iterated Local Search and Set Partitioning.
    """

    def __init__(self, config: Optional[Union[HILSConfig, Dict[str, Any]]] = None):
        """Initialize HILS policy with optional config.

        Args:
            config: HILSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HILSConfig

    def _get_config_key(self) -> str:
        """Return config key for HILS."""
        return "hils"

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
        Run HILS solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        hils_config = HILSConfig(**values)
        params = HILSParams.from_config(hils_config)

        solver = HILSSolver(
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
