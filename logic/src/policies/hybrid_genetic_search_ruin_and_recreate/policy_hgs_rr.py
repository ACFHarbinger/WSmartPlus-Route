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
        routes, profit, solver_cost = run_hgs_rr(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes=mandatory_nodes,
        )
        return routes, profit, solver_cost
