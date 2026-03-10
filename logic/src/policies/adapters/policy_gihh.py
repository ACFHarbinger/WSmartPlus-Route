"""
GIHH Policy Adapter.

Adapts the Hyper-Heuristic with Two Guidance Indicators (GIHH) logic
to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import GIHHConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.guided_indicators_hyper_heuristic import run_gihh

from .factory import PolicyRegistry


@PolicyRegistry.register("gihh")
class GIHHPolicy(BaseRoutingPolicy):
    """
    Hyper-Heuristic with Two Guidance Indicators policy class.

    Uses IRI (Improvement Rate Indicator) and TBI (Time-based Indicator)
    to adaptively select low-level heuristics during search.
    """

    def __init__(self, config: Optional[Union[GIHHConfig, Dict[str, Any]]] = None):
        """Initialize GIHH policy with optional config.

        Args:
            config: GIHHConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GIHHConfig

    def _get_config_key(self) -> str:
        """Return config key for GIHH."""
        return "gihh"

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
        Run GIHH solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        routes, profit, solver_cost = run_gihh(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes=mandatory_nodes,
        )
        return routes, profit, solver_cost
