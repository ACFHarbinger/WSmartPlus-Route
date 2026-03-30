"""
GIHH Policy Adapter.

Adapts the Hyper-Heuristic with Two Guidance Indicators (GIHH) logic
to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import GIHHConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .gihh import GIHHSolver
from .params import GIHHParams


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
        params = GIHHParams(
            time_limit=float(values.get("time_limit", 60.0)),
            max_iterations=int(values.get("max_iterations", 1000)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seg=values.get("seg", 80),
            alpha=values.get("alpha", 0.5),
            beta=values.get("beta", 0.4),
            gamma=values.get("gamma", 0.1),
            min_prob=values.get("min_prob", 0.05),
            nonimp_threshold=values.get("nonimp_threshold", 150),
        )

        solver = GIHHSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )
        return solver.solve()
