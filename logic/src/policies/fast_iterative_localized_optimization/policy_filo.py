"""
FILO Policy Adapter.

Adapts the Fast Iterative Localized Optimization (FILO) logic to the agnostic interface.
"""

from dataclasses import fields as dc_fields
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import FILOConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.fast_iterative_localized_optimization.filo import FILOSolver
from logic.src.policies.fast_iterative_localized_optimization.params import FILOParams


@PolicyRegistry.register("filo")
class FILOPolicy(BaseRoutingPolicy):
    """
    FILO policy class.

    Visits pre-selected 'must_go' bins using Fast Iterative Localized Optimization.
    """

    def __init__(self, config: Optional[Union[FILOConfig, Dict[str, Any]]] = None):
        """Initialize FILO policy with optional config.

        Args:
            config: FILOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return FILOConfig

    def _get_config_key(self) -> str:
        """Return config key for FILO."""
        return "filo"

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
        Run FILO solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        # Build configuration and extract structured params
        valid_fields = {f.name for f in dc_fields(FILOConfig)}
        filtered_values = {k: v for k, v in values.items() if k in valid_fields}
        filo_config = FILOConfig(**filtered_values)
        params = FILOParams.from_config(filo_config)

        solver = FILOSolver(
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
