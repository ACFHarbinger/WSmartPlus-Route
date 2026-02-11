"""
VRPP Policy Wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies import VRPPConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.adapters.factory import PolicyRegistry
from logic.src.policies.vehicle_routing_problem_with_profits.interface import run_vrpp_optimizer


@PolicyRegistry.register("vrpp")
class VRPPPolicy(BaseRoutingPolicy):
    """
    Agnostic VRPP Policy adapter.
    Delegates to run_vrpp_optimizer.
    """

    def __init__(self, config: Optional[VRPPConfig] = None):
        """Initialize VRPP policy with optional config.

        Args:
            config: Optional VRPPConfig dataclass with solver parameters.
        """
        super().__init__(config)

    def _get_config_key(self) -> str:
        return "vrpp"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_demands: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float]:
        """Not used - VRPP requires specialized execute()."""
        return [], 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the VRPP policy.
        """
        # 1. Extract context
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        bins = kwargs.get("bins")
        distance_matrix = kwargs.get("distance_matrix")
        must_go = kwargs.get("must_go", [])
        config = kwargs.get("config", {})

        # 2. Load parameters and merge with config
        _, _, _, values = self._load_area_params(area, waste_type, config)

        # 3. Handle bins input (WSmart+ simulator passes bins object or amounts)
        amounts = bins
        if hasattr(bins, "c"):
            amounts = bins.c

        # Ensure binsids are present (local 1-based for solver consistency)
        n_bins = len(amounts) if amounts is not None else 0
        binsids = list(range(1, n_bins + 1))

        # 4. Run optimizer
        route, profit, cost = run_vrpp_optimizer(
            bins=amounts,  # type: ignore[arg-type]
            distance_matrix=distance_matrix,  # type: ignore[arg-type]
            param=kwargs.get("param", 0.0),
            media=kwargs.get("media", np.array([])),
            desviopadrao=kwargs.get("desviopadrao", np.array([])),
            values=values,
            binsids=binsids,
            must_go=must_go,
            number_vehicles=kwargs.get("number_vehicles", 1),
            time_limit=int(values.get("time_limit", 60)),
            optimizer=values.get("engine", "gurobi"),
        )

        return route, cost, {"profit": profit}
