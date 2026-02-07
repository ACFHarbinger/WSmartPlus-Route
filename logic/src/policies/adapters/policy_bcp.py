"""
BCP Policy Adapter.

Adapts the Branch-Cut-and-Price (BCP) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.branch_cut_and_price import run_bcp

from .factory import PolicyRegistry


@PolicyRegistry.register("bcp")
class BCPPolicy(BaseRoutingPolicy):
    """
    Branch-Cut-and-Price policy class.

    Visits pre-selected 'must_go' bins using exact or heuristic BCP solvers.
    """

    def _get_config_key(self) -> str:
        """Return config key for BCP."""
        return "bcp"

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
        """
        Run BCP solver.

        All nodes in sub_demands are treated as must-go for the solver.

        Returns:
            Tuple of (routes, solver_cost)
        """
        # All subset nodes are must-go (matching HGS strategy)
        must_go_subset: Set[int] = set(sub_demands.keys())

        routes, solver_cost = run_bcp(
            sub_dist_matrix,
            sub_demands,
            capacity,
            revenue,
            cost_unit,
            values,
            must_go_indices=must_go_subset,
            env=kwargs.get("model_env"),
        )
        return routes, solver_cost
