"""
BPC Policy Adapter.

Adapts the Branch-and-Price-and-Cut (BPC) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BPCConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.branch_price_cut import run_bpc


@PolicyRegistry.register("bpc")
class BPCPolicy(BaseRoutingPolicy):
    """
    Branch-and-Price-and-Cut policy class.

    Visits pre-selected 'must_go' bins using exact or heuristic BPC solvers.
    """

    def __init__(self, config: Optional[Union[BPCConfig, Dict[str, Any]]] = None):
        """Initialize BPC policy with optional config.

        Args:
            config: BPCConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return BPCConfig

    def _get_config_key(self) -> str:
        """Return config key for BPC."""
        return "bpc"

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
        Run BPC solver.

        All nodes in mandatory_nodes are treated as must-go for the solver.
        In VRPP mode, additional nodes from sub_wastes might be collected if profitable.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        # Convert local mandatory indices to a set of must-go nodes for the solver
        must_go_indices: Set[int] = set(mandatory_nodes)

        routes, solver_cost = run_bpc(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            must_go_indices=must_go_indices,
            env=kwargs.get("model_env"),
        )

        # Compute profit: collected revenue - distance cost
        visited = {n for route in routes for n in route}
        collected_revenue = sum(sub_wastes.get(n, 0) * revenue for n in visited)
        dist_cost = 0.0
        for route in routes:
            path = [0] + route + [0]
            for i in range(len(path) - 1):
                dist_cost += sub_dist_matrix[path[i]][path[i + 1]]
        profit = collected_revenue - dist_cost * cost_unit

        return routes, profit, solver_cost
