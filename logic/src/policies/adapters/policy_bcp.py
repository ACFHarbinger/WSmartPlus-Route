"""
BCP Policy Adapter.

Adapts the Branch-Cut-and-Price (BCP) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BCPConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.branch_cut_and_price import run_bcp

from .factory import PolicyRegistry


@PolicyRegistry.register("bcp")
class BCPPolicy(BaseRoutingPolicy):
    """
    Branch-Cut-and-Price policy class.

    Visits pre-selected 'must_go' bins using exact or heuristic BCP solvers.
    """

    def __init__(self, config: Optional[Union[BCPConfig, Dict[str, Any]]] = None):
        """Initialize BCP policy with optional config.

        Args:
            config: BCPConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return BCPConfig

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
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float]:
        """
        Run BCP solver.

        All nodes in mandatory_nodes are treated as must-go for the solver.
        In VRPP mode, additional nodes from sub_demands might be collected if profitable.

        Returns:
            Tuple of (routes, solver_cost)
        """
        # Convert local mandatory indices to a set of must-go nodes for the solver
        must_go_indices: Set[int] = set(mandatory_nodes)

        routes, solver_cost = run_bcp(
            sub_dist_matrix,
            sub_demands,
            capacity,
            revenue,
            cost_unit,
            values,
            must_go_indices=must_go_indices,
            env=kwargs.get("model_env"),
        )
        return routes, solver_cost
