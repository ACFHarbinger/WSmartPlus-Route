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
            move_operators=values.get(
                "move_operators",
                [
                    "swap_intra",
                    "relocate_intra",
                    "two_opt_intra",
                    "swap_inter",
                    "relocate_inter",
                    "two_opt_star",
                    "exchange_10",
                    "exchange_11",
                    "exchange_21",
                ],
            ),
            perturbation_operators=values.get(
                "perturbation_operators", ["random_removal", "string_removal", "route_removal"]
            ),
            iri_weight=values.get("iri_weight", 0.6),
            tbi_weight=values.get("tbi_weight", 0.4),
            learning_rate=values.get("learning_rate", 0.1),
            memory_size=values.get("memory_size", 50),
            epsilon=values.get("epsilon", 0.2),
            epsilon_decay=values.get("epsilon_decay", 0.995),
            min_epsilon=values.get("min_epsilon", 0.01),
            accept_equal=values.get("accept_equal", True),
            accept_worse_prob=values.get("accept_worse_prob", 0.05),
            acceptance_decay=values.get("acceptance_decay", 0.99),
            iri_window=values.get("iri_window", 20),
            tbi_window=values.get("tbi_window", 20),
            restarts=values.get("restarts", 1),
            restart_threshold=values.get("restart_threshold", 100),
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
