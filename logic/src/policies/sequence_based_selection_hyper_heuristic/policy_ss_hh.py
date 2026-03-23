"""
SS-HH Policy Adapter.

Adapts the Sequence-based Selection Hyper-Heuristic (SS-HH) solver to the
agnostic BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ss_hh import SSHHConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.sequence_based_selection_hyper_heuristic.params import SSHHParams
from logic.src.policies.sequence_based_selection_hyper_heuristic.solver import SSHHSolver


@PolicyRegistry.register("ss_hh")
class SSHHPolicy(BaseRoutingPolicy):
    """
    SS-HH policy class.

    Visits bins using the online-learning Sequence-based Selection Hyper-Heuristic.
    """

    def __init__(self, config: Optional[Union[SSHHConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SSHHConfig

    def _get_config_key(self) -> str:
        return "ss_hh"

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
        params = SSHHParams(
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
            threshold_infeasible=float(values.get("threshold_infeasible", 0.001)),
            threshold_feasible_base=float(values.get("threshold_feasible_base", 0.0001)),
            threshold_decay_rate=float(values.get("threshold_decay_rate", 0.01)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = SSHHSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
