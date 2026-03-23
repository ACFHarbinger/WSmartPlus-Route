"""
Simulator adapter for the POPMUSIC matheuristic.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.popmusic import POPMUSICConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.popmusic.solver import run_popmusic


@PolicyRegistry.register("popmusic")
class POPMUSICPolicy(BaseRoutingPolicy):
    """
    Adapter for the POPMUSIC (Partial Optimization Metaheuristic Under Special
    Intensification Conditions) matheuristic.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return POPMUSICConfig

    def _get_config_key(self) -> str:
        return "popmusic"

    def _run_solver(
        self,
        sub_dist_matrix: Any,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run POPMUSIC solver.
        """
        routes, total_cost, profit, info = run_popmusic(
            coords=kwargs["coords"],
            must_go=mandatory_nodes,
            distance_matrix=sub_dist_matrix,
            n_vehicles=kwargs.get("n_vehicles", 1),
            subproblem_size=values.get("subproblem_size", 3),
            max_iterations=values.get("max_iterations", 10),
            base_solver=values.get("base_solver", "fast_tsp"),
            base_solver_config=values.get("base_solver_config"),
            cluster_solver=values.get("cluster_solver", "fast_tsp"),
            cluster_solver_config=values.get("cluster_solver_config"),
            initial_solver=values.get("initial_solver", "nearest_neighbor"),
            seed=values.get("seed", 42),
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        # return routes as List[List[int]]
        return routes, profit, total_cost
