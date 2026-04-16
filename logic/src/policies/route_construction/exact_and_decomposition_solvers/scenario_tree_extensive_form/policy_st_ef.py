"""
Scenario-Tree Extensive Form (ST-EF) Policy Adapter.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.configs.policies.st_ef import ScenarioTreeExtensiveFormConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine import (
    ScenarioTreeExtensiveFormEngine,
)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.tree import (
    ScenarioTree,
)


@RouteConstructorRegistry.register("st_ef")
class ScenarioTreeExtensiveFormPolicy(BaseRoutingPolicy):
    """
    Adapter for the Scenario-Tree Extensive Form (ST-EF) policy.
    """

    @classmethod
    def _config_class(cls):
        return ScenarioTreeExtensiveFormConfig

    def _get_config_key(cls) -> str:
        return "st_ef"

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
        ST-EF bypasses the standard single-day _run_solver via its execute override,
        but we implement this to satisfy the abstract base class.
        """
        return [], 0.0, 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Executes the ST-EF solver.
        """
        # 1. State extraction
        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0)

        # Mapping from framework kwargs to config/params
        cfg: ScenarioTreeExtensiveFormConfig = self.config

        # 2. Build Scenario Tree
        customers = list(range(1, distance_matrix.shape[0]))
        tree = ScenarioTree(
            num_days=cfg.num_days,
            num_realizations=cfg.num_realizations,
            customers=customers,
            mean_increment=cfg.mean_increment,
            seed=cfg.seed,
        )

        # 3. Instantiate and run Engine
        engine = ScenarioTreeExtensiveFormEngine(
            tree=tree,
            distance_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            waste_weight=cfg.waste_weight,
            cost_weight=cfg.cost_weight,
            overflow_penalty=cfg.overflow_penalty,
            time_limit=cfg.time_limit,
            mip_gap=cfg.mip_gap,
            use_mtz=cfg.use_mtz,
            verbose=False,  # Toggle via framework logging if needed
        )

        # 4. Solve and return Day 1 action
        route, expected_val = engine.solve()

        if not route:
            # Fallback if no feasible route found
            return [0, 0], 0.0, {"expected_value": expected_val}

        # Calculate actual cost of the extracted route for return
        actual_cost = 0.0
        for i in range(len(route) - 1):
            actual_cost += distance_matrix[route[i], route[i + 1]]

        return route, float(actual_cost), {"expected_value": expected_val}
