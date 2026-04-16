from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from logic.src.configs.policies.lbbd import LBBDConfig
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

if TYPE_CHECKING:
    from logic.src.pipeline.simulations.bins.prediction import ScenarioTree

from .lbbd_engine import LBBDEngine


@RouteConstructorRegistry.register("lbbd")
class LBBDPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Adapter for the Logic-Based Benders Decomposition (LBBD) policy.
    Supports multi-period stochastic decision making via ScenarioTree.
    """

    @classmethod
    def _config_class(cls):
        return LBBDConfig

    def _get_config_key(cls) -> str:
        return "lbbd"

    def _run_multi_period_solver(
        self,
        tree: ScenarioTree,
        capacity: float,
        revenue: float,
        cost_unit: float,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """
        Execute the Logic-Based Benders Decomposition (LBBD) solver logic.

        This method decomposes the multi-period stochastic routing problem into
        a Master Problem (scheduling assignment) and several Sub-problems
        (specific daily routing). It utilizes logic-based cuts to communicate
        between the scheduling level and the routing level, ensuring that
        Day 0 routes are optimized with respect to both immediate profit and
        expected longer-term resource availability across scenarios.

        Args:
            tree (ScenarioTree): Tree of future fill rate realization scenarios.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            **kwargs: Additional context, including:
                - distance_matrix (np.ndarray): Symmetric distance matrix.

        Returns:
            Tuple[List[List[List[int]]], float, Dict[str, Any]]:
                A 3-tuple containing:
                - full_plan: Collection plan (nested list by day and vehicle).
                - profit: The optimal objective value (cumulative net profit).
                - stats: Execution statistics and decomposition iterations.
        """
        distance_matrix = kwargs["distance_matrix"]
        cfg: LBBDConfig = self.config

        # Instantiate and run Engine
        engine = LBBDEngine(config=cfg, distance_matrix=distance_matrix, tree=tree, capacity=capacity)

        # Solve for the full horizon T
        full_plan, profit, stats = engine.solve()

        return full_plan, profit, stats

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
        Satisfies the abstract base class, but LBBD uses _run_multi_period_solver.
        """
        return [], 0.0, 0.0
