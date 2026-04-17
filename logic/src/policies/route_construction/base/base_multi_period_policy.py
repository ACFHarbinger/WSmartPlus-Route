"""
Base Multi-Period Routing Policy.

Provides a template for routing policies that perform T-period optimization
over a scenario tree, rather than myopic single-day routing.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.pipeline.simulations.bins.prediction import ScenarioTree

if TYPE_CHECKING:
    pass

from .base_routing_policy import BaseRoutingPolicy


class BaseMultiPeriodRoutingPolicy(BaseRoutingPolicy):
    """
    Base class for policies optimized for stochastic multi-period routing (SIRP).

    This class provides utilities for:
    - Horizon management (T days).
    - Scenario tree traversal.
    - Future mandatory node prediction.
    - Overflow (stockout) penalty calculation.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.horizon: int = getattr(self.config, "horizon", 7)
        self.stockout_penalty: float = getattr(self.config, "stockout_penalty", 500.0)

    def _get_scenario_tree(self, kwargs: Dict[str, Any]) -> ScenarioTree:
        """Extract the scenario tree from the simulation context."""
        tree = kwargs.get("scenario_tree")
        if tree is None:
            raise ValueError(f"{type(self).__name__} requires a ScenarioTree in the context.")
        return tree

    def _predict_mandatory_nodes_for_horizon(
        self, tree: ScenarioTree, initial_mandatory: List[int]
    ) -> Dict[int, List[int]]:
        """
        Predict mandatory nodes for each day in the horizon based on the tree.

        Returns:
            Dict[day, list_of_bin_ids]
        """
        mandatory_map: Dict[int, List[int]] = {0: initial_mandatory}

        # We traverse the main scenario path (or all branches) to identify
        # when bins are expected to overflow (>100%).
        for t in range(1, tree.horizon + 1):
            scenarios = tree.get_scenarios_at_day(t)
            # For simplicity, we mark a node as mandatory if it overflows in ANY scenario at day t,
            # but hasn't been collected in previous days t' < t.
            # Real solvers might handle this more robustly.
            day_mandatory = []
            for scenario in scenarios:
                overflow_bins = np.where(scenario.wastes >= 100.0)[0] + 1  # 1-based
                for b_id in overflow_bins:
                    if b_id not in day_mandatory:
                        # Check if it was already mandatory in a previous day
                        already_seen = False
                        for prev_t in range(t):
                            if b_id in mandatory_map.get(prev_t, []):
                                already_seen = True
                                break
                        if not already_seen:
                            day_mandatory.append(int(b_id))

            mandatory_map[t] = day_mandatory

        return mandatory_map

    def _calculate_stockout_costs(self, inventory_sequence: np.ndarray) -> float:
        """
        Calculate penalties for waste overflows (>100%).

        Args:
            inventory_sequence: array of shape (N, T) with fill levels.

        Returns:
            Total penalty cost.
        """
        overflows = np.maximum(0, inventory_sequence - 100.0)
        return float(np.sum(overflows) * self.stockout_penalty)

    @abstractmethod
    def _run_multi_period_solver(
        self,
        tree: ScenarioTree,
        capacity: float,
        revenue: float,
        cost_unit: float,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """
        Solve the T-period optimization problem.

        Returns:
            Tuple of (plan[day][route][node], expected_profit, metadata).
        """
        pass

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
        Dummy implementation of the single-period solver.

        Multi-period policies override the execute() method directly (or rely on
        the BaseMultiPeriodRoutingPolicy override) to call _run_multi_period_solver,
        so this method is never invoked in the SIRP workflow.
        """
        return [], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the multi-period routing policy with scenario-based lookahead.

        Unlike standard myopic policies, multi-period policies evaluate
        decisions over a $T$-day horizon using a scenario tree. This allows
        the solver to anticipate future bin overflows and optimize the
        expected long-term profit.

        The execution flow is:
        1. Environment & Parameter Extraction: Load area-specific constants
           and the scenario tree from the simulation context.
        2. Multi-Period Optimization: Solve the $T$-period problem to generate
           a full collection plan for the entire horizon.
        3. Decision Extraction: Extract the optimized routes for Day 0 (the
           current physical stage) to be returned to the simulator.
        4. Context Update: Update the `MultiDayContext` with the snapshots of
           the full plan for future reference.

        Args:
            **kwargs: Context dictionary containing:
                - scenario_tree (ScenarioTree): Tree of future bin fill rate realizations.
                - distance_matrix (np.ndarray): Localized distance matrix.
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
                A 5-tuple containing:
                - tour: Optimized collection routes for Day 0.
                - cost: Total travel cost of the Day 0 routes.
                - profit: Expected cumulative profit over the $T$-day horizon
                  (as calculated by the multi-period solver).
                - search_context: The propagated or initialized search context.
                - multi_day_context: The updated multi-day state metadata.
        """
        # 1. Standard parameter loading from BaseRoutingPolicy
        capacity, revenue, cost_unit, values = self._load_area_params(
            kwargs.get("area", "Rio Maior"),
            kwargs.get("waste_type", "plastic"),
            kwargs.get("config", {}),
        )

        tree = self._get_scenario_tree(kwargs)
        multi_day_ctx: Optional[MultiDayContext] = kwargs.get("multi_day_context")
        search_ctx: Optional[SearchContext] = kwargs.get("search_context")

        # 2. Run Multi-Period Solver
        # This returns a full plan for T days
        full_plan, profit, solver_metadata = self._run_multi_period_solver(
            tree=tree,
            capacity=capacity,
            revenue=revenue,
            cost_unit=cost_unit,
            **kwargs,
        )

        # 3. Extract Day 0 Tour
        day_0_tours = full_plan[0] if full_plan else [[0, 0]]
        # If it's a single vehicle solver, return as a flat list if needed
        # but the interface allows List[List[int]].
        tour = day_0_tours if len(day_0_tours) > 1 else day_0_tours[0]

        # 4. Compute Day 0 Cost
        distance_matrix = kwargs["distance_matrix"]
        cost = 0.0
        if tour and isinstance(tour[0], list):
            # Narrow to List[List[int]] for multiple vehicles
            for r in cast(List[List[int]], tour):
                for i in range(len(r) - 1):
                    cost += distance_matrix[r[i], r[i + 1]]
        elif tour:
            # Narrow to List[int] for single vehicle
            tour_flat = cast(List[int], tour)
            for i in range(len(tour_flat) - 1):
                cost += distance_matrix[tour_flat[i], tour_flat[i + 1]]

        # 5. Update MultiDayContext
        if multi_day_ctx:
            multi_day_ctx = multi_day_ctx.update(
                full_plan_snapshot=full_plan, extra={**multi_day_ctx.extra, **solver_metadata}
            )

        return tour, cost, profit, search_ctx, multi_day_ctx
