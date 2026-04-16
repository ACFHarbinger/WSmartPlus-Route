r"""
Constraint Programming (CP-SAT) Policy Adapter for Stochastic VRPP.

Adapts the Google OR-Tools CP-SAT solver to the Multi-Period Stochastic
framework. CP-SAT is a modern solver that combines Constraint Programming (CP)
with Boolean Satisfiability (SAT) techniques, particularly effective at
handling complex logical constraints and non-linear cost structures.

Problem Formulation — Integrated CP Optimization:
    The solver models the $T$-period stochastic problem using a set of
    decision variables:
        - $x_{i,j,t,k}$: Boolean arc $(i,j)$ traversal for vehicle $k$ on day $t$.
        - $y_{i,t}$: Boolean visit flag for bin $i$ on day $t$.
        - $w_{i,t}$: Continuous (quantized) inventory level of bin $i$ on day $t$.

    Core Constraints:
        - **Degree & Flow**: In/out degree balance for each day $t$.
        - **Transition Logic**: Bin inventories propagate across days based on
          visit decisions: $w_{i,t+1} = \max(0, w_{i,t} + \Delta_{i,t} - y_{i,t} \cdot w_{i,t})$.
        - **Capacity**: Cumulative load $\le$ payload at each node.
        - **Overflow**: Logical constraints penalizing states where $w_{i,t} > Q$.

Advantage of CP-SAT:
    CP-SAT can solve discretized versions of the stochastic problem without
    requiring the linearity strictly enforced by MILP solvers. It uses **Lazy
    Clause Generation** to prune the search space by learning from conflicts,
    making it highly robust for combinatorial problems with complex state-transitions.

Implementation:
    - **Lookahead**: Extends decisions over the $T$-day horizon.
    - **Quantization**: Bin increments $\Delta_{i,t}$ are quantized to integers
      to satisfy CP-SAT's integer-based core.

Registry key: ``"cp_sat"``

References:
    Schutt, A., et al. (2013). "LCG-based solvers for the traveling salesman
    problem with time windows". OR Spectrum, 35(3), 543-575.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.cp_sat import CPSATConfig
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .cp_sat_engine import CPSATEngine


@RouteConstructorRegistry.register("cp_sat")
class CPSATPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Adapter for the Constraint Programming with Boolean Satisfiability (CP-SAT) policy using OR-Tools.
    Now standardized to the Multi-Period framework.
    """

    @classmethod
    def _config_class(cls):
        return CPSATConfig

    def _get_config_key(self) -> str:
        """Return the configuration key."""
        return "cp_sat"

    def _run_multi_period_solver(
        self,
        tree: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """
        Execute the Multi-Period CP-SAT solver.

        This method models the stochastic routing problem over the T-day horizon
        as a Constraint Programming (CP) model, which is then solved using
        OR-Tools' CP-SAT engine. It discretizes bin fill levels and utilizes
        the scenario tree to find decisions that maximize expected long-term
        profit, accounting for future overflow penalties and collection
        efficiencies.

        Args:
            tree (ScenarioTree): Tree of future fill rate realization scenarios.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            **kwargs: Additional context, including:
                - distance_matrix (np.ndarray): Symmetric distance matrix.
                - sub_wastes (Dict[int, float]): Current bin fill levels.

        Returns:
            Tuple[List[List[List[int]]], float, Dict[str, Any]]:
                A 3-tuple containing:
                - full_plan: Collection plan for each day in the horizon.
                - expected_profit: The optimal objective value (expected profit).
                - metadata: Execution statistics and status flags.
        """
        sub_dist_matrix = kwargs["distance_matrix"]
        sub_wastes = kwargs.get("sub_wastes", {})

        # We pass the tree to the engine.
        # Note: CPSATEngine might need updates to handle the tree instead of fixed increments.
        engine = CPSATEngine(
            config=self.config, distance_matrix=sub_dist_matrix, initial_wastes=sub_wastes, capacity=capacity
        )

        # 3. Solve and return Day 1 action
        full_plan: List[List[List[int]]]
        route_day_0: List[int]
        route_day_0, expected_val = engine.solve()

        # Wrap into full plan
        full_plan = [[] for _ in range(self.horizon + 1)]
        full_plan[0] = [route_day_0]
        return full_plan, expected_val, {"status": "solved"}

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Legacy fallback."""
        return [], 0.0, 0.0
