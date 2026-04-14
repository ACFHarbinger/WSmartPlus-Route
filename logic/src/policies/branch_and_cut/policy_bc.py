"""
Branch-and-Cut (BC) Policy Adapter.

Adapts the core Branch-and-Cut solver logic to the systems-agnostic
policy interface, handling parameter mapping, profit calculation, and environment
integration.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BCConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.other.branching_solvers.vrpp_model import VRPPModel

from .bc import BranchAndCutSolver
from .params import BCParams


@PolicyRegistry.register("bc")
class BranchAndCutPolicy(BaseRoutingPolicy):
    """
    Adapter for the Branch-and-Cut routing solver.

    This policy implements an exact optimization method that combines:
    - Cutting planes to strengthen LP relaxations
    - Branch-and-bound for integer optimization
    - Separation algorithms for subtour elimination and capacity constraints

    As an exact solver, it provides provable optimality guarantees within
    the specified MIP gap and time limit.
    """

    def __init__(self, config: Optional[Union[BCConfig, Dict[str, Any]]] = None):
        """Initialize the BC policy adapter.

        Args:
            config: A typed BCConfig dataclass, a raw dictionary for parsing,
                or None to use framework defaults.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration dataclass type for automatic parsing."""
        return BCConfig

    def _get_config_key(self) -> str:
        """Return the unique identification key for this policy's configuration."""
        return "bc"

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
        Run Branch-and-Cut solver.

        All nodes in mandatory_nodes are treated as must-go for the solver.
        In VRPP mode, additional nodes from sub_wastes might be collected if profitable.

        Args:
            sub_dist_matrix: The localized distance matrix for candidates.
            sub_wastes: Current fill levels for available customer nodes.
            capacity: The maximum payload of the vehicle.
            revenue: Expected revenue per unit of waste collected.
            cost_unit: Cost per unit of distance traveled.
            values: Merged configuration dictionary from simulation and YAML.
            mandatory_nodes: Indices of nodes that MUST be visited.
            **kwargs: Additional solver parameters.

        Returns:
            A tuple containing (List of routes, total profit, total travel cost).
        """
        n_nodes = len(sub_dist_matrix)

        # Build VRPP model
        model = VRPPModel(
            n_nodes=n_nodes,
            cost_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            revenue_per_kg=revenue,
            cost_per_km=cost_unit,
            mandatory_nodes=set(mandatory_nodes),
        )

        # Create solver with standardized configuration parameters
        params = BCParams.from_config(values)
        if "profit_aware_operators" in kwargs:
            params.profit_aware_operators = kwargs["profit_aware_operators"]
        if "vrpp" in kwargs:
            params.vrpp = kwargs["vrpp"]

        solver = BranchAndCutSolver(
            model=model,
            params=params,
        )

        # Solve
        routes, profit, stats = solver.solve()

        # Compute travel cost from the solver's objective value.
        # Objective = travel_cost - waste_collected (Gurobi minimizes)
        # profit = -ObjVal = waste_collected - travel_cost
        # => travel_cost = waste_collected - profit

        # 1. Total waste collected (revenue side)
        total_revenue = sum(model.get_node_profit(i) for route in routes for i in route)

        # 2. Derive distance cost: dist_cost = travel_cost
        # travel_cost = waste_collected - profit (already in monetary units)
        dist_cost = max(0.0, total_revenue - profit)

        return routes, profit, dist_cost
