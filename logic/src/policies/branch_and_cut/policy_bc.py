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

from .bc import BranchAndCutSolver
from .vrpp_model import VRPPModel


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

        # Create solver with configuration parameters
        solver = BranchAndCutSolver(
            model=model,
            time_limit=values.get("time_limit", 300.0),
            mip_gap=values.get("mip_gap", 0.0),
            max_cuts_per_round=values.get("max_cuts_per_round", 50),
            use_heuristics=values.get("use_heuristics", True),
            verbose=values.get("verbose", False),
            profit_aware_operators=kwargs.get("profit_aware_operators", False),
            vrpp=kwargs.get("vrpp", False),
        )

        # Solve
        tour, profit, stats = solver.solve()

        # Convert tour to routes format (single route for single vehicle)
        # Remove depot (0) from tour for route representation
        if tour and len(tour) > 2:  # More than just depot visits
            route = [node for node in tour if node != 0]
            routes = [route] if route else []
        else:
            routes = []

        # Calculate actual distance cost
        dist_cost = 0.0
        if tour:
            for i in range(len(tour) - 1):
                dist_cost += sub_dist_matrix[tour[i]][tour[i + 1]]
            dist_cost *= cost_unit

        # Return routes, profit, and distance cost
        return routes, profit, dist_cost
