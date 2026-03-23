"""
Branch-and-Price (BP) Policy Adapter.

Adapts the core Branch-and-Price solver logic to the systems-agnostic
policy interface, handling parameter mapping, profit calculation, and environment
integration.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BPConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .bp import BranchAndPriceSolver


@PolicyRegistry.register("bp")
class BranchAndPricePolicy(BaseRoutingPolicy):
    """
    Adapter for the Branch-and-Price routing solver.

    This policy implements a column generation method that handles large-scale
    optimization by implicitly enumerating exponentially many variables (routes).

    The algorithm uses:
    1. Set partitioning master problem (select routes to cover nodes)
    2. Pricing subproblem (generate profitable routes via RCSPP)
    3. Column generation (iteratively add routes with positive reduced cost)
    4. Branch-and-bound for integrality (optional Ryan-Foster branching)

    Key advantages:
    - Handles problems with huge solution spaces
    - Provides strong LP bounds (tighter than compact formulations)
    - Scalable to large instances
    """

    def __init__(self, config: Optional[Union[BPConfig, Dict[str, Any]]] = None):
        """Initialize the BP policy adapter.

        Args:
            config: A typed BPConfig dataclass, a raw dictionary for parsing,
                or None to use framework defaults.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration dataclass type for automatic parsing."""
        return BPConfig

    def _get_config_key(self) -> str:
        """Return the unique identification key for this policy's configuration."""
        return "bp"

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
        Run Branch-and-Price solver.

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
        n_nodes = len(sub_dist_matrix) - 1  # Exclude depot from count
        mandatory_set = set(mandatory_nodes)

        # Create Branch-and-Price solver
        solver = BranchAndPriceSolver(
            n_nodes=n_nodes,
            cost_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            revenue_per_kg=revenue,
            cost_per_km=cost_unit,
            mandatory_nodes=mandatory_set,
            max_iterations=values.get("max_iterations", 100),
            max_routes_per_iteration=values.get("max_routes_per_iteration", 10),
            optimality_gap=values.get("optimality_gap", 1e-4),
            use_ryan_foster=values.get("use_ryan_foster_branching", False),
            max_branch_nodes=values.get("max_branch_nodes", 1000),
            use_exact_pricing=values.get("use_exact_pricing", False),
        )

        # Solve
        tour, profit, statistics = solver.solve()

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
            prev = 0
            for node in tour:
                if node != 0:  # Skip depot in middle
                    dist_cost += sub_dist_matrix[prev, node]
                    prev = node
                elif prev != 0:  # Return to depot
                    dist_cost += sub_dist_matrix[prev, 0]
                    prev = 0
            dist_cost *= cost_unit

        # Return routes, profit, and distance cost
        return routes, profit, dist_cost
