"""
Branch-and-Price Policy Adapter.

Integrates the Branch-and-Price solver with the WSmart+ Route policy framework.
Provides both a policy class and a convenience function for running the solver.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .solver import BranchAndPriceSolver


class PolicyBP:
    """
    Branch-and-Price Policy for VRPP.

    Wraps the Branch-and-Price solver to conform to the WSmart+ Route policy interface.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        max_routes_per_iteration: int = 10,
        optimality_gap: float = 1e-4,
        use_ryan_foster_branching: bool = False,
        max_branch_nodes: int = 1000,
        use_exact_pricing: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the Branch-and-Price policy.

        Args:
            max_iterations: Maximum column generation iterations
            max_routes_per_iteration: Maximum routes to generate per pricing call
            optimality_gap: Convergence tolerance for column generation
            use_ryan_foster_branching: Use Ryan-Foster branching (True) or direct IP (False)
            max_branch_nodes: Maximum nodes to explore in branch-and-bound tree
            use_exact_pricing: Use exact DP RCSPP solver (True) or heuristic (False)
            **kwargs: Additional arguments (ignored)
        """
        self.max_iterations = max_iterations
        self.max_routes_per_iteration = max_routes_per_iteration
        self.optimality_gap = optimality_gap
        self.use_ryan_foster_branching = use_ryan_foster_branching
        self.max_branch_nodes = max_branch_nodes
        self.use_exact_pricing = use_exact_pricing
        self.name = "Branch-and-Price"

    def solve(
        self,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        mandatory_nodes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Solve the VRPP using Branch-and-Price.

        Args:
            cost_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 is depot
            wastes: Dictionary mapping node ID to waste volume
            capacity: Vehicle capacity
            revenue_per_kg: Revenue per unit of waste collected
            cost_per_km: Cost per unit of distance traveled
            mandatory_nodes: List of node indices that must be visited

        Returns:
            Dictionary with solution details:
                - tour: Tour as list of node indices
                - profit: Total profit
                - cost: Total distance cost
                - revenue: Total revenue
                - nodes_visited: Set of visited nodes
                - statistics: Solver statistics
        """
        n_nodes = len(cost_matrix) - 1
        mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()

        solver = BranchAndPriceSolver(
            n_nodes=n_nodes,
            cost_matrix=cost_matrix,
            wastes=wastes,
            capacity=capacity,
            revenue_per_kg=revenue_per_kg,
            cost_per_km=cost_per_km,
            mandatory_nodes=mandatory_set,
            max_iterations=self.max_iterations,
            max_routes_per_iteration=self.max_routes_per_iteration,
            optimality_gap=self.optimality_gap,
            use_ryan_foster=self.use_ryan_foster_branching,
            max_branch_nodes=self.max_branch_nodes,
        )

        tour, profit, statistics = solver.solve()

        # Calculate distance and revenue
        total_distance = 0.0
        prev = 0

        for node in tour:
            if node != 0:  # Skip depot
                total_distance += cost_matrix[prev, node]
                prev = node
            elif prev != 0:  # Return to depot
                total_distance += cost_matrix[prev, 0]
                prev = 0

        cost = total_distance * cost_per_km
        total_waste = sum(wastes.get(node, 0.0) for node in tour if node != 0)
        revenue = total_waste * revenue_per_kg

        # Nodes visited (excluding depot)
        nodes_visited = set(node for node in tour if node != 0)

        return {
            "tour": tour,
            "profit": profit,
            "cost": cost,
            "revenue": revenue,
            "distance": total_distance,
            "waste_collected": total_waste,
            "nodes_visited": nodes_visited,
            "statistics": statistics,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Call the solver (convenience method).

        Args:
            *args: Positional arguments passed to solve()
            **kwargs: Keyword arguments passed to solve()

        Returns:
            Solution dictionary
        """
        return self.solve(*args, **kwargs)


def run_branch_and_price(
    cost_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    revenue_per_kg: float,
    cost_per_km: float,
    mandatory_nodes: Optional[List[int]] = None,
    max_iterations: int = 100,
    max_routes_per_iteration: int = 10,
    optimality_gap: float = 1e-4,
    use_ryan_foster_branching: bool = False,
    max_branch_nodes: int = 1000,
) -> Dict[str, Any]:
    """
    Convenience function to run Branch-and-Price solver.

    Args:
        cost_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 is depot
        wastes: Dictionary mapping node ID to waste volume
        capacity: Vehicle capacity
        revenue_per_kg: Revenue per unit of waste collected
        cost_per_km: Cost per unit of distance traveled
        mandatory_nodes: List of node indices that must be visited
        max_iterations: Maximum column generation iterations
        max_routes_per_iteration: Maximum routes to generate per pricing call
        optimality_gap: Convergence tolerance for column generation
        use_ryan_foster_branching: Use Ryan-Foster branching (True) or direct IP (False)
        max_branch_nodes: Maximum nodes to explore in branch-and-bound tree

    Returns:
        Solution dictionary with tour, profit, and statistics
    """
    policy = PolicyBP(
        max_iterations=max_iterations,
        max_routes_per_iteration=max_routes_per_iteration,
        optimality_gap=optimality_gap,
        use_ryan_foster_branching=use_ryan_foster_branching,
        max_branch_nodes=max_branch_nodes,
    )

    return policy.solve(
        cost_matrix=cost_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue_per_kg=revenue_per_kg,
        cost_per_km=cost_per_km,
        mandatory_nodes=mandatory_nodes,
    )
