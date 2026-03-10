"""
Solution representation for GIHH.

This module defines the Solution class used by the GIHH algorithm.
"""

import random
from typing import Dict, List, Optional

import numpy as np


class Solution:
    """
    Solution representation for Vehicle Routing Problem with Profits.

    Attributes:
        routes: List of routes, where each route is a list of node indices.
        dist_matrix: Distance matrix.
        wastes: Dictionary of node wastes.
        capacity: Vehicle capacity.
        revenue: Revenue multiplier.
        cost_unit: Cost multiplier.
        profit: Total profit (revenue - cost).
        cost: Total routing cost.
        revenue_total: Total revenue from collected waste.
    """

    def __init__(
        self,
        routes: List[List[int]],
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
    ):
        """
        Initialize solution.

        Args:
            routes: List of routes.
            dist_matrix: Distance matrix.
            wastes: Waste dictionary.
            capacity: Vehicle capacity.
            revenue: Revenue per unit.
            cost_unit: Cost per distance unit.
        """
        self.routes = routes
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.revenue = revenue
        self.cost_unit = cost_unit

        self.profit = 0.0
        self.cost = 0.0
        self.revenue_total = 0.0

        self.evaluate()

    def evaluate(self) -> None:
        """Evaluate the solution and update profit, cost, and revenue."""
        total_distance = 0.0
        total_waste = 0.0

        for route in self.routes:
            if len(route) == 0:
                continue

            # Calculate route distance: depot -> route -> depot
            route_distance = self.dist_matrix[0, route[0]]
            for i in range(len(route) - 1):
                route_distance += self.dist_matrix[route[i], route[i + 1]]
            route_distance += self.dist_matrix[route[-1], 0]

            total_distance += route_distance

            # Calculate route waste
            route_waste = sum(self.wastes.get(node, 0.0) for node in route)
            total_waste += route_waste

        self.cost = total_distance * self.cost_unit
        self.revenue_total = total_waste * self.revenue
        self.profit = self.revenue_total - self.cost

    def copy(self) -> "Solution":
        """Create a deep copy of the solution."""
        return Solution(
            routes=[route[:] for route in self.routes],
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue=self.revenue,
            cost_unit=self.cost_unit,
        )

    def is_feasible(self) -> bool:
        """Check if solution respects capacity constraints."""
        for route in self.routes:
            route_load = sum(self.wastes.get(node, 0.0) for node in route)
            if route_load > self.capacity:
                return False
        return True


def create_initial_solution(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    revenue: float,
    cost_unit: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
) -> Solution:
    """
    Create an initial solution using a simple greedy heuristic.

    Args:
        dist_matrix: Distance matrix.
        wastes: Waste dictionary.
        capacity: Vehicle capacity.
        revenue: Revenue per unit.
        cost_unit: Cost per distance unit.
        mandatory_nodes: List of nodes that MUST be visited.
        rng: Random number generator.

    Returns:
        Initial solution.
    """
    if rng is None:
        rng = random.Random()

    n_nodes = len(dist_matrix) - 1
    nodes = list(range(1, n_nodes + 1))

    # Ensure mandatory nodes are visited
    if mandatory_nodes:
        unvisited = set(mandatory_nodes)
    else:
        unvisited = set(nodes)

    routes: List[List[int]] = []
    current_route: List[int] = []
    current_load = 0.0

    # Greedy construction: nearest feasible node
    while unvisited:
        if len(current_route) == 0:
            # Start new route from depot
            last_node = 0
        else:
            last_node = current_route[-1]

        # Find nearest feasible node
        best_node = None
        best_distance = float("inf")

        for node in unvisited:
            node_waste = wastes.get(node, 0.0)
            if current_load + node_waste <= capacity:
                distance = dist_matrix[last_node, node]
                if distance < best_distance:
                    best_distance = distance
                    best_node = node

        if best_node is not None:
            # Add node to current route
            current_route.append(best_node)
            current_load += wastes.get(best_node, 0.0)
            unvisited.remove(best_node)
        else:
            # Cannot add any more nodes to current route
            if len(current_route) > 0:
                routes.append(current_route)
            current_route = []
            current_load = 0.0

            # If still have unvisited nodes, start new route
            if unvisited:
                continue
            else:
                break

    # Add last route if not empty
    if len(current_route) > 0:
        routes.append(current_route)

    return Solution(routes, dist_matrix, wastes, capacity, revenue, cost_unit)
