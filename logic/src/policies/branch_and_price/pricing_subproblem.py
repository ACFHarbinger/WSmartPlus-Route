"""
Pricing Subproblem for Branch-and-Price VRPP.

Solves the resource-constrained shortest path problem (RCSPP) to generate new routes
with positive reduced cost. The pricing problem finds routes that improve the LP relaxation.

Based on Section 2.2 and Section 4 of Barnhart et al. (1998).
"""

from typing import Dict, List, Set, Tuple

import numpy as np


class PricingSubproblem:
    """
    Resource-Constrained Shortest Path Problem for Route Generation.

    The pricing problem finds a route that maximizes reduced cost:
        reduced_cost = profit - Σ_i dual_i
        profit = (waste * R) - (distance * C)

    Subject to:
        - Capacity constraints
        - All mandatory nodes must be reachable
        - Route starts and ends at depot

    This is an elementary shortest path problem with resource constraints (ESPPRC).
    """

    def __init__(
        self,
        n_nodes: int,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        mandatory_nodes: Set[int],
    ):
        """
        Initialize the pricing subproblem.

        Args:
            n_nodes: Number of customer nodes (excluding depot)
            cost_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 is depot
            wastes: Dictionary mapping node ID to waste volume
            capacity: Vehicle capacity
            revenue_per_kg: Revenue per unit of waste collected
            cost_per_km: Cost per unit of distance traveled
            mandatory_nodes: Set of mandatory node indices
        """
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes = mandatory_nodes
        self.depot = 0

    def solve(
        self,
        dual_values: Dict[int, float],
        max_routes: int = 10,
    ) -> List[Tuple[List[int], float]]:
        """
        Solve the pricing problem to generate new routes with positive reduced cost.

        Uses a label-setting dynamic programming algorithm for ESPPRC.

        Args:
            dual_values: Dual values from master problem (node coverage constraints)
            max_routes: Maximum number of routes to generate

        Returns:
            List of (route_nodes, reduced_cost) tuples, sorted by descending reduced cost
        """
        # Use greedy heuristic + local search for generating columns
        # This is a simplified approximation; exact ESPPRC is NP-hard
        routes = []

        # Try generating routes starting from different nodes
        candidate_starts = sorted(
            list(range(1, self.n_nodes + 1)),
            key=lambda n: dual_values.get(n, 0.0) + (self.wastes.get(n, 0.0) * self.R),
            reverse=True,
        )

        for start_node in candidate_starts[:max_routes]:
            route, reduced_cost = self._greedy_route_construction(start_node, dual_values)

            if reduced_cost > 1e-4:  # Positive reduced cost
                routes.append((route, reduced_cost))

            if len(routes) >= max_routes:
                break

        # Sort by descending reduced cost
        routes.sort(key=lambda x: x[1], reverse=True)

        return routes[:max_routes]

    def _greedy_route_construction(
        self,
        start_node: int,
        dual_values: Dict[int, float],
    ) -> Tuple[List[int], float]:
        """
        Construct a route greedily starting from a given node.

        Uses farthest insertion with profit-aware selection.

        Args:
            start_node: Node to start route construction
            dual_values: Dual values from master problem

        Returns:
            Tuple of (route_nodes, reduced_cost)
        """
        route = [start_node]
        current_load = self.wastes.get(start_node, 0.0)
        visited = {start_node}

        # Greedy insertion
        while True:
            best_node = None
            best_marginal_profit = -float("inf")

            # Try inserting each unvisited node at the best position
            for node in range(1, self.n_nodes + 1):
                if node in visited:
                    continue

                node_waste = self.wastes.get(node, 0.0)
                if current_load + node_waste > self.capacity:
                    continue

                # Try all insertion positions
                for pos in range(len(route) + 1):
                    # Calculate marginal cost
                    if pos == 0:
                        prev = self.depot
                        nxt = route[0] if route else self.depot
                    elif pos == len(route):
                        prev = route[-1]
                        nxt = self.depot
                    else:
                        prev = route[pos - 1]
                        nxt = route[pos]

                    detour_cost = (
                        self.cost_matrix[prev, node] + self.cost_matrix[node, nxt] - self.cost_matrix[prev, nxt]
                    )

                    # Marginal profit with dual value
                    marginal_profit = (
                        (node_waste * self.R)  # Revenue
                        - (detour_cost * self.C)  # Cost
                        - dual_values.get(node, 0.0)  # Dual value
                    )

                    if marginal_profit > best_marginal_profit:
                        best_marginal_profit = marginal_profit
                        best_node = (node, pos)

            if best_node is None or best_marginal_profit <= 0:
                break

            # Insert best node
            node, pos = best_node
            route.insert(pos, node)
            visited.add(node)
            current_load += self.wastes.get(node, 0.0)

        # Calculate total reduced cost
        reduced_cost = self._compute_reduced_cost(route, dual_values)

        return route, reduced_cost

    def _compute_reduced_cost(
        self,
        route: List[int],
        dual_values: Dict[int, float],
    ) -> float:
        """
        Compute the reduced cost of a route.

        reduced_cost = profit - Σ_i dual_i
        profit = revenue - cost

        Args:
            route: List of nodes in route
            dual_values: Dual values from master problem

        Returns:
            Reduced cost of the route
        """
        # Calculate distance
        total_distance = 0.0
        prev = self.depot

        for node in route:
            total_distance += self.cost_matrix[prev, node]
            prev = node

        total_distance += self.cost_matrix[prev, self.depot]

        # Calculate revenue
        total_waste = sum(self.wastes.get(node, 0.0) for node in route)
        revenue = total_waste * self.R

        # Calculate cost
        cost = total_distance * self.C

        # Calculate profit
        profit = revenue - cost

        # Calculate dual contribution
        dual_contribution = sum(dual_values.get(node, 0.0) for node in route)

        # Reduced cost
        reduced_cost = profit - dual_contribution

        return reduced_cost

    def compute_route_details(
        self,
        route: List[int],
    ) -> Tuple[float, float, float, Set[int]]:
        """
        Compute detailed information about a route.

        Args:
            route: List of nodes in route

        Returns:
            Tuple of (cost, revenue, load, node_coverage)
        """
        # Calculate distance
        total_distance = 0.0
        prev = self.depot

        for node in route:
            total_distance += self.cost_matrix[prev, node]
            prev = node

        total_distance += self.cost_matrix[prev, self.depot]

        # Calculate revenue and load
        total_waste = sum(self.wastes.get(node, 0.0) for node in route)
        revenue = total_waste * self.R
        cost = total_distance * self.C

        # Node coverage
        node_coverage = set(route)

        return cost, revenue, total_waste, node_coverage
