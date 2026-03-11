"""
Ruin and Recreate operator for Fast Iterative Localized Optimization (FILO).
"""

from typing import Dict, List, Tuple

import numpy as np


class RuinAndRecreate:
    """
    Operator that applies localized Ruin & Recreate (shaking) to a solution.
    Based on the FILO algorithm.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        rng: np.random.Generator,
    ):
        """Initialize RuinAndRecreate operator."""
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.rng = rng

        # Pre-compute neighbors list (omitting depot) sorted by distance for each node
        self.neighbors: List[List[int]] = []
        for i in range(len(self.d)):
            # Sort neighbors by distance, exclude depot (index 0) and self
            sorted_neighbors = np.argsort(self.d[i])
            filtered = [int(n) for n in sorted_neighbors if n != 0 and n != i]
            self.neighbors.append(filtered)

    def apply(  # noqa: C901
        self,
        routes: List[List[int]],
        omega: List[int],
        all_customers: List[int],
    ) -> Tuple[List[List[int]], int, List[int]]:
        """
        Apply Ruin and Recreate to the current routing solution.

        Args:
            routes: Current routes. Will NOT be modified in-place; deep copy used.
            omega: Array/List of shaking intensities per node (1-indexed for customers).
            all_customers: List of all valid customer IDs.

        Returns:
            Tuple of (new_routes, seed_node, ruined_customers)
        """
        # Deep copy to avoid mutating the original
        working_routes = [r[:] for r in routes if r]

        # Select a random seed node
        seed = self.rng.choice(all_customers)
        # Omega might be 1-indexed based or sized up to len(distance_matrix)
        N = omega[seed]

        removed = []
        visited_routes = set()
        curr = seed

        # Helpers for routing queries
        def _build_route_map() -> Dict[int, Tuple[int, int]]:
            """Build map from customer ID -> (route_index, position_index)."""
            rmap = {}
            for r_idx, route in enumerate(working_routes):
                for p_idx, node in enumerate(route):
                    rmap[node] = (r_idx, p_idx)
            return rmap

        route_map = _build_route_map()

        for _ in range(N):
            if curr == 0 or curr not in route_map:
                break

            r_idx, p_idx = route_map[curr]
            route = working_routes[r_idx]

            removed.append(curr)
            visited_routes.add(r_idx)

            next_node = -1

            if len(route) > 1 and self.rng.random() < 0.5:
                # Move within the current route
                if self.rng.random() < 0.5:
                    # Next vertex
                    next_p = p_idx + 1
                    if next_p >= len(route):
                        next_p = 0 if len(route) > 1 else -1
                    if next_p != -1 and route[next_p] == curr:
                        # Should not happen if len > 1, but safeguard
                        next_p = (next_p + 1) % len(route)
                    next_node = route[next_p] if next_p != -1 else -1
                else:
                    # Prev vertex
                    prev_p = p_idx - 1
                    if prev_p < 0:
                        prev_p = len(route) - 1 if len(route) > 1 else -1
                    if prev_p != -1 and route[prev_p] == curr:
                        prev_p = (prev_p - 1) % len(route)
                    next_node = route[prev_p] if prev_p != -1 else -1
            else:
                # Jump to neighbor route
                if self.rng.random() < 0.5:
                    # Unvisited routes
                    for neighbor in self.neighbors[curr]:
                        if neighbor not in route_map:
                            continue
                        neighbor_r_idx = route_map[neighbor][0]
                        if neighbor_r_idx not in visited_routes:
                            next_node = neighbor
                            break
                else:
                    # Any valid customer
                    for neighbor in self.neighbors[curr]:
                        if neighbor in route_map:
                            next_node = neighbor
                            break

            # Remove `curr` from route_map and the actual route
            # For simplicity without breaking p_idx of others, we update later.
            # Actually, removing alters index. Better to just mark as removed in route_map
            del route_map[curr]

            if next_node == -1 or next_node not in route_map:
                break
            curr = next_node

        # Physically remove from routes based on `removed`
        removed_set = set(removed)
        new_routes = []
        for r in working_routes:
            filtered_r = [n for n in r if n not in removed_set]
            if filtered_r:
                new_routes.append(filtered_r)

        working_routes = new_routes

        # Determine removal insertion order
        order_type = self.rng.integers(0, 4)
        if order_type == 0:
            self.rng.shuffle(removed)
        elif order_type == 1:
            # Descending demand/waste
            removed.sort(key=lambda x: self.waste.get(x, 0.0), reverse=True)
        elif order_type == 2:
            # Descending cost to depot
            removed.sort(key=lambda x: self.d[x, 0], reverse=True)
        elif order_type == 3:
            # Ascending cost to depot
            removed.sort(key=lambda x: self.d[x, 0])

        # Re-insert dynamically
        for customer in removed:
            best_r_idx = -1
            best_p_idx = -1
            best_cost = float("inf")
            c_waste = self.waste.get(customer, 0.0)

            for r_idx, route in enumerate(working_routes):
                current_load = sum(self.waste.get(n, 0.0) for n in route)
                if current_load + c_waste > self.Q:
                    continue

                # Try inserting before each point (including the depot at start, which means beginning of route)
                # i.e. 0 -> route[0] -> route[1] -> ... -> route[-1] -> 0

                # Insert at the beginning: 0 -> customer -> route[0]
                cost_start = -self.d[0, route[0]] + self.d[0, customer] + self.d[customer, route[0]]
                if cost_start < best_cost:
                    best_cost = cost_start
                    best_r_idx = r_idx
                    best_p_idx = 0

                # Insert in the middle
                for p_idx in range(1, len(route)):
                    prev_node = route[p_idx - 1]
                    next_node = route[p_idx]
                    cost = -self.d[prev_node, next_node] + self.d[prev_node, customer] + self.d[customer, next_node]
                    if cost < best_cost:
                        best_cost = cost
                        best_r_idx = r_idx
                        best_p_idx = p_idx

                # Insert at the end: route[-1] -> customer -> 0
                cost_end = -self.d[route[-1], 0] + self.d[route[-1], customer] + self.d[customer, 0]
                if cost_end < best_cost:
                    best_cost = cost_end
                    best_r_idx = r_idx
                    best_p_idx = len(route)

            if best_r_idx == -1:
                # Build new route
                working_routes.append([customer])
            else:
                working_routes[best_r_idx].insert(best_p_idx, customer)

        return working_routes, seed, removed
