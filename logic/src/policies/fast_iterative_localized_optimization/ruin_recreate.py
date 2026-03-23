"""
Ruin and Recreate operator for Fast Iterative Localized Optimization (FILO).

Upgraded to be strictly Profit-Aware for the VRPP.
"""

import copy
from typing import Dict, List, Set, Tuple

import numpy as np


class RuinAndRecreate:
    """
    Operator that applies localized Ruin & Recreate (shaking) to a solution.
    Modified to evaluate insertions based on economic profit (Revenue - Cost).
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        rng: np.random.Generator,
        profit_aware_operators: bool = False,
        vrpp: bool = True,
    ):
        """Initialize RuinAndRecreate operator."""
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.rng = rng
        self.profit_aware_operators = profit_aware_operators
        self.vrpp = vrpp

        # Pre-compute neighbors list (omitting depot) sorted by distance for each node
        self.neighbors: List[List[int]] = []
        for i in range(len(self.d)):
            sorted_neighbors = np.argsort(self.d[i])
            filtered = [int(n) for n in sorted_neighbors if n != 0 and n != i]
            self.neighbors.append(filtered)

    def apply(  # noqa: C901
        self,
        routes: List[List[int]],
        omega: List[int],
        all_customers: List[int],
        mandatory_nodes: List[int],
    ) -> Tuple[List[List[int]], int, List[int]]:
        """
        Apply Ruin and Recreate to the current routing solution.

        Args:
            routes: Current routes.
            omega: Subset of active localized nodes to pick the seed from.
            all_customers: List of all valid nodes in the environment.
            mandatory_nodes: Nodes that must be collected regardless of profit.

        Returns:
            Tuple of (new_routes, number_of_nodes_ruined, list_of_ruined_nodes)
        """
        working_routes = copy.deepcopy(routes)
        current_loads = [sum(self.waste.get(node, 0.0) for node in route) for route in working_routes]
        mandatory_set: Set[int] = set(mandatory_nodes)

        # --- 1. RUIN PHASE (Spatial Localization) ---
        n_remove = self.rng.integers(5, max(6, int(len(all_customers) * 0.15)))
        removed_customers: List[int] = []

        if omega:
            seed = int(self.rng.choice(omega))
            removed_customers.append(seed)
            for neighbor in self.neighbors[seed]:
                if len(removed_customers) >= n_remove:
                    break
                removed_customers.append(neighbor)

        # Add a few completely random customers to guarantee ergodicity
        random_additions = min(3, len(all_customers))
        random_nodes = self.rng.choice(all_customers, size=random_additions, replace=False)
        for rn in random_nodes:
            if int(rn) not in removed_customers:
                removed_customers.append(int(rn))

        # Actually remove them from routes
        for r_idx, route in enumerate(working_routes):
            working_routes[r_idx] = [n for n in route if n not in removed_customers]
            current_loads[r_idx] = sum(self.waste.get(n, 0.0) for n in working_routes[r_idx])

        # --- 2. RECREATE PHASE (Profit-Aware Greedy) ---
        if self.vrpp:
            # Consider all customers not currently in routes (including those just removed)
            visited = {n for r in working_routes for n in r}
            reinsertion_pool = [n for n in all_customers if n not in visited]
        else:
            reinsertion_pool = removed_customers

        # Shuffle pool to avoid deterministic insertion traps
        self.rng.shuffle(reinsertion_pool)

        for customer in reinsertion_pool:
            best_profit = -float("inf")
            best_r_idx = -2
            best_p_idx = -1

            customer_waste = self.waste.get(customer, 0.0)
            revenue = customer_waste * self.R
            is_mandatory = customer in mandatory_set

            for r_idx, route in enumerate(working_routes):
                if current_loads[r_idx] + customer_waste > self.Q:
                    continue

                if len(route) == 0:
                    continue

                # Insert at the beginning: 0 -> customer -> route[0]
                dist_start = self.d[0, customer] + self.d[customer, route[0]] - self.d[0, route[0]]
                profit_start = revenue - (dist_start * self.C)
                if profit_start > best_profit:
                    best_profit = profit_start
                    best_r_idx = r_idx
                    best_p_idx = 0

                # Insert in the middle
                for p_idx in range(1, len(route)):
                    prev_node = route[p_idx - 1]
                    next_node = route[p_idx]
                    dist_mid = self.d[prev_node, customer] + self.d[customer, next_node] - self.d[prev_node, next_node]
                    profit_mid = revenue - (dist_mid * self.C)
                    if profit_mid > best_profit:
                        best_profit = profit_mid
                        best_r_idx = r_idx
                        best_p_idx = p_idx

                # Insert at the end: route[-1] -> customer -> 0
                dist_end = self.d[route[-1], customer] + self.d[customer, 0] - self.d[route[-1], 0]
                profit_end = revenue - (dist_end * self.C)
                if profit_end > best_profit:
                    best_profit = profit_end
                    best_r_idx = r_idx
                    best_p_idx = len(route)

            # Check creating an entirely new route: 0 -> customer -> 0
            dist_new = self.d[0, customer] + self.d[customer, 0]
            profit_new = revenue - (dist_new * self.C)
            if profit_new > best_profit:
                best_profit = profit_new
                best_r_idx = -1

            # --- ECONOMIC TERMINATION / EVALUATION ---
            if best_r_idx != -2:
                # If profit_aware_operators is enabled, we drop nodes that create financial loss.
                if self.profit_aware_operators and not is_mandatory and best_profit < -1e-4:
                    continue  # Node stays unassigned (opportunistic starvation prevented!)

                if best_r_idx == -1:
                    working_routes.append([customer])
                    current_loads.append(customer_waste)
                else:
                    working_routes[best_r_idx].insert(best_p_idx, customer)
                    current_loads[best_r_idx] += customer_waste

            elif is_mandatory:
                # Mandatory node could not fit in any existing route; force open a new route
                working_routes.append([customer])
                current_loads.append(customer_waste)

        # Cleanup any empty routes generated by the ruin phase
        working_routes = [r for r in working_routes if r]

        return working_routes, len(removed_customers), removed_customers
