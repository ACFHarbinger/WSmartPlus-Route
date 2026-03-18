"""
Operator implementations for HULK hyper-heuristic.

Wraps unstringing/stringing operators and local search moves.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

# Local search operators are implemented directly in this module
from ..other.operators.repair import greedy_insertion as greedy_insertion_op
from ..other.operators.unstringing_stringing import (
    apply_type_i_us,
    apply_type_ii_us,
    apply_type_iii_us,
    apply_type_iv_us,
    stringing_insertion,
)
from .solution import Solution


class HULKOperators:
    """Operator collection for HULK hyper-heuristic."""

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize operators.

        Args:
            dist_matrix: Distance matrix.
            wastes: Waste dictionary.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            mandatory_nodes: Must-visit nodes.
            seed: Random seed.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.mandatory_nodes = mandatory_nodes or []
        self.rng = random.Random(seed) if seed is not None else random.Random()

    # ===== Unstringing Operators (Destroy) =====

    def apply_unstring_type_i(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Apply Type I unstringing."""
        return self._apply_unstring(solution, n_remove, apply_type_i_us)

    def apply_unstring_type_ii(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Apply Type II unstringing."""
        return self._apply_unstring(solution, n_remove, apply_type_ii_us)

    def apply_unstring_type_iii(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Apply Type III unstringing."""
        return self._apply_unstring(solution, n_remove, apply_type_iii_us)

    def apply_unstring_type_iv(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        """Apply Type IV unstringing."""
        return self._apply_unstring(solution, n_remove, apply_type_iv_us)

    def _apply_unstring(self, solution: Solution, n_remove: int, unstring_func) -> Tuple[Solution, List[int]]:
        """Helper to apply unstringing operations."""
        routes = [list(r) for r in solution.routes]
        removed = []

        for _ in range(n_remove):
            # Find valid routes (length > 4)
            valid_routes = [i for i, r in enumerate(routes) if len(r) > 4]
            if not valid_routes:
                break

            r_idx = self.rng.choice(valid_routes)
            route = routes[r_idx]

            # Select node to remove
            i = self.rng.randint(1, len(route) - 2)
            node = route[i]

            # Select parameters for unstring operation
            valid_targets = [idx for idx in range(1, len(route) - 1) if idx not in (i - 1, i, i + 1)]

            if len(valid_targets) < 2:
                continue

            try:
                j, k = self.rng.sample(valid_targets, 2)

                # For type III and IV, we need additional parameters
                if unstring_func in (apply_type_iii_us, apply_type_iv_us):
                    if len(valid_targets) < 3:
                        continue
                    l = self.rng.choice([t for t in valid_targets if t not in (j, k)])
                    new_route = unstring_func(route, i, j, k, l)
                else:
                    new_route = unstring_func(route, i, j, k)

                if len(new_route) < len(route):
                    routes[r_idx] = new_route
                    removed.append(node)
            except Exception:
                continue

        new_solution = Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)
        return new_solution, removed

    # ===== Stringing Operators (Repair) =====

    def apply_string_repair(
        self, solution: Solution, removed: List[int], string_type: str, expand_pool: bool = False
    ) -> Solution:
        """
        Apply stringing operator to reinsert removed nodes.

        Args:
            solution: Current solution.
            removed: Nodes to reinsert.
            string_type: Type of stringing ("type_i", "type_ii", "type_iii", "type_iv").
            expand_pool: Includes globally unassigned pool for VRPP support.

        Returns:
            Repaired solution.
        """
        string_map = {
            "type_i": 1,
            "type_ii": 2,
            "type_iii": 3,
            "type_iv": 4,
        }

        if string_type not in string_map:
            # Fallback to greedy
            return self._greedy_repair(solution, removed, expand_pool=expand_pool)

        op_type = string_map[string_type]
        routes = [list(r) for r in solution.routes]

        routes = stringing_insertion(
            routes=routes,
            removed_nodes=removed,
            string_type=op_type,
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.rng,
            expand_pool=expand_pool,
        )

        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)

    def _greedy_repair(self, solution: Solution, removed: List[int], expand_pool: bool = False) -> Solution:
        """Greedy repair fallback."""
        routes = greedy_insertion_op(
            [list(r) for r in solution.routes],
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
            cost_unit=self.C,
            expand_pool=expand_pool,
        )
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)

    # ===== Local Search Operators =====

    def _simple_2_opt(self, route: List[int]) -> Tuple[List[int], bool]:
        """Simple 2-opt implementation."""
        n = len(route)
        if n <= 3:
            return route, False

        best_route = list(route)
        best_dist = self._calc_route_distance(best_route)
        improved = False

        for i in range(n - 2):
            for j in range(i + 2, n):
                # Reverse segment between i+1 and j
                new_route = route[: i + 1] + route[i + 1 : j + 1][::-1] + route[j + 1 :]
                new_dist = self._calc_route_distance(new_route)

                if new_dist < best_dist - 1e-6:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True

        return best_route, improved

    def _calc_route_distance(self, route: List[int]) -> float:
        """Calculate distance for a single route."""
        if not route:
            return 0.0
        dist = self.dist_matrix[0][route[0]]
        for i in range(len(route) - 1):
            dist += self.dist_matrix[route[i]][route[i + 1]]
        dist += self.dist_matrix[route[-1]][0]
        return dist

    def apply_2_opt(self, solution: Solution) -> Solution:
        """Apply 2-opt local search."""
        routes = [list(r) for r in solution.routes]
        improved = False

        for r_idx in range(len(routes)):
            new_route, improvement = self._simple_2_opt(routes[r_idx])
            if improvement:
                routes[r_idx] = new_route
                improved = True

        if improved:
            return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)
        return solution

    def apply_3_opt(self, solution: Solution) -> Solution:
        """Apply 3-opt local search (simplified version)."""
        # For simplicity, apply 2-opt twice
        result = self.apply_2_opt(solution)
        result2 = self.apply_2_opt(result)
        return result2 if result2.cost < result.cost else result

    def apply_swap(self, solution: Solution) -> Solution:
        """Apply swap move between routes."""
        routes = [list(r) for r in solution.routes]
        if len(routes) < 2:
            return solution

        best_solution = solution
        best_cost = solution.cost

        # Try swapping nodes between different routes
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if not routes[i] or not routes[j]:
                    continue

                for pos_i in range(len(routes[i])):
                    for pos_j in range(len(routes[j])):
                        # Swap nodes
                        test_routes = [list(r) for r in routes]
                        test_routes[i][pos_i], test_routes[j][pos_j] = (
                            test_routes[j][pos_j],
                            test_routes[i][pos_i],
                        )

                        # Check capacity
                        load_i = sum(self.wastes.get(n, 0) for n in test_routes[i])
                        load_j = sum(self.wastes.get(n, 0) for n in test_routes[j])

                        if load_i <= self.capacity and load_j <= self.capacity:
                            test_sol = Solution(
                                test_routes,
                                self.dist_matrix,
                                self.wastes,
                                self.capacity,
                                self.R,
                                self.C,
                            )

                            if test_sol.cost < best_cost - 1e-6:
                                best_solution = test_sol
                                best_cost = test_sol.cost

        return best_solution

    def apply_relocate(self, solution: Solution) -> Solution:
        """Apply relocate move."""
        routes = [list(r) for r in solution.routes]
        best_solution = solution
        best_cost = solution.cost

        # Try relocating nodes between routes
        for i in range(len(routes)):
            if not routes[i]:
                continue

            for pos in range(len(routes[i])):
                for j in range(len(routes)):
                    if i == j:
                        continue

                    # Try inserting at each position in route j
                    for ins_pos in range(len(routes[j]) + 1):
                        test_routes = [list(r) for r in routes]
                        # Remove from route i
                        removed = test_routes[i].pop(pos)
                        # Insert into route j
                        test_routes[j].insert(ins_pos, removed)

                        # Check capacity
                        load_j = sum(self.wastes.get(n, 0) for n in test_routes[j])

                        if load_j <= self.capacity:
                            test_sol = Solution(
                                test_routes,
                                self.dist_matrix,
                                self.wastes,
                                self.capacity,
                                self.R,
                                self.C,
                            )

                            if test_sol.cost < best_cost - 1e-6:
                                best_solution = test_sol
                                best_cost = test_sol.cost

        return best_solution
