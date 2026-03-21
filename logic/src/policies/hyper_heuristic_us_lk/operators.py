"""
Operator implementations for HULK hyper-heuristic.

Wraps unstringing/stringing operators and local search moves.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..other.operators import (
    greedy_insertion as greedy_insertion_op,
)
from ..other.operators import (
    regret_2_insertion,
    shaw_removal,
    string_removal,
)
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

    def apply_unstring_type_i(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        return self._apply_unstring(solution, n_remove, apply_type_i_us)

    def apply_unstring_type_ii(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        return self._apply_unstring(solution, n_remove, apply_type_ii_us)

    def apply_unstring_type_iii(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        return self._apply_unstring(solution, n_remove, apply_type_iii_us)

    def apply_unstring_type_iv(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        return self._apply_unstring(solution, n_remove, apply_type_iv_us)

    def apply_unstring_shaw(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        routes = [list(r) for r in solution.routes]
        partial, removed = shaw_removal(routes, n_remove, self.dist_matrix)
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C), removed

    def apply_unstring_string(self, solution: Solution, n_remove: int) -> Tuple[Solution, List[int]]:
        routes = [list(r) for r in solution.routes]
        partial, removed = string_removal(routes, n_remove, self.dist_matrix, rng=self.rng)
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C), removed

    def _apply_unstring(self, solution: Solution, n_remove: int, unstring_func) -> Tuple[Solution, List[int]]:
        routes = [list(r) for r in solution.routes]
        removed = []
        for _ in range(n_remove):
            valid_routes = [i for i, r in enumerate(routes) if len(r) > 4]
            if not valid_routes:
                break
            r_idx = self.rng.choice(valid_routes)
            route = routes[r_idx]
            i = self.rng.randint(1, len(route) - 2)
            node = route[i]
            # Type I-IV logic usually needs 2 params j, k
            valid_targets = [idx for idx in range(1, len(route) - 1) if idx not in (i - 1, i, i + 1)]
            if len(valid_targets) < 2:
                continue
            j, k = self.rng.sample(valid_targets, 2)
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
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C), removed

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
        string_map = {"type_i": 1, "type_ii": 2, "type_iii": 3, "type_iv": 4}
        if string_type not in string_map:
            if string_type == "regret_2":
                return self._regret_2_repair(solution, removed, expand_pool)
            return self._greedy_repair(solution, removed, expand_pool)
        op_type = string_map[string_type]
        routes = stringing_insertion(
            [list(r) for r in solution.routes],
            removed,
            op_type,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.mandatory_nodes,
            self.rng,
            expand_pool,
        )
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)

    def _greedy_repair(self, solution: Solution, removed: List[int], expand_pool: bool = False) -> Solution:
        routes = greedy_insertion_op(
            [list(r) for r in solution.routes],
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.mandatory_nodes,
            cost_unit=self.C,
            expand_pool=expand_pool,
        )
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)

    def _regret_2_repair(self, solution: Solution, removed: List[int], expand_pool: bool = False) -> Solution:
        routes = regret_2_insertion(
            [list(r) for r in solution.routes],
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.mandatory_nodes,
            cost_unit=self.C,
            expand_pool=expand_pool,
        )
        return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)

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
        """
        Apply 3-opt (K-opt) local search.
        Follows the 3-opt reconnection logic for VRP tours.
        """
        routes = [list(r) for r in solution.routes]
        improved = False

        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 4:
                continue

            # Simplified 3-opt: try random 3-edge reconnections
            for _ in range(5):  # Simplified intensity
                if len(route) < 4:
                    break
                i, j, k = sorted(self.rng.sample(range(len(route)), 3))

                # Standard 3-opt moves
                best_r = route
                best_d = self._dist(route)

                s1, s2, s3, s4 = route[: i + 1], route[i + 1 : j + 1], route[j + 1 : k + 1], route[k + 1 :]

                options = [
                    s1 + s2[::-1] + s3[::-1] + s4,
                    s1 + s3 + s2 + s4,
                    s1 + s3[::-1] + s2 + s4,
                    s1 + s2 + s3[::-1] + s4,
                    s1 + s3 + s2[::-1] + s4,
                ]

                for opt in options:
                    d = self._dist(opt)
                    if d < best_d - 1e-6:
                        best_d = d
                        best_r = opt
                        improved = True

                route = best_r
            routes[r_idx] = route

        if improved:
            return Solution(routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C)
        return solution

    def apply_swap(self, solution: Solution) -> Solution:
        """Apply swap move between routes."""
        routes = [list(r) for r in solution.routes]
        if len(routes) < 2:
            return solution

        best_solution = solution
        best_cost = solution.cost

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if not routes[i] or not routes[j]:
                    continue

                for pos_i in range(len(routes[i])):
                    for pos_j in range(len(routes[j])):
                        test_routes = [list(r) for r in routes]
                        test_routes[i][pos_i], test_routes[j][pos_j] = (
                            test_routes[j][pos_j],
                            test_routes[i][pos_i],
                        )

                        load_i = sum(self.wastes.get(n, 0) for n in test_routes[i])
                        load_j = sum(self.wastes.get(n, 0) for n in test_routes[j])

                        if load_i <= self.capacity and load_j <= self.capacity:
                            test_sol = Solution(
                                test_routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C
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

        for i in range(len(routes)):
            if not routes[i]:
                continue
            for pos in range(len(routes[i])):
                for j in range(len(routes)):
                    if i == j:
                        continue
                    for ins_pos in range(len(routes[j]) + 1):
                        test_routes = [list(r) for r in routes]
                        removed = test_routes[i].pop(pos)
                        test_routes[j].insert(ins_pos, removed)
                        load_j = sum(self.wastes.get(n, 0) for n in test_routes[j])
                        if load_j <= self.capacity:
                            test_sol = Solution(
                                test_routes, self.dist_matrix, self.wastes, self.capacity, self.R, self.C
                            )
                            if test_sol.cost < best_cost - 1e-6:
                                best_solution = test_sol
                                best_cost = test_sol.cost
        return best_solution
