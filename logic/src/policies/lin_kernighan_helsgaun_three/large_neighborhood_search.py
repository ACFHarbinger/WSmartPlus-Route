"""
LKH-3 Large Neighborhood Search (LNS) Matheuristic.

Wraps the LKH-3 k-opt core in an outer ruin-and-recreate loop to handle
VRPP subset selection. The LKH-3 engine optimizes routing for the current
active subset, while LNS operators manage node inclusion/exclusion based
on profitability.

Architecture
------------
1. **Initialize**: Build feasible subset + route with LKH-3
2. **Optimize**: Run LKH-3 until local routing optimum
3. **Plateau Check**: If stuck, apply Destroy → Repair
4. **Deep Plateau**: If stuck for N iters, apply Perturbation
5. **Loop**: Return modified solution to step 2

Operator Dispatch
-----------------
- ``profit_aware_operators=False``: Standard CVRP operators (removed nodes only)
- ``profit_aware_operators=True``: VRPP operators (removed + unvisited pool)

Example
-------
>>> from logic.src.policies.lin_kernighan_helsgaun_three.large_neighborhood_search import LKH3_LNS
>>> solver = LKH3_LNS(
...     distance_matrix=dist,
...     wastes={1: 10, 2: 20, 3: 30},
...     capacity=100.0,
...     revenue=2.0,
...     cost_unit=1.0,
...     profit_aware_operators=True,
...     mandatory_nodes=[1],
...     seed=42,
... )
>>> routes, profit = solver.solve(max_iterations=100, lkh_trials=500, n_vehicles=3)
"""

from __future__ import annotations

from random import Random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.lin_kernighan_helsgaun_three.candidate_set import (
    compute_alpha_measures,
    get_candidate_set,
)
from logic.src.policies.lin_kernighan_helsgaun_three.lin_kernighan_helsgaun import (
    solve_lkh,
)
from logic.src.policies.other.operators.destroy.historical import (
    historical_profit_removal,
    historical_removal,
)
from logic.src.policies.other.operators.destroy.neighbor import (
    neighbor_profit_removal,
    neighbor_removal,
)

# Destroy operators
from logic.src.policies.other.operators.destroy.route import (
    route_profit_removal,
    route_removal,
)
from logic.src.policies.other.operators.destroy.sector import (
    sector_profit_removal,
    sector_removal,
)
from logic.src.policies.other.operators.perturbation.evolutionary import (
    evolutionary_perturbation,
    evolutionary_perturbation_profit,
)

# Perturbation operators
from logic.src.policies.other.operators.perturbation.genetic_transformation import (
    genetic_transformation,
    genetic_transformation_profit,
)

# Repair operators
from logic.src.policies.other.operators.repair.deep import (
    deep_insertion,
    deep_profit_insertion,
)
from logic.src.policies.other.operators.repair.nearest import (
    nearest_insertion,
    nearest_profit_insertion,
)
from logic.src.policies.other.operators.repair.savings import (
    savings_insertion,
    savings_profit_insertion,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder


class LKH3_LNS:
    """
    LKH-3 + Large Neighborhood Search matheuristic for VRPP.

    Implements a two-level optimization architecture where the LKH-3 core
    handles routing optimization for a given subset of nodes, while the LNS
    outer loop manages subset selection via adaptive destroy-repair operators.

    Attributes:
        distance_matrix: (N×N) distance matrix (node 0 = depot).
        wastes: Dict mapping node ID to demand/profit value.
        capacity: Vehicle capacity constraint.
        revenue: Revenue per unit collected (R parameter).
        cost_unit: Cost per distance unit (C parameter).
        profit_aware_operators: If True, operators consider unvisited nodes
            and make profit-based decisions (VRPP mode).
        mandatory_nodes: Set of nodes that must be visited regardless of profit.
        coords: Node coordinates (N×2) for geometric operators like sector removal.
        np_rng: NumPy random generator for array operations.
        rng: Random number generator for reproducibility.
        seed: Seed for the random number generator.
        recorder: Optional state recorder for visualization.
        history: Historical cost tracking for adaptive operator selection.
        elite_pool: Pool of elite solutions for perturbation operators.
        max_pool_size: Maximum number of elite solutions to retain.
        n_original: Number of original nodes (before augmentation).
        R: Revenue per unit collected (VRPP parameter).
        C: Cost per distance unit.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue: float = 1.0,
        cost_unit: float = 1.0,
        profit_aware_operators: bool = False,
        mandatory_nodes: Optional[List[int]] = None,
        coords: Optional[np.ndarray] = None,
        np_rng: Optional[np.random.Generator] = None,
        rng: Optional[Random] = None,
        seed: int = 42,
        recorder: Optional[PolicyStateRecorder] = None,
        max_pool_size: int = 5,
        n_original: int = 0,
        R: float = 1.0,
        C: float = 1.0,
        perturb_operator_weights: Optional[List[float]] = None,
    ):
        """Initialize LKH-3 LNS matheuristic solver.

        Args:
            distance_matrix: (N×N) symmetric distance matrix.
            wastes: Node demands/profits as {node_id: value}.
            capacity: Vehicle capacity.
            revenue: Revenue per unit collected (VRPP parameter).
            cost_unit: Cost per distance unit.
            profit_aware_operators: Toggle VRPP mode (subset selection).
            mandatory_nodes: Nodes that must be visited.
            coords: Node coordinates (N×2) for geometric operators.
            np_rng: NumPy random generator for array operations.
            rng: Random number generator for reproducibility.
            seed: Seed for the random number generator.
            recorder: Optional state recorder for visualization.
            max_pool_size: Maximum number of elite solutions to retain.
            n_original: Number of original nodes (before augmentation).
            R: Revenue per unit collected (VRPP parameter).
            C: Cost per distance unit.
        """
        self.distance_matrix = distance_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.revenue = revenue
        self.cost_unit = cost_unit
        self.profit_aware_operators = profit_aware_operators
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()
        self.coords = coords
        self.seed = seed
        self.rng = rng if rng is not None else Random(seed)
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng(seed)
        self.n_original = n_original
        self.R = R
        self.C = C
        self.perturb_operator_weights = perturb_operator_weights if perturb_operator_weights is not None else [0.6, 0.4]

        # Historical memory for adaptive operator selection
        self.history: Dict[int, float] = {}

        # Elite solution pool for perturbation
        self.elite_pool: List[List[List[int]]] = []
        self.max_pool_size = max_pool_size

        # Cached candidates for warm-starting (Phase 2: Fix LNS Cold Start Amnesia)
        # Generated once during first solve() call and reused across all LKH-3 invocations
        self.cached_candidates: Optional[Dict[int, List[int]]] = None

        # Recorder for output visualization
        self._viz = recorder

    def solve(
        self,
        max_iterations: int = 100,
        lkh_trials: int = 500,
        n_vehicles: int = 3,
        plateau_limit: int = 10,
        deep_plateau_limit: int = 30,
        popmusic_subpath_size: int = 50,
        popmusic_trials: int = 50,
        popmusic_max_candidates: int = 5,
        max_k_opt: int = 5,
        use_ip_merging: bool = True,
    ) -> Tuple[List[List[int]], float]:
        """
        Main LNS+LKH-3 matheuristic loop.

        Algorithm outline:
        1. Initialize with greedy profitable subset (VRPP) or all nodes (CVRP)
        2. For each iteration:
           a. Optimize current subset with LKH-3 k-opt
           b. If improved, update best solution and elite pool
           c. If plateau reached, apply Destroy-Repair
           d. If deep plateau, apply Perturbation against elite
        3. Return best solution found

        Args:
            max_iterations: Total LNS iterations.
            lkh_trials: Max k-opt iterations per LKH-3 run.
            n_vehicles: Number of available vehicles.
            plateau_limit: Iterations without improvement before Destroy-Repair.
            deep_plateau_limit: Iterations before Perturbation.
            popmusic_subpath_size: POPMUSIC sub-path size for large instances.
            popmusic_trials: Number of POPMUSIC runs.
            popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
            max_k_opt: Maximum k for k-opt moves (2-5).
            use_ip_merging: If True, use IP-based tour recombination.

        Returns:
            Tuple of (best_routes, best_objective) where:
            - best_routes: List of routes (each route is list of node IDs)
            - best_objective: Profit (VRPP) or negative cost (CVRP)
        """
        # 1. Initialize with greedy solution
        best_routes, best_obj = self._initialize_solution(
            lkh_trials,
            n_vehicles,
            popmusic_subpath_size,
            popmusic_trials,
            popmusic_max_candidates,
            max_k_opt,
            use_ip_merging,
        )
        current_routes = [r[:] for r in best_routes]

        iterations_since_improvement = 0

        # 2. Main LNS loop
        for _iteration in range(max_iterations):
            # 2a. Optimize current subset with LKH-3
            optimized_routes, obj = self._optimize_routes(
                current_routes,
                lkh_trials,
                n_vehicles,
                popmusic_subpath_size,
                popmusic_trials,
                popmusic_max_candidates,
                max_k_opt,
                use_ip_merging,
            )

            # 2b. Update best solution
            if obj > best_obj + 1e-6:
                best_routes = [r[:] for r in optimized_routes]
                best_obj = obj
                iterations_since_improvement = 0
                self._update_elite_pool(best_routes)
            else:
                iterations_since_improvement += 1

            # Record state for visualization
            if self._viz:
                self._viz.record(
                    iteration=_iteration,
                    best_obj=best_obj,
                    curr_obj=obj,
                )

            # Update historical scores for adaptive operator selection
            self._update_history(optimized_routes, obj)

            # 2c. Plateau check: Apply Destroy-Repair
            if iterations_since_improvement >= plateau_limit:
                current_routes = self._destroy_repair(optimized_routes)
                iterations_since_improvement = 0

            # 2d. Deep plateau: Apply Perturbation
            elif iterations_since_improvement >= deep_plateau_limit and self.elite_pool:
                current_routes = self._perturbation(optimized_routes)
                iterations_since_improvement = 0

        return best_routes, best_obj

    def _initialize_solution(
        self,
        lkh_trials: int,
        n_vehicles: int,
        popmusic_subpath_size: int,
        popmusic_trials: int,
        popmusic_max_candidates: int,
        max_k_opt: int,
        use_ip_merging: bool,
    ) -> Tuple[List[List[int]], float]:
        """
        Build initial feasible solution:
        - Start with mandatory nodes
        - Greedily add profitable nodes by marginal profit score
        - Stop when capacity limit reached or no profitable nodes remain

        Args:
            lkh_trials: Max LKH-3 iterations.
            n_vehicles: Number of vehicles.
            popmusic_subpath_size: POPMUSIC parameter.
            popmusic_trials: POPMUSIC parameter.
            popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
            max_k_opt: Maximum k for k-opt.
            use_ip_merging: IP-based tour merging flag.

        Returns:
            Tuple of (routes, objective).
        """
        # VRPP: Start with mandatory nodes + greedy profit selection
        active_nodes = list(self.mandatory_nodes)
        n = len(self.distance_matrix)

        # Compute marginal profit for each optional node
        profit_scores: List[Tuple[float, int]] = []
        for node in range(1, n):
            if node not in self.mandatory_nodes:
                fill = self.wastes.get(node, 0.0)
                revenue = self.revenue * fill
                # Estimate routing cost as direct depot round-trip
                cost = self.distance_matrix[0, node] + self.distance_matrix[node, 0]
                profit = revenue - self.cost_unit * cost
                profit_scores.append((profit, node))

        # Sort by profit descending
        profit_scores.sort(reverse=True)

        # Greedily add profitable nodes within capacity
        total_load = sum(self.wastes.get(n, 0.0) for n in active_nodes)
        total_capacity = self.capacity * n_vehicles
        for profit, node in profit_scores:
            if profit > 0 and total_load + self.wastes.get(node, 0.0) <= total_capacity:
                active_nodes.append(node)
                total_load += self.wastes.get(node, 0.0)

        return self._route_nodes(
            active_nodes,
            lkh_trials,
            n_vehicles,
            popmusic_subpath_size,
            popmusic_trials,
            popmusic_max_candidates,
            max_k_opt,
            use_ip_merging,
        )

    def _route_nodes(
        self,
        nodes: List[int],
        lkh_trials: int,
        n_vehicles: int,
        popmusic_subpath_size: int,
        popmusic_trials: int,
        popmusic_max_candidates: int,
        max_k_opt: int,
        use_ip_merging: bool,
        initial_routes: Optional[List[List[int]]] = None,
    ) -> Tuple[List[List[int]], float]:
        """
        Route a given set of nodes with LKH-3.

        Creates a sub-problem containing only the specified nodes, runs LKH-3,
        and maps the result back to original node indices.

        **Phase 2 Fix: Warm-Start Integration**
        If `initial_routes` is provided (non-None), the method flattens the routes
        into a 1D augmented tour with dummy depot indices and passes it to LKH-3
        as `initial_tour`. This preserves the repaired structure from destroy-repair
        operators and avoids expensive nearest-neighbor reconstruction.

        Args:
            nodes: List of customer node IDs to route.
            lkh_trials: Max LKH-3 iterations.
            n_vehicles: Number of vehicles.
            popmusic_subpath_size: POPMUSIC parameter.
            popmusic_trials: POPMUSIC parameter.
            popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
            max_k_opt: Maximum k for k-opt.
            use_ip_merging: IP-based tour merging flag.
            initial_routes: Optional warm-start routes (global node indices).
                If provided, flattened to 1D tour and passed to solve_lkh.

        Returns:
            Tuple of (routes_global, objective) where:
            - routes_global: Routes with original node indices
            - objective: Profit (VRPP) or negative cost (CVRP)
        """
        if not nodes:
            return [[]], 0.0

        # Build sub-problem distance matrix and demand array
        node_map = {0: 0}  # Depot always at index 0
        reverse_map = {0: 0}  # Global → Local mapping
        for idx, node in enumerate(nodes, start=1):
            node_map[idx] = node
            reverse_map[node] = idx

        n_sub = len(nodes) + 1
        sub_dist = np.zeros((n_sub, n_sub))
        sub_waste = np.zeros(n_sub)

        for i in range(n_sub):
            for j in range(n_sub):
                orig_i = node_map[i]
                orig_j = node_map[j]
                sub_dist[i, j] = self.distance_matrix[orig_i, orig_j]

        for i in range(1, n_sub):
            orig_node = node_map[i]
            sub_waste[i] = self.wastes.get(orig_node, 0.0)

        # Phase 2: Flatten initial_routes to 1D augmented tour (warm-start)
        initial_tour_local: Optional[List[int]] = None
        if initial_routes is not None:
            # Map global routes to local sub-problem indices
            local_routes: List[List[int]] = []
            for route in initial_routes:
                local_route = [reverse_map[n] for n in route if n in reverse_map]
                if local_route:
                    local_routes.append(local_route)

            # Flatten to 1D tour with augmented dummy depots [N, N+1, ...]
            if local_routes:
                # Flatten with depot separators: [0, route1, 0, route2, 0, ...]
                # Then convert to augmented dummy depot encoding
                n_original = n_sub  # Original graph size (before augmentation)
                n_routes = len(local_routes)

                # Build augmented tour: [0, r1, N, r2, N+1, r3, 0]
                initial_tour_local = [0]
                for idx, route in enumerate(local_routes):
                    initial_tour_local.extend(route)
                    if idx < n_routes - 1:
                        # Insert augmented dummy depot: N, N+1, N+2, ...
                        dummy_idx = n_original + idx
                        initial_tour_local.append(dummy_idx)
                initial_tour_local.append(0)

        # Phase 2: Generate and cache candidate sets on first call
        if self.cached_candidates is None:
            alpha = compute_alpha_measures(sub_dist)
            self.cached_candidates = get_candidate_set(
                sub_dist,
                alpha,
                max_candidates=popmusic_max_candidates,
            )

        # Run LKH-3 with warm-start
        routes, cost = solve_lkh(
            distance_matrix=sub_dist,
            initial_tour=initial_tour_local,  # Phase 2: Pass warm-start tour
            waste=sub_waste,
            capacity=self.capacity,
            max_iterations=lkh_trials,
            popmusic_subpath_size=popmusic_subpath_size,
            popmusic_trials=popmusic_trials,
            popmusic_max_candidates=popmusic_max_candidates,
            max_k_opt=max_k_opt,
            use_ip_merging=use_ip_merging,
            max_pool_size=self.max_pool_size,
            n_vehicles=n_vehicles,
            candidate_set=self.cached_candidates,  # Phase 2: Reuse cached candidates
            recorder=self._viz,
            np_rng=self.np_rng,
            rng=self.rng,
            seed=self.seed,
            n_original=self.n_original,
        )

        # Extract routes and map back to original indices
        routes_global = [[node_map[n] for n in route] for route in routes]

        # Compute objective
        routing_cost = self._compute_routing_cost(routes_global)
        if self.profit_aware_operators:
            # VRPP: Profit = Revenue - Cost
            collected_fill = sum(self.wastes.get(n, 0.0) for route in routes_global for n in route)
            profit = self.revenue * collected_fill - self.cost_unit * routing_cost
            return routes_global, profit
        else:
            # CVRP: Minimize cost (return negative for maximization framework)
            return routes_global, -routing_cost

    def _optimize_routes(
        self,
        routes: List[List[int]],
        lkh_trials: int,
        n_vehicles: int,
        popmusic_subpath_size: int,
        popmusic_trials: int,
        popmusic_max_candidates: int,
        max_k_opt: int,
        use_ip_merging: bool,
    ) -> Tuple[List[List[int]], float]:
        """
        Re-optimize current routes with LKH-3 using warm-start.

        Passes the current route structure as ``initial_routes`` so that
        LKH-3 starts from the repaired tour rather than regenerating
        from scratch (fixes Cold Start Amnesia).

        Args:
            routes: Current routes to warm-start from.
            lkh_trials: Max LKH-3 iterations.
            n_vehicles: Number of vehicles.
            popmusic_subpath_size: POPMUSIC parameter.
            popmusic_trials: POPMUSIC parameter.
            popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
            max_k_opt: Maximum k for k-opt.
            use_ip_merging: IP-based tour merging flag.

        Returns:
            Tuple of (optimized_routes, objective).
        """
        active_nodes = [n for route in routes for n in route]
        return self._route_nodes(
            active_nodes,
            lkh_trials,
            n_vehicles,
            popmusic_subpath_size,
            popmusic_trials,
            popmusic_max_candidates,
            max_k_opt,
            use_ip_merging,
            initial_routes=routes,  # Phase 2: Warm-start from repaired routes
        )

    def _destroy_repair(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Apply Destroy (Ruin) → Repair (Recreate) operators.

        Randomly selects one destroy operator and one repair operator,
        removes 10-20% of nodes, then reinserts them.

        Args:
            routes: Current solution.

        Returns:
            Modified routes after destroy-repair.
        """
        # Select operators via roulette wheel
        destroy_op = self._select_destroy_operator()
        repair_op = self._select_repair_operator()

        # Determine removal size (10-20% of nodes)
        total_nodes = sum(len(r) for r in routes)
        n_remove = max(1, int(self.rng.uniform(0.1, 0.2) * total_nodes))

        # Destroy phase
        modified_routes, removed_nodes = destroy_op(routes, n_remove)

        # Repair phase
        repaired_routes = repair_op(modified_routes, removed_nodes)

        return repaired_routes

    def _select_destroy_operator(self) -> Callable[[List[List[int]], int], Tuple[List[List[int]], List[int]]]:
        """
        Roulette-wheel selection of destroy operator.

        Returns a callable that takes (routes, n_remove) and returns
        (modified_routes, removed_nodes).

        Returns:
            Destroy operator function.
        """
        if self.profit_aware_operators:
            operators = [
                self._wrap_historical_profit_removal,
                self._wrap_neighbor_profit_removal,
                self._wrap_sector_profit_removal,
            ]
        else:
            operators = [
                self._wrap_historical_removal,
                self._wrap_neighbor_removal,
                self._wrap_sector_removal,
            ]

        return self.rng.choice(operators)

    def _select_repair_operator(self) -> Callable[[List[List[int]], List[int]], List[List[int]]]:
        """
        Roulette-wheel selection of repair operator.

        Returns a callable that takes (routes, removed_nodes) and returns
        updated routes.

        Returns:
            Repair operator function.
        """
        if self.profit_aware_operators:
            operators = [
                self._wrap_savings_profit_insertion,
                self._wrap_deep_profit_insertion,
                self._wrap_nearest_profit_insertion,
            ]
        else:
            operators = [
                self._wrap_savings_insertion,
                self._wrap_deep_insertion,
                self._wrap_nearest_insertion,
            ]

        return self.rng.choice(operators)

    # -----------------------------------------------------------------------
    # Operator Wrappers (Standard CVRP)
    # -----------------------------------------------------------------------

    def _wrap_historical_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Wrapper for historical_removal."""
        return historical_removal(routes, n_remove, self.history, rng=self.rng, noise=0.1)

    def _wrap_neighbor_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Wrapper for neighbor_removal."""
        return neighbor_removal(routes, n_remove, self.distance_matrix, rng=self.rng)

    def _wrap_sector_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Wrapper for sector_removal (fallback to neighbor if no coords)."""
        if self.coords is not None:
            return sector_removal(routes, n_remove, self.coords, depot=(0.0, 0.0), rng=self.rng)
        else:
            return self._wrap_neighbor_removal(routes, n_remove)

    def _wrap_savings_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrapper for savings_insertion."""
        return savings_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=list(self.mandatory_nodes),
            expand_pool=True,
        )

    def _wrap_deep_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrapper for deep_insertion."""
        return deep_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            alpha=0.3,
            mandatory_nodes=list(self.mandatory_nodes),
            expand_pool=True,
        )

    def _wrap_nearest_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrapper for nearest_insertion."""
        return nearest_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=list(self.mandatory_nodes),
            expand_pool=True,
        )

    # -----------------------------------------------------------------------
    # Operator Wrappers (VRPP Profit-Aware)
    # -----------------------------------------------------------------------

    def _wrap_historical_profit_removal(
        self, routes: List[List[int]], n_remove: int
    ) -> Tuple[List[List[int]], List[int]]:
        """Wrapper for historical_profit_removal."""
        return historical_profit_removal(
            routes,
            n_remove,
            self.history,
            self.distance_matrix,
            self.wastes,
            self.revenue,
            self.cost_unit,
            alpha=0.5,
            rng=self.rng,
            noise=0.1,
        )

    def _wrap_neighbor_profit_removal(
        self, routes: List[List[int]], n_remove: int
    ) -> Tuple[List[List[int]], List[int]]:
        """Wrapper for neighbor_profit_removal."""
        return neighbor_profit_removal(
            routes,
            n_remove,
            self.distance_matrix,
            self.wastes,
            self.revenue,
            self.cost_unit,
            rng=self.rng,
        )

    def _wrap_sector_profit_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Wrapper for sector_profit_removal (fallback to neighbor if no coords)."""
        if self.coords is not None:
            return sector_profit_removal(
                routes,
                n_remove,
                self.coords,
                self.distance_matrix,
                self.wastes,
                self.revenue,
                self.cost_unit,
                depot=(0.0, 0.0),
                rng=self.rng,
            )
        else:
            return self._wrap_neighbor_profit_removal(routes, n_remove)

    def _wrap_savings_profit_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrapper for savings_profit_insertion with expand_pool=True."""
        return savings_profit_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=list(self.mandatory_nodes),
            expand_pool=True,  # Consider unvisited nodes for VRPP
        )

    def _wrap_deep_profit_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrapper for deep_profit_insertion with expand_pool=True."""
        return deep_profit_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=list(self.mandatory_nodes),
            expand_pool=True,
        )

    def _wrap_nearest_profit_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrapper for nearest_profit_insertion with expand_pool=True."""
        return nearest_profit_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=list(self.mandatory_nodes),
            expand_pool=True,  # Consider unvisited nodes for VRPP
        )

    # -----------------------------------------------------------------------
    # Perturbation Operator
    # -----------------------------------------------------------------------

    def _select_worst_routes(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Select the n worst routes based on profit-oriented strategy."""
        target_routes = []
        all_removed_nodes = []
        for _ in range(n):
            # Break early if we run out of routes
            if not routes:
                break

            if self.profit_aware_operators:
                routes, removed = route_profit_removal(
                    routes=routes,
                    strategy="worst_profit",
                    dist_matrix=self.distance_matrix,
                    wastes=self.wastes,
                    R=self.R,
                    C=self.C,
                    rng=self.rng,
                )
            else:
                routes, removed = route_removal(
                    routes=routes,
                    strategy="costliest",
                    dist_matrix=self.distance_matrix,
                    wastes=self.wastes,
                    rng=self.rng,
                )
            target_routes.append(routes)
            all_removed_nodes.extend(removed)
        return target_routes, all_removed_nodes  # type: ignore[return-value]

    def _perturbation(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Apply deep perturbation operator against elite solution.

        Randomly selects an elite solution from the pool and applies
        genetic transformation to preserve common edges while rewiring
        the rest.

        Args:
            routes: Current solution.

        Returns:
            Perturbed routes.
        """
        # 1. Fallback: If no elite solutions exist, you MUST use Evolutionary
        if not self.elite_pool:
            # Target the 2 smallest/worst routes for localized balancing
            target_routes, _ = self._select_worst_routes(routes, n=2)
            if self.profit_aware_operators:
                return evolutionary_perturbation_profit(
                    routes,
                    self.distance_matrix,
                    self.capacity,
                    self.wastes,
                    self.R,
                    self.C,
                    target_routes,
                    rng=self.rng,
                )
            else:
                return evolutionary_perturbation(
                    routes, self.distance_matrix, self.capacity, self.wastes, target_routes, rng=self.rng
                )

        # 2. Adaptive Selection (Roulette Wheel)
        operators = ["genetic", "evolutionary"]
        chosen_op = self.rng.choices(operators, weights=self.perturb_operator_weights, k=1)[0]
        if chosen_op == "genetic":
            # Pull a random elite solution to ensure diversity
            elite_ref = self.rng.choice(self.elite_pool)

            if self.profit_aware_operators:
                return genetic_transformation_profit(
                    routes, elite_ref, self.distance_matrix, self.wastes, self.capacity, self.R, self.C, self.rng
                )
            else:
                return genetic_transformation(
                    routes, elite_ref, self.distance_matrix, self.wastes, self.capacity, self.rng
                )

        elif chosen_op == "evolutionary":
            target_routes, _ = self._select_worst_routes(routes, n=2)
            if self.profit_aware_operators:
                return evolutionary_perturbation_profit(
                    routes,
                    self.distance_matrix,
                    self.capacity,
                    self.wastes,
                    self.R,
                    self.C,
                    target_routes,
                    rng=self.rng,
                )
            else:
                return evolutionary_perturbation(
                    routes, self.distance_matrix, self.capacity, self.wastes, target_routes, rng=self.rng
                )

        return routes

    # -----------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------

    def _update_history(self, routes: List[List[int]], obj: float) -> None:
        """
        Update historical cost scores for nodes.

        Uses exponential moving average to track node quality over time.
        Nodes frequently appearing in high-quality solutions get lower scores.

        Args:
            routes: Current solution.
            obj: Objective value (profit or negative cost).
        """
        alpha = 0.1  # EMA weight for new observation

        for route in routes:
            for node in route:
                if node not in self.history:
                    self.history[node] = -obj  # Lower obj is better for history
                else:
                    # Update: history[n] ← α·(-obj) + (1-α)·history[n]
                    self.history[node] = alpha * (-obj) + (1 - alpha) * self.history[node]

    def _update_elite_pool(self, routes: List[List[int]]) -> None:
        """
        Add solution to elite pool.

        Maintains a fixed-size pool of diverse high-quality solutions
        for use in perturbation operators.

        Args:
            routes: Solution to add to pool.
        """
        # Deep copy routes
        self.elite_pool.append([r[:] for r in routes])

        # Maintain pool size
        if len(self.elite_pool) > self.max_pool_size:
            self.elite_pool.pop(0)

    def _compute_routing_cost(self, routes: List[List[int]]) -> float:
        """
        Compute total routing distance.

        Sums up all depot-to-customer, customer-to-customer, and
        customer-to-depot edges across all routes.

        Args:
            routes: List of routes (no depot nodes).

        Returns:
            Total routing cost.
        """
        cost = 0.0
        for route in routes:
            if not route:
                continue
            # Depot to first customer
            cost += self.distance_matrix[0, route[0]]
            # Customer to customer
            for i in range(len(route) - 1):
                cost += self.distance_matrix[route[i], route[i + 1]]
            # Last customer to depot
            cost += self.distance_matrix[route[-1], 0]
        return cost
