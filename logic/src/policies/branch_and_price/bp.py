"""
Branch-and-Price Solver for VRPP.

Implements the complete Branch-and-Price algorithm with:
- Column generation at each node
- Pluggable branching strategy (edge branching or Ryan-Foster)
- LP relaxation solving with pricing
- Integer solution recovery

Based on Sections 5 and 6 of Barnhart et al. (1998).

Reference:
    Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W. P., & Vance, P. H. (1998).
    "Branch-and-price: Column Generation for Solving Huge Integer Programs".
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .branching import (
    AnyBranchingConstraint,
    BranchAndBoundTree,
    BranchNode,
)
from .master_problem import Route, VRPPMasterProblem
from .pricing_subproblem import PricingSubproblem
from .rcspp_dp import RCSPPSolver


class BranchAndPriceSolver:
    """
    Branch-and-Price Algorithm for VRPP with Column Generation.

    Algorithm:
        1. Initialise with greedy routes.
        2. Solve LP relaxation with column generation.
        3. If LP solution is integer, terminate.
        4. Otherwise branch using the configured strategy and recurse.

    Branching strategies
    --------------------
    ``"edge"`` (default)
        Branches on the most-fractional directed arc (x_{uv}).  Integrates
        natively with the DP label extension.

    ``"ryan_foster"``
        Selection of a node-pair (r, s) to branch on. Produces two
        child nodes where r and s MUST or MUST NOT appear in the same
        route. Note: This is theoretically a heuristic when using a
        Set Covering master problem (default).
        **WARNING:** Ryan-Foster branching loses its theoretical exactness
        guarantee when applied to a Set Covering master problem (>= 1), as it
        can erroneously prune optimal over-covering solutions. Use 'edge'
        branching for rigorous proofs of optimality.

    Label-Correcting Algorithm
    --------------------------
    When ``use_exact_pricing=True``, the DP-based ``RCSPPSolver`` is used.
    This employs a **Label-Correcting** algorithm (FIFO queue) to handle
    potential negative edge costs during column generation. Two sub-modes
    are available via ``use_ng_routes``:

    ``use_ng_routes=True`` (default)
        Uses ng-route relaxation (Baldacci et al. 2011) — far fewer labels,
        scalable to large instances.

    ``use_ng_routes=False``
        Falls back to exact ESPPRC — every generated route is elementary,
        but the label count grows exponentially with instance size.
    """

    def __init__(
        self,
        n_nodes: int,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        mandatory_nodes: Optional[Set[int]] = None,
        max_iterations: int = 100,
        max_routes_per_iteration: int = 10,
        optimality_gap: float = 1e-4,
        use_ryan_foster: bool = False,
        branching_strategy: str = "edge",
        tree_search_strategy: str = "best_first",
        max_branch_nodes: int = 1000,
        use_exact_pricing: bool = False,
        vehicle_limit: Optional[int] = None,
        use_ng_routes: bool = True,
        ng_neighborhood_size: int = 8,
        cleanup_frequency: int = 20,
        cleanup_threshold: float = -100.0,
        early_termination_gap: float = 1e-3,
        multiple_waste_types: bool = False,
        allow_heuristic_ryan_foster: bool = False,
    ) -> None:
        """
        Initialise the Branch-and-Price solver.

        Args:
            n_nodes: Number of customer nodes (excluding depot, depot = 0).
            cost_matrix: Distance matrix of shape (n_nodes+1, n_nodes+1).
            wastes: Mapping from node ID to waste volume.
            capacity: Vehicle payload capacity.
            revenue_per_kg: Revenue per unit of waste collected.
            cost_per_km: Operating cost per unit of distance travelled.
            mandatory_nodes: Node indices that must be visited.
            max_iterations: Maximum column-generation iterations per B&B node.
            max_routes_per_iteration: Maximum routes generated per pricing call.
            optimality_gap: Reduced-cost convergence tolerance.
            use_ryan_foster: Deprecated — prefer ``branching_strategy``.
                When True and ``branching_strategy`` is at its default
                ``"edge"``, the strategy is silently promoted to
                ``"ryan_foster"``.
            branching_strategy: ``"edge"`` (default) or ``"ryan_foster"``.
                Takes precedence over ``use_ryan_foster``.
            tree_search_strategy: ``"best_first"`` (default) or ``"depth_first"``.
            max_branch_nodes: Maximum B&B nodes to explore.
            use_exact_pricing: Use exact DP RCSPP solver (True) or the greedy
                heuristic pricer (False).
            vehicle_limit: Maximum number of routes (vehicles), or None.
            use_ng_routes: Enable ng-route relaxation in the exact DP pricer
                (Baldacci et al. 2011).  Only relevant when
                ``use_exact_pricing=True``.  Default True.
            ng_neighborhood_size: Size of each node's ng-neighborhood N_i.
                Larger values tighten the relaxation (approach exact ESPPRC)
                at the cost of more labels.  Default 8.
            cleanup_frequency: Frequency (iterations) of column pool deletion.
            cleanup_threshold: Reduced cost threshold for column deletion.
            early_termination_gap: Duality gap for CG early termination.
        """
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes: Set[int] = mandatory_nodes or set()
        self.max_iterations = max_iterations
        self.max_routes_per_iteration = max_routes_per_iteration
        self.optimality_gap = optimality_gap
        self.max_branch_nodes = max_branch_nodes
        self.use_exact_pricing = use_exact_pricing
        self.vehicle_limit = vehicle_limit
        self.depot = 0
        self.use_ng_routes = use_ng_routes
        self.ng_neighborhood_size = ng_neighborhood_size
        self.cleanup_frequency = cleanup_frequency
        self.cleanup_threshold = cleanup_threshold
        self.early_termination_gap = early_termination_gap
        self.multiple_waste_types = multiple_waste_types
        self.allow_heuristic_ryan_foster = allow_heuristic_ryan_foster

        # Tracking exactness
        self.proven_optimal: bool = True

        # Resolve branching strategy.
        if branching_strategy == "edge" and use_ryan_foster:
            branching_strategy = "ryan_foster"

        if branching_strategy == "divergence" and not self.multiple_waste_types:
            logging.info(
                "Single-commodity VRP detected. Promoting 'divergence' branching to 'multi_edge_partition' strategy."
            )
            branching_strategy = "multi_edge_partition"

        self.branching_strategy: str = branching_strategy
        self.tree_search_strategy = tree_search_strategy
        self.use_ryan_foster: bool = branching_strategy == "ryan_foster"

        if self.use_ryan_foster and not self.allow_heuristic_ryan_foster:
            raise ValueError(
                "Ryan-Foster branching is theoretically inexact when used with a Set Covering "
                "master problem (the default). It can prune optimal over-covering solutions. "
                "To acknowledge this and proceed, pass 'allow_heuristic_ryan_foster=True' to the solver, "
                "or use the mathematically exact 'edge' branching strategy."
            )

        if self.use_ryan_foster:
            warnings.warn(
                "Ryan-Foster branching is theoretically a heuristic when used with a Set Covering "
                "master problem (>= 1). It may lead to unbalanced trees or over-coverage issues. "
                "Edge branching is the mathematically exact default.",
                UserWarning,
                stacklevel=2,
            )

        # Statistics
        self.num_iterations: int = 0
        self.num_columns_generated: int = 0
        self.lp_bound: float = 0.0
        self.ip_solution: float = 0.0
        self.tree: Optional[BranchAndBoundTree] = None

    # ------------------------------------------------------------------
    # Top-level solve
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Solve the VRPP using Branch-and-Price.

        Returns:
            Tuple of ``(tour, profit, statistics)``.
        """
        if self.use_ryan_foster:
            return self._solve_with_branching()  # type: ignore[return-value]
        return self._solve_without_branching()

    # ------------------------------------------------------------------
    # Solve without branching (column generation + direct IP)
    # ------------------------------------------------------------------

    def _solve_without_branching(self) -> Tuple[List[int], float, Dict[str, Any]]:
        """Solve with column generation followed by a direct MIP solve."""
        master = self._make_master()
        heuristic_pricing, exact_pricing = self._make_pricing()

        initial_routes = self._generate_initial_routes(heuristic_pricing)
        master.build_model(initial_routes)

        lp_obj, route_values = self._column_generation(master, (heuristic_pricing, exact_pricing))
        self.lp_bound = lp_obj

        if self._is_integer_solution(route_values):
            selected = [master.routes[i] for i, v in route_values.items() if v > 0.5]
            return self._routes_to_tour(selected), lp_obj, self._get_statistics()

        ip_obj, selected = master.solve_ip()
        self.ip_solution = ip_obj
        return self._routes_to_tour(selected), ip_obj, self._get_statistics()

    # ------------------------------------------------------------------
    # Solve with branching
    # ------------------------------------------------------------------

    def _solve_with_branching(self) -> Tuple[List[int], Optional[float], Dict[str, Any]]:
        """Solve using the configured branching strategy in a B&B tree."""
        self.tree = BranchAndBoundTree(strategy=self.branching_strategy, search_strategy=self.tree_search_strategy)

        root_lp, root_values, root_routes = self._solve_node(self.tree.root)
        self.tree.root.lp_bound = root_lp
        self.tree.root.route_values = root_values
        self.tree.root.is_integer = self._is_integer_solution(root_values)
        self.lp_bound = root_lp  # type: ignore[assignment]

        if self.tree.root.is_integer:
            selected = [root_routes[i] for i, v in root_values.items() if v > 0.5]
            return self._routes_to_tour(selected), root_lp, self._get_statistics()

        self.tree.nodes_explored += 1

        while not self.tree.is_empty() and self.tree.nodes_explored < self.max_branch_nodes:
            node = self.tree.get_next_node()
            if node is None:
                break

            if (
                self.tree.best_integer_solution is not None
                and node.lp_bound is not None
                and node.lp_bound <= self.tree.best_integer_solution
            ):
                self.tree.nodes_pruned += 1
                continue

            lp_obj, route_values, routes = self._solve_node(node)
            node.lp_bound = lp_obj
            node.route_values = route_values

            if lp_obj is None:
                node.is_infeasible = True
                self.tree.nodes_pruned += 1
                continue

            if self._is_integer_solution(route_values):
                node.is_integer = True
                node.ip_solution = lp_obj
                if self.tree.update_incumbent(node, lp_obj):
                    self.ip_solution = lp_obj
                    self.tree.prune_by_bound()
                continue

            result = self.tree.branch(node, routes, route_values)
            if result is None:
                logging.warning(
                    "Branching stalled (no fractional candidates found). Falling back to restricted "
                    "MIP solve. This is a Primal Heuristic; global optimality is no longer guaranteed."
                )
                self.proven_optimal = False
                master = self._build_master_for_node(node, routes)
                ip_obj, selected = master.solve_ip()
                if ip_obj is not None:
                    node.is_integer = True
                    node.ip_solution = ip_obj
                    if self.tree.update_incumbent(node, ip_obj):
                        self.ip_solution = ip_obj
                        self.tree.prune_by_bound()
            else:
                left_child, right_child = result
                self.tree.add_node(left_child)
                self.tree.add_node(right_child)

        if self.tree.best_integer_node is not None:
            _, rv, routes = self._solve_node(self.tree.best_integer_node)
            if rv:
                selected = [routes[i] for i, v in rv.items() if v > 0.5]
                return (
                    self._routes_to_tour(selected),
                    self.tree.best_integer_solution,
                    self._get_statistics(),
                )

        root_master = self._build_master_for_node(self.tree.root, root_routes)
        ip_obj, selected = root_master.solve_ip()
        if ip_obj is not None:
            self.ip_solution = ip_obj
            return self._routes_to_tour(selected), ip_obj, self._get_statistics()

        return [0], 0.0, self._get_statistics()

    # ------------------------------------------------------------------
    # Node solving
    # ------------------------------------------------------------------

    def _solve_node(self, node: BranchNode) -> Tuple[Optional[float], Dict[int, float], List[Route]]:
        """Solve the LP relaxation at a single B&B node via column generation."""
        master = self._make_master()
        constraints = node.get_all_constraints()
        heuristic_pricing, exact_pricing = self._make_pricing()

        initial_routes = self._generate_initial_routes(heuristic_pricing)
        feasible = [r for r in initial_routes if node.is_route_feasible(r)]
        if not feasible:
            return None, {}, []

        master.build_model(feasible)

        try:
            lp_obj, route_values = self._column_generation_with_constraints(
                master, (heuristic_pricing, exact_pricing), node, constraints
            )
        except RuntimeError:
            return None, {}, []

        return lp_obj, route_values, master.routes

    # ------------------------------------------------------------------
    # Column generation
    # ------------------------------------------------------------------

    def _column_generation(
        self,
        master: VRPPMasterProblem,
        pricing: Tuple[PricingSubproblem, Optional[RCSPPSolver]],
    ) -> Tuple[float, Dict[int, float]]:
        """Perform unconstrained column generation with heuristic fallback."""
        heuristic_pricer, exact_pricer = pricing

        for i in range(self.max_iterations):
            self.num_iterations += 1
            lp_obj, route_values = master.solve_lp_relaxation()
            dual_values = master.get_reduced_cost_coefficients()

            # 1. Try heuristic pricer first (Two-Phase Pricing)
            new_routes = self._call_pricing(heuristic_pricer, dual_values, constraints=None)  # type: ignore[arg-type]

            # 2. Fallback to exact pricer if heuristic fails
            if (not new_routes or max(rc for _, rc in new_routes) < self.optimality_gap) and exact_pricer is not None:
                new_routes = self._call_pricing(exact_pricer, dual_values, constraints=None)  # type: ignore[arg-type]

            if not new_routes:
                break

            # Task 4 & Part 5 Task 3: LP Termination via Lagrangian Bound
            max_rc = max(rc for _, rc in new_routes)
            if max_rc < self.optimality_gap:
                break

            # Estimate fleet limit for unconstrained instances to calculate Lagrangian bound
            limit = self.vehicle_limit if self.vehicle_limit is not None else self.n_nodes
            # Note: self.n_nodes is a true math upper bound (one vehicle per customer).
            # Tighter heuristic approximation could be max(self.n_nodes/2, 5).

            lagrangian_bound = lp_obj + max_rc * limit
            if abs(lp_obj - lagrangian_bound) < self.early_termination_gap:
                break

            self._add_routes_to_master(master, heuristic_pricer, new_routes)

            # Task 3: Column Pool Management
            if (i + 1) % self.cleanup_frequency == 0:
                master.remove_unpromising_columns(self.cleanup_threshold)

        return master.solve_lp_relaxation()

    def _column_generation_with_constraints(
        self,
        master: VRPPMasterProblem,
        pricing: Tuple[PricingSubproblem, Optional[RCSPPSolver]],
        node: BranchNode,
        constraints: List[AnyBranchingConstraint],
    ) -> Tuple[float, Dict[int, float]]:
        """Perform constrained column generation with heuristic fallback."""
        heuristic_pricer, exact_pricer = pricing

        for i in range(self.max_iterations):
            self.num_iterations += 1
            lp_obj, route_values = master.solve_lp_relaxation()
            dual_values = master.get_reduced_cost_coefficients()

            # 1. Try heuristic pricer first (Two-Phase Pricing)
            new_routes = self._call_pricing(heuristic_pricer, dual_values, constraints=constraints)  # type: ignore[arg-type]

            # 2. Fallback to exact pricer if heuristic fails
            if (not new_routes or max(rc for _, rc in new_routes) < self.optimality_gap) and exact_pricer is not None:
                new_routes = self._call_pricing(exact_pricer, dual_values, constraints=constraints)  # type: ignore[arg-type]

            if not new_routes:
                break

            feasible: List[Tuple[Route, float]] = []
            for route_nodes, rc in new_routes:
                cost, revenue, load, cov = heuristic_pricer.compute_route_details(route_nodes)
                route = Route(nodes=route_nodes, cost=cost, revenue=revenue, load=load, node_coverage=cov)
                # Exact solver already respects constraints via state enforcement.
                # Heuristic pricer requires post-hoc check for non-edge constraints (like Ryan-Foster).
                if (self.use_exact_pricing and exact_pricer is not None) or node.is_route_feasible(route):
                    feasible.append((route, rc))

            if not feasible:
                break

            # Task 4 & Part 5 Task 3: LP Termination via Lagrangian Bound
            max_rc = max(rc for _, rc in feasible)
            if max_rc < self.optimality_gap:
                break

            limit = self.vehicle_limit if self.vehicle_limit is not None else self.n_nodes
            lagrangian_bound = lp_obj + max_rc * limit
            if abs(lp_obj - lagrangian_bound) < self.early_termination_gap:
                break

            for route, _ in feasible:
                master.add_route(route)
                self.num_columns_generated += 1

            # Task 3: Column Pool Management
            if (i + 1) % self.cleanup_frequency == 0:
                master.remove_unpromising_columns(self.cleanup_threshold)

        return master.solve_lp_relaxation()

    # ------------------------------------------------------------------
    # Pricing dispatch
    # ------------------------------------------------------------------

    def _call_pricing(
        self,
        pricing: Any,
        dual_values: Dict[int, float],
        constraints: Optional[List[AnyBranchingConstraint]],
    ) -> List[Tuple[List[int], float]]:
        """
        Dispatch a pricing call to the correct solver with the right kwargs.

        RCSPPSolver uses ``branching_constraints``; PricingSubproblem uses
        ``active_constraints``.
        """
        if self.use_exact_pricing:
            return pricing.solve(
                dual_values=dual_values,
                max_routes=self.max_routes_per_iteration,
                branching_constraints=constraints,
            )
        else:
            return pricing.solve(
                dual_values=dual_values,
                max_routes=self.max_routes_per_iteration,
                active_constraints=constraints,
            )

    # ------------------------------------------------------------------
    # Initial column generation
    # ------------------------------------------------------------------

    def _generate_initial_routes(self, pricing: Any) -> List[Route]:
        """
        Generate high-quality initial feasible columns for the master problem.

        Uses a Greedy Nearest Neighbor heuristic to cover all mandatory nodes
        with feasible routes, obeying capacity constraints. This provides a
        stronger initial LP bound and more stable early dual variables than
        simple artificial out-and-back routes.

        Args:
            pricing: The pricing subproblem (used to compute cost/revenue).

        Returns:
            List of feasible Route objects covering all mandatory nodes.
        """
        routes: List[Route] = []
        uncovered = sorted(list(self.mandatory_nodes))

        while uncovered:
            current_route: List[int] = []
            current_load = 0.0
            current_node = self.depot

            while True:
                # Find nearest uncovered mandatory node that fits.
                best_next = None
                best_dist = float("inf")

                for candidate in uncovered:
                    demand = self.wastes.get(candidate, 0.0)
                    if current_load + demand <= self.capacity:
                        dist = self.cost_matrix[current_node, candidate]
                        if dist < best_dist:
                            best_dist = dist
                            best_next = candidate

                if best_next is None:
                    break  # Capacity reached or no closer nodes.

                current_route.append(best_next)
                current_load += self.wastes.get(best_next, 0.0)
                uncovered.remove(best_next)
                current_node = best_next

            # Finalize route return to depot.
            if current_route:
                cost, revenue, load, coverage = pricing.compute_route_details(current_route)
                routes.append(
                    Route(
                        nodes=current_route,
                        cost=cost,
                        revenue=revenue,
                        load=load,
                        node_coverage=coverage,
                    )
                )

        return routes

    def _add_routes_to_master(
        self,
        master: VRPPMasterProblem,
        pricing: Any,
        new_routes: List[Tuple[List[int], float]],
    ) -> None:
        """Add a batch of new route candidates to the master problem."""
        for route_nodes, _ in new_routes:
            cost, revenue, load, cov = pricing.compute_route_details(route_nodes)
            route = Route(nodes=route_nodes, cost=cost, revenue=revenue, load=load, node_coverage=cov)
            master.add_route(route)
            self.num_columns_generated += 1

    # ------------------------------------------------------------------
    # Tour construction with deduplication
    # ------------------------------------------------------------------

    def _routes_to_tour(self, routes: List[Route]) -> List[int]:
        """
        Convert selected routes to a flat tour, deduplicating customer visits.

        Because the master problem uses Set Covering (≥ 1), the IP may select
        overlapping routes.  Each customer node is included exactly once in the
        output; subsequent routes that would revisit a node simply skip it.
        Empty routes (all nodes already visited) are omitted to avoid bare
        depot → depot arcs.

        Returns:
            Flat tour: [depot, n1, n2, ..., depot, n3, ..., depot].
        """
        tour: List[int] = [self.depot]
        visited_nodes: Set[int] = {self.depot}

        for route in routes:
            unique_nodes = [n for n in route.nodes if n not in visited_nodes]
            if not unique_nodes:
                continue
            tour.extend(unique_nodes)
            tour.append(self.depot)
            visited_nodes.update(unique_nodes)

        return tour

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_master(self) -> VRPPMasterProblem:
        """Construct a fresh VRPPMasterProblem with the solver's parameters."""
        return VRPPMasterProblem(
            n_nodes=self.n_nodes,
            mandatory_nodes=self.mandatory_nodes,
            cost_matrix=self.cost_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue_per_kg=self.R,
            cost_per_km=self.C,
            vehicle_limit=self.vehicle_limit,
        )

    def _make_pricing(self) -> Tuple[PricingSubproblem, Optional[RCSPPSolver]]:
        """
        Construct the pricing solvers for this configuration.

        Returns:
            Tuple of (heuristic_pricer, exact_pricer).
            heuristic_pricer: Always provided (PricingSubproblem).
            exact_pricer: RCSPPSolver if use_exact_pricing=True, else None.
        """
        kwargs: Dict[str, Any] = dict(
            n_nodes=self.n_nodes,
            cost_matrix=self.cost_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue_per_kg=self.R,
            cost_per_km=self.C,
            mandatory_nodes=self.mandatory_nodes,
        )

        heuristic_pricer = PricingSubproblem(**kwargs)
        exact_pricer = None

        if self.use_exact_pricing:
            exact_pricer = RCSPPSolver(
                **kwargs,
                use_ng_routes=self.use_ng_routes,
                ng_neighborhood_size=self.ng_neighborhood_size,
            )

        return heuristic_pricer, exact_pricer

    def _build_master_for_node(self, node: BranchNode, routes: List[Route]) -> VRPPMasterProblem:
        """Build a master problem pre-loaded with constraint-feasible routes."""
        master = self._make_master()
        feasible = [r for r in routes if node.is_route_feasible(r)]
        master.build_model(feasible)
        return master

    def _is_integer_solution(self, route_values: Dict[int, float], tol: float = 1e-4) -> bool:
        """Return True if every LP route value is within *tol* of 0 or 1."""
        return all(abs(v - round(v)) <= tol for v in route_values.values())

    def _get_statistics(self) -> Dict[str, Any]:
        """Collect and return solver statistics."""
        stats: Dict[str, Any] = {
            "num_iterations": self.num_iterations,
            "num_columns_generated": self.num_columns_generated,
            "lp_bound": self.lp_bound,
            "ip_solution": self.ip_solution,
            "gap": abs(self.lp_bound - self.ip_solution) / max(abs(self.lp_bound), 1e-10),
            "branching_strategy": self.branching_strategy,
            "use_ng_routes": self.use_ng_routes,
            "ng_neighborhood_size": self.ng_neighborhood_size,
            "proven_optimal": self.proven_optimal,
        }
        if self.tree is not None:
            stats.update(self.tree.get_statistics())
        return stats
