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
        Branches on fractional node-pair co-occurrence (Ryan & Foster 1981).

    Exact-pricing mode
    ------------------
    When ``use_exact_pricing=True``, the DP-based ``RCSPPSolver`` is used.
    Two sub-modes are available via ``use_ng_routes``:

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
        max_branch_nodes: int = 1000,
        use_exact_pricing: bool = False,
        vehicle_limit: Optional[int] = None,
        use_ng_routes: bool = True,
        ng_neighborhood_size: int = 8,
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

        # Resolve branching strategy.
        if branching_strategy == "edge" and use_ryan_foster:
            branching_strategy = "ryan_foster"
        self.branching_strategy: str = branching_strategy
        self.use_ryan_foster: bool = branching_strategy == "ryan_foster"

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
        pricing = self._make_pricing()

        initial_routes = self._generate_initial_routes(pricing)
        master.build_model(initial_routes)

        lp_obj, route_values = self._column_generation(master, pricing)
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
        self.tree = BranchAndBoundTree(strategy=self.branching_strategy)

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
        pricing = self._make_pricing()

        initial_routes = self._generate_initial_routes(pricing)
        feasible = [r for r in initial_routes if node.is_route_feasible(r)]
        if not feasible:
            return None, {}, []

        master.build_model(feasible)

        try:
            lp_obj, route_values = self._column_generation_with_constraints(master, pricing, node, constraints)
        except RuntimeError:
            return None, {}, []

        return lp_obj, route_values, master.routes

    # ------------------------------------------------------------------
    # Column generation
    # ------------------------------------------------------------------

    def _column_generation(
        self,
        master: VRPPMasterProblem,
        pricing: Any,
    ) -> Tuple[float, Dict[int, float]]:
        """Perform unconstrained column generation (root / no-branching path)."""
        for _ in range(self.max_iterations):
            self.num_iterations += 1
            lp_obj, route_values = master.solve_lp_relaxation()
            dual_values = master.get_reduced_cost_coefficients()

            new_routes = self._call_pricing(pricing, dual_values, constraints=None)  # type: ignore[arg-type]
            if not new_routes:
                break
            if max(rc for _, rc in new_routes) < self.optimality_gap:
                break

            self._add_routes_to_master(master, pricing, new_routes)

        return master.solve_lp_relaxation()

    def _column_generation_with_constraints(
        self,
        master: VRPPMasterProblem,
        pricing: Any,
        node: BranchNode,
        constraints: List[AnyBranchingConstraint],
    ) -> Tuple[float, Dict[int, float]]:
        """Perform column generation at a constrained B&B node."""
        for _ in range(self.max_iterations):
            self.num_iterations += 1
            lp_obj, route_values = master.solve_lp_relaxation()
            dual_values = master.get_reduced_cost_coefficients()

            new_routes = self._call_pricing(pricing, dual_values, constraints=constraints)  # type: ignore[arg-type]
            if not new_routes:
                break

            feasible: List[Tuple[Route, float]] = []
            for route_nodes, rc in new_routes:
                cost, revenue, load, cov = pricing.compute_route_details(route_nodes)
                route = Route(nodes=route_nodes, cost=cost, revenue=revenue, load=load, node_coverage=cov)
                if self.use_exact_pricing or node.is_route_feasible(route):
                    feasible.append((route, rc))

            if not feasible:
                break
            if max(rc for _, rc in feasible) < self.optimality_gap:
                break

            for route, _ in feasible:
                master.add_route(route)
                self.num_columns_generated += 1

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
        """Generate a seed column set using zero dual values."""
        dummy_duals: Dict[int, float] = {n: 0.0 for n in range(1, self.n_nodes + 1)}
        candidates = self._call_pricing(pricing, dummy_duals, constraints=None)
        routes: List[Route] = []
        for route_nodes, _ in candidates[: min(self.n_nodes, 20)]:
            cost, revenue, load, cov = pricing.compute_route_details(route_nodes)
            routes.append(Route(nodes=route_nodes, cost=cost, revenue=revenue, load=load, node_coverage=cov))
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

    def _make_pricing(self) -> Any:
        """
        Construct the appropriate pricing solver for this configuration.

        When ``use_exact_pricing=True``, the RCSPPSolver receives the ng-route
        parameters so that the relaxation mode is correctly propagated.
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
        if self.use_exact_pricing:
            return RCSPPSolver(
                **kwargs,
                use_ng_routes=self.use_ng_routes,
                ng_neighborhood_size=self.ng_neighborhood_size,
            )
        return PricingSubproblem(**kwargs)

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
        }
        if self.tree is not None:
            stats.update(self.tree.get_statistics())
        return stats
