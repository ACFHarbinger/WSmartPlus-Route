"""
Branch-and-Price Solver for VRPP.

Implements the complete Branch-and-Price algorithm with:
- Column generation at each node
- Ryan-Foster branching for set partitioning
- LP relaxation solving with pricing
- Integer solution recovery

Based on Sections 5 and 6 of Barnhart et al. (1998).

Reference:
    Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W. P., & Vance, P. H. (1998).
    "Branch-and-price: Column Generation for Solving Huge Integer Programs".
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .master_problem import Route, VRPPMasterProblem
from .pricing_subproblem import PricingSubproblem
from .rcspp_dp import RCSPPSolver
from .ryan_foster_branching import (
    BranchAndBoundTree,
    BranchNode,
    RyanFosterBranching,
)


class BranchAndPriceSolver:
    """
    Branch-and-Price Algorithm for VRPP with Column Generation.

    Algorithm:
        1. Initialize with greedy routes
        2. Solve LP relaxation with column generation
        3. If LP solution is integer, terminate
        4. Otherwise, branch using Ryan-Foster branching
        5. Solve child nodes recursively
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
        use_ryan_foster: bool = True,
        max_branch_nodes: int = 1000,
        use_exact_pricing: bool = False,
    ):
        """
        Initialize the Branch-and-Price solver.

        Args:
            n_nodes: Number of customer nodes (excluding depot)
            cost_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 is depot
            wastes: Dictionary mapping node ID to waste volume
            capacity: Vehicle capacity
            revenue_per_kg: Revenue per unit of waste collected
            cost_per_km: Cost per unit of distance traveled
            mandatory_nodes: Set of node indices that must be visited
            max_iterations: Maximum column generation iterations per node
            max_routes_per_iteration: Maximum routes to generate per pricing call
            optimality_gap: Convergence tolerance for column generation
            use_ryan_foster: Use Ryan-Foster branching (True) or direct IP (False)
            max_branch_nodes: Maximum nodes to explore in branch-and-bound tree
            use_exact_pricing: Use exact DP RCSPP solver (True) or heuristic (False)
        """
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes = mandatory_nodes if mandatory_nodes else set()
        self.max_iterations = max_iterations
        self.max_routes_per_iteration = max_routes_per_iteration
        self.optimality_gap = optimality_gap
        self.use_ryan_foster = use_ryan_foster
        self.max_branch_nodes = max_branch_nodes
        self.use_exact_pricing = use_exact_pricing
        self.depot = 0

        # Statistics
        self.num_iterations = 0
        self.num_columns_generated = 0
        self.lp_bound = 0.0
        self.ip_solution = 0.0
        self.tree: Optional[BranchAndBoundTree] = None

    def solve(self) -> Tuple[List[int], float, Dict]:
        """
        Solve the VRPP using Branch-and-Price.

        Returns:
            Tuple of (tour, profit, statistics)
        """
        if self.use_ryan_foster:
            return self._solve_with_branching()
        else:
            return self._solve_without_branching()

    def _solve_without_branching(self) -> Tuple[List[int], float, Dict]:
        """
        Solve using column generation followed by direct IP solving.

        Returns:
            Tuple of (tour, profit, statistics)
        """
        # Initialize master problem
        master = VRPPMasterProblem(
            n_nodes=self.n_nodes,
            mandatory_nodes=self.mandatory_nodes,
            cost_matrix=self.cost_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue_per_kg=self.R,
            cost_per_km=self.C,
        )

        # Initialize pricing solver (exact DP or heuristic)
        if self.use_exact_pricing:
            pricing = RCSPPSolver(
                n_nodes=self.n_nodes,
                cost_matrix=self.cost_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                revenue_per_kg=self.R,
                cost_per_km=self.C,
                mandatory_nodes=self.mandatory_nodes,
            )
        else:
            pricing = PricingSubproblem(
                n_nodes=self.n_nodes,
                cost_matrix=self.cost_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                revenue_per_kg=self.R,
                cost_per_km=self.C,
                mandatory_nodes=self.mandatory_nodes,
            )

        # Generate initial columns using greedy heuristic
        initial_routes = self._generate_initial_routes(pricing)
        master.build_model(initial_routes)

        # Solve LP relaxation with column generation
        lp_obj, route_values = self._column_generation(master, pricing)
        self.lp_bound = lp_obj

        # Check if LP solution is integer
        if self._is_integer_solution(route_values):
            # Extract integer solution
            selected_routes = [master.routes[idx] for idx, val in route_values.items() if val > 0.5]
            tour = self._routes_to_tour(selected_routes)
            profit = lp_obj

            return tour, profit, self._get_statistics()

        # Solve IP directly
        ip_obj, selected_routes = master.solve_ip()
        self.ip_solution = ip_obj

        tour = self._routes_to_tour(selected_routes)

        return tour, ip_obj, self._get_statistics()

    def _solve_with_branching(self) -> Tuple[List[int], float, Dict]:
        """
        Solve using Ryan-Foster branching.

        Returns:
            Tuple of (tour, profit, statistics)
        """
        # Initialize branch-and-bound tree
        self.tree = BranchAndBoundTree()

        # Solve root node
        root_lp, root_values, root_routes = self._solve_node(self.tree.root)
        self.tree.root.lp_bound = root_lp
        self.tree.root.route_values = root_values
        self.tree.root.is_integer = self._is_integer_solution(root_values)

        self.lp_bound = root_lp

        if self.tree.root.is_integer:
            # Root is integer, we're done
            selected_routes = [root_routes[idx] for idx, val in root_values.items() if val > 0.5]
            tour = self._routes_to_tour(selected_routes)
            return tour, root_lp, self._get_statistics()

        # Add root to tree if not integer
        self.tree.nodes_explored += 1

        # Branch-and-bound loop
        while not self.tree.is_empty() and self.tree.nodes_explored < self.max_branch_nodes:
            # Get next node to process
            node = self.tree.get_next_node()

            if node is None:
                break

            # Check if node can be pruned by bound
            if (
                self.tree.best_integer_solution is not None
                and node.lp_bound is not None
                and node.lp_bound <= self.tree.best_integer_solution
            ):
                self.tree.nodes_pruned += 1
                continue

            # Solve node
            lp_obj, route_values, routes = self._solve_node(node)
            node.lp_bound = lp_obj
            node.route_values = route_values

            if lp_obj is None:
                # Node is infeasible
                node.is_infeasible = True
                self.tree.nodes_pruned += 1
                continue

            # Check if solution is integer
            if self._is_integer_solution(route_values):
                node.is_integer = True
                node.ip_solution = lp_obj

                # Update incumbent
                if self.tree.update_incumbent(node, lp_obj):
                    self.ip_solution = lp_obj

                    # Prune nodes by bound
                    self.tree.prune_by_bound()

                continue

            # Solution is fractional, branch
            branching_pair = RyanFosterBranching.find_branching_pair(
                routes=routes,
                route_values=route_values,
            )

            if branching_pair is None:
                # Couldn't find branching pair, solve IP directly
                master = self._build_master_for_node(node, routes)
                ip_obj, selected_routes = master.solve_ip()

                if ip_obj is not None:
                    node.is_integer = True
                    node.ip_solution = ip_obj

                    if self.tree.update_incumbent(node, ip_obj):
                        self.ip_solution = ip_obj
                        self.tree.prune_by_bound()

                continue

            # Create child nodes
            node_r, node_s = branching_pair
            left_child, right_child = RyanFosterBranching.create_child_nodes(node, node_r, node_s)

            # Add children to tree
            self.tree.add_node(left_child)
            self.tree.add_node(right_child)

        # Extract best solution
        if self.tree.best_integer_node is not None:
            # Reconstruct solution from best node
            # For simplicity, resolve the node to get routes
            _, route_values, routes = self._solve_node(self.tree.best_integer_node)

            if route_values:
                selected_routes = [routes[idx] for idx, val in route_values.items() if val > 0.5]
                tour = self._routes_to_tour(selected_routes)
                return tour, self.tree.best_integer_solution, self._get_statistics()

        # Fallback: solve root IP if no integer solution found
        root_master = self._build_master_for_node(self.tree.root, root_routes)
        ip_obj, selected_routes = root_master.solve_ip()

        if ip_obj is not None:
            self.ip_solution = ip_obj
            tour = self._routes_to_tour(selected_routes)
            return tour, ip_obj, self._get_statistics()

        # No solution found
        return [0], 0.0, self._get_statistics()

    def _solve_node(self, node: BranchNode) -> Tuple[Optional[float], Dict[int, float], List[Route]]:
        """
        Solve a node in the branch-and-bound tree.

        Args:
            node: Node to solve

        Returns:
            Tuple of (lp_objective, route_values, routes)
            Returns (None, {}, []) if node is infeasible
        """
        # Build master problem with branching constraints
        master = VRPPMasterProblem(
            n_nodes=self.n_nodes,
            mandatory_nodes=self.mandatory_nodes,
            cost_matrix=self.cost_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue_per_kg=self.R,
            cost_per_km=self.C,
        )

        # Get all constraints for this node
        constraints = node.get_all_constraints()

        # Initialize pricing solver (exact DP or heuristic)
        if self.use_exact_pricing:
            pricing = RCSPPSolver(
                n_nodes=self.n_nodes,
                cost_matrix=self.cost_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                revenue_per_kg=self.R,
                cost_per_km=self.C,
                mandatory_nodes=self.mandatory_nodes,
            )
        else:
            pricing = PricingSubproblem(
                n_nodes=self.n_nodes,
                cost_matrix=self.cost_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                revenue_per_kg=self.R,
                cost_per_km=self.C,
                mandatory_nodes=self.mandatory_nodes,
            )

        # Generate initial routes respecting constraints
        initial_routes = self._generate_initial_routes(pricing)
        feasible_routes = [r for r in initial_routes if node.is_route_feasible(r)]

        if not feasible_routes:
            # No feasible initial routes
            return None, {}, []

        master.build_model(feasible_routes)

        # Column generation with constraint checking
        try:
            lp_obj, route_values = self._column_generation_with_constraints(master, pricing, node, constraints)
        except RuntimeError:
            # Infeasible
            return None, {}, []

        return lp_obj, route_values, master.routes

    def _build_master_for_node(self, node: BranchNode, routes: List[Route]) -> VRPPMasterProblem:
        """
        Build master problem for a node with existing routes.

        Args:
            node: Branch node
            routes: Existing routes

        Returns:
            Master problem instance
        """
        master = VRPPMasterProblem(
            n_nodes=self.n_nodes,
            mandatory_nodes=self.mandatory_nodes,
            cost_matrix=self.cost_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            revenue_per_kg=self.R,
            cost_per_km=self.C,
        )

        # Filter routes by branching constraints
        feasible_routes = [r for r in routes if node.is_route_feasible(r)]
        master.build_model(feasible_routes)

        return master

    def _column_generation_with_constraints(
        self,
        master: VRPPMasterProblem,
        pricing,  # Union[PricingSubproblem, RCSPPSolver]
        node: BranchNode,
        constraints: List,
    ) -> Tuple[float, Dict[int, float]]:
        """
        Perform column generation respecting branching constraints.

        Args:
            master: Master problem
            pricing: Pricing subproblem (heuristic or exact DP solver)
            node: Current branch node
            constraints: List of branching constraints

        Returns:
            Tuple of (lp_objective, route_values)
        """
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            self.num_iterations += 1

            # Solve master LP
            lp_obj, route_values = master.solve_lp_relaxation()

            # Get dual values
            dual_values = master.get_reduced_cost_coefficients()

            # Solve pricing problem (pass constraints if using exact DP solver)
            if self.use_exact_pricing:
                new_routes = pricing.solve(
                    dual_values=dual_values,
                    max_routes=self.max_routes_per_iteration,
                    branching_constraints=constraints,
                )
            else:
                new_routes = pricing.solve(
                    dual_values=dual_values,
                    max_routes=self.max_routes_per_iteration,
                )

            if not new_routes:
                break

            # Filter by branching constraints (only needed for heuristic pricing)
            # Exact DP solver already enforces constraints during labeling
            feasible_new_routes = []
            for route_nodes, reduced_cost in new_routes:
                cost, revenue, load, node_coverage = pricing.compute_route_details(route_nodes)

                route = Route(
                    nodes=route_nodes,
                    cost=cost,
                    revenue=revenue,
                    load=load,
                    node_coverage=node_coverage,
                )

                # Skip feasibility check if using exact pricing (already enforced)
                if self.use_exact_pricing or node.is_route_feasible(route):
                    feasible_new_routes.append((route, reduced_cost))

            if not feasible_new_routes:
                # No feasible routes with positive reduced cost
                break

            max_reduced_cost = max(rc for _, rc in feasible_new_routes)

            if max_reduced_cost < self.optimality_gap:
                break

            # Add feasible columns to master
            for route, _ in feasible_new_routes:
                master.add_route(route)
                self.num_columns_generated += 1

        # Solve final LP
        lp_obj, route_values = master.solve_lp_relaxation()

        return lp_obj, route_values

    def _generate_initial_routes(self, pricing) -> List[Route]:
        """
        Generate initial routes using greedy heuristic or exact DP.

        Args:
            pricing: Pricing solver (PricingSubproblem or RCSPPSolver)

        Returns:
            List of initial routes
        """
        routes = []

        # Use pricing solver with zero dual values to generate initial routes
        dummy_duals = {node: 0.0 for node in range(1, self.n_nodes + 1)}

        # For exact pricing, no constraints in initial route generation
        if self.use_exact_pricing:
            route_candidates = pricing.solve(
                dual_values=dummy_duals,
                max_routes=min(self.n_nodes, 20),
                branching_constraints=None,
            )
        else:
            route_candidates = pricing.solve(
                dual_values=dummy_duals,
                max_routes=min(self.n_nodes, 20),
            )

        for route_nodes, _ in route_candidates:
            cost, revenue, load, node_coverage = pricing.compute_route_details(route_nodes)

            route = Route(
                nodes=route_nodes,
                cost=cost,
                revenue=revenue,
                load=load,
                node_coverage=node_coverage,
            )
            routes.append(route)

        return routes

    def _column_generation(
        self,
        master: VRPPMasterProblem,
        pricing,  # Union[PricingSubproblem, RCSPPSolver]
    ) -> Tuple[float, Dict[int, float]]:
        """
        Perform column generation to solve LP relaxation.

        Args:
            master: Master problem
            pricing: Pricing solver (heuristic or exact DP)

        Returns:
            Tuple of (lp_objective, route_values)
        """
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            self.num_iterations += 1

            # Solve master LP
            lp_obj, route_values = master.solve_lp_relaxation()

            # Get dual values
            dual_values = master.get_reduced_cost_coefficients()

            # Solve pricing problem (no constraints in non-branching mode)
            if self.use_exact_pricing:
                new_routes = pricing.solve(
                    dual_values=dual_values,
                    max_routes=self.max_routes_per_iteration,
                    branching_constraints=None,
                )
            else:
                new_routes = pricing.solve(
                    dual_values=dual_values,
                    max_routes=self.max_routes_per_iteration,
                )

            if not new_routes:
                # No new columns with positive reduced cost
                break

            max_reduced_cost = max(rc for _, rc in new_routes)

            if max_reduced_cost < self.optimality_gap:
                # Convergence
                break

            # Add new columns to master
            for route_nodes, _reduced_cost in new_routes:
                cost, revenue, load, node_coverage = pricing.compute_route_details(route_nodes)

                route = Route(
                    nodes=route_nodes,
                    cost=cost,
                    revenue=revenue,
                    load=load,
                    node_coverage=node_coverage,
                )

                master.add_route(route)
                self.num_columns_generated += 1

        # Solve final LP
        lp_obj, route_values = master.solve_lp_relaxation()

        return lp_obj, route_values

    def _is_integer_solution(self, route_values: Dict[int, float], tol: float = 1e-4) -> bool:
        """
        Check if the LP solution is integer.

        Args:
            route_values: Route selection values
            tol: Tolerance for integrality

        Returns:
            True if all values are close to 0 or 1
        """
        return all(abs(val - round(val)) <= tol for val in route_values.values())

    def _routes_to_tour(self, routes: List[Route]) -> List[int]:
        """
        Convert routes to a single tour representation.

        Args:
            routes: List of selected routes

        Returns:
            Tour as [depot, node1, node2, ..., depot, node3, ..., depot]
        """
        tour = [self.depot]

        for route in routes:
            tour.extend(route.nodes)
            tour.append(self.depot)

        return tour

    def _get_statistics(self) -> Dict:
        """Get solver statistics."""
        stats = {
            "num_iterations": self.num_iterations,
            "num_columns_generated": self.num_columns_generated,
            "lp_bound": self.lp_bound,
            "ip_solution": self.ip_solution,
            "gap": abs(self.lp_bound - self.ip_solution) / max(abs(self.lp_bound), 1e-10),
        }

        if self.tree is not None:
            stats.update(self.tree.get_statistics())

        return stats
