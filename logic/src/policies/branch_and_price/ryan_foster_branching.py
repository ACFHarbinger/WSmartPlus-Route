"""
Ryan-Foster Branching for Set Partitioning Problems.

Implements the branching scheme described in:
Ryan, D. M., & Foster, B. A. (1981). "An Integer Programming Approach to Scheduling".
In Wren, A. (ed.) Computer Scheduling of Public Transport, pp. 269-280.

The key idea is to branch on pairs of nodes (r, s) rather than individual variables:
- Left branch: Nodes r and s must be in the SAME route
- Right branch: Nodes r and s must be in DIFFERENT routes

This preserves the pricing subproblem structure better than standard variable branching.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .master_problem import Route


class BranchingConstraint:
    """Represents a Ryan-Foster branching constraint."""

    def __init__(self, node_r: int, node_s: int, together: bool):
        """
        Initialize a branching constraint.

        Args:
            node_r: First node in the pair
            node_s: Second node in the pair
            together: True if r and s must be together, False if separate
        """
        self.node_r = node_r
        self.node_s = node_s
        self.together = together

    def is_route_feasible(self, route: Route) -> bool:
        """
        Check if a route satisfies this branching constraint.

        Args:
            route: Route to check

        Returns:
            True if route satisfies constraint, False otherwise
        """
        r_in_route = self.node_r in route.node_coverage
        s_in_route = self.node_s in route.node_coverage

        if self.together:
            # Both must be in route or both must be out
            return r_in_route == s_in_route
        else:
            # At most one can be in route
            return not (r_in_route and s_in_route)

    def __repr__(self) -> str:
        relation = "TOGETHER" if self.together else "SEPARATE"
        return f"Constraint({self.node_r}, {self.node_s}: {relation})"


class BranchNode:
    """Represents a node in the branch-and-bound tree."""

    def __init__(
        self,
        constraints: Optional[List[BranchingConstraint]] = None,
        parent: Optional["BranchNode"] = None,
        depth: int = 0,
    ):
        """
        Initialize a branch node.

        Args:
            constraints: List of branching constraints at this node
            parent: Parent node in the tree
            depth: Depth in the tree (0 = root)
        """
        self.constraints = constraints if constraints else []
        self.parent = parent
        self.depth = depth
        self.lp_bound: Optional[float] = None
        self.ip_solution: Optional[float] = None
        self.is_integer: bool = False
        self.is_infeasible: bool = False
        self.route_values: Optional[Dict[int, float]] = None

    def get_all_constraints(self) -> List[BranchingConstraint]:
        """
        Get all constraints from root to this node.

        Returns:
            List of all active branching constraints
        """
        constraints = []
        node = self
        while node is not None:
            constraints.extend(node.constraints)
            node = node.parent
        return constraints

    def is_route_feasible(self, route: Route) -> bool:
        """
        Check if a route satisfies all branching constraints.

        Args:
            route: Route to check

        Returns:
            True if route is feasible under all constraints
        """
        return all(constraint.is_route_feasible(route) for constraint in self.get_all_constraints())


class RyanFosterBranching:
    """
    Ryan-Foster branching strategy for set partitioning problems.

    Implements Proposition 1 from Ryan & Foster (1981):
    If the LP solution is fractional, there exist two rows r and s such that:
        0 < Σ_{k: y_rk = y_sk = 1} λ_k < 1

    We branch by enforcing:
    - Left: Σ_{k: y_rk = y_sk = 1} λ_k = 1  (r and s together)
    - Right: Σ_{k: y_rk = y_sk = 1} λ_k = 0  (r and s separate)
    """

    @staticmethod
    def find_branching_pair(
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-6,
    ) -> Optional[Tuple[int, int]]:
        """
        Find a pair of nodes (r, s) to branch on.

        Implements the search described in Proposition 1 of Ryan & Foster (1981).

        Args:
            routes: List of routes in the master problem
            route_values: Current LP solution values
            tol: Numerical tolerance

        Returns:
            Tuple (node_r, node_s) to branch on, or None if solution is integer
        """
        # Find a fractional variable
        fractional_route_idx = None
        for idx, val in route_values.items():
            if abs(val - round(val)) > tol:
                fractional_route_idx = idx
                break

        if fractional_route_idx is None:
            # No fractional variables found
            return None

        # Get the fractional route
        fractional_route = routes[fractional_route_idx]
        nodes_in_route = sorted(fractional_route.node_coverage)

        if len(nodes_in_route) < 2:
            # Route has only one node, can't branch on pairs
            return None

        # Select first node r from the fractional route
        node_r = nodes_in_route[0]

        # Find another node s such that 0 < Σ_{k: r,s together} λ_k < 1
        for node_s in nodes_in_route[1:]:
            # Calculate sum of λ for routes containing both r and s
            sum_together = 0.0

            for idx, val in route_values.items():
                route = routes[idx]
                if node_r in route.node_coverage and node_s in route.node_coverage:
                    sum_together += val

            # Check if this gives a valid branching pair
            if tol < sum_together < 1.0 - tol:
                return (node_r, node_s)

        # Try other combinations if first node didn't work
        for i, node_r in enumerate(nodes_in_route):
            for node_s in nodes_in_route[i + 1 :]:
                sum_together = 0.0

                for idx, val in route_values.items():
                    route = routes[idx]
                    if node_r in route.node_coverage and node_s in route.node_coverage:
                        sum_together += val

                if tol < sum_together < 1.0 - tol:
                    return (node_r, node_s)

        # Couldn't find a valid branching pair (shouldn't happen if solution is fractional)
        return None

    @staticmethod
    def create_child_nodes(
        parent: BranchNode,
        node_r: int,
        node_s: int,
    ) -> Tuple[BranchNode, BranchNode]:
        """
        Create two child nodes with branching constraints.

        Args:
            parent: Parent node
            node_r: First node in branching pair
            node_s: Second node in branching pair

        Returns:
            Tuple of (left_child, right_child)
            - left_child: r and s must be TOGETHER
            - right_child: r and s must be SEPARATE
        """
        # Left child: r and s together
        left_constraint = BranchingConstraint(node_r, node_s, together=True)
        left_child = BranchNode(
            constraints=[left_constraint],
            parent=parent,
            depth=parent.depth + 1,
        )

        # Right child: r and s separate
        right_constraint = BranchingConstraint(node_r, node_s, together=False)
        right_child = BranchNode(
            constraints=[right_constraint],
            parent=parent,
            depth=parent.depth + 1,
        )

        return left_child, right_child

    @staticmethod
    def modify_pricing_for_constraint(
        constraint: BranchingConstraint,
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Modify the pricing subproblem to enforce a branching constraint.

        Args:
            constraint: Branching constraint to enforce
            cost_matrix: Original distance matrix
            wastes: Original waste volumes

        Returns:
            Tuple of (modified_cost_matrix, modified_wastes)
        """
        modified_cost = cost_matrix.copy()
        modified_wastes = wastes.copy()

        if constraint.together:
            # r and s must be together: Merge them into a single "super-node"
            # This is handled in the pricing subproblem by always including both
            # For now, we don't modify the problem structure
            pass
        else:
            # r and s must be separate: Forbid them from being in the same route
            # This can be enforced by making the edge (r, s) infinitely expensive
            r, s = constraint.node_r, constraint.node_s
            modified_cost[r, s] = float("inf")
            modified_cost[s, r] = float("inf")

        return modified_cost, modified_wastes


class BranchAndBoundTree:
    """
    Manages the branch-and-bound tree for Ryan-Foster branching.

    Uses best-first search strategy (best LP bound).
    """

    def __init__(self):
        """Initialize the branch-and-bound tree."""
        self.root = BranchNode()
        self.open_nodes: List[BranchNode] = [self.root]
        self.best_integer_solution: Optional[float] = None
        self.best_integer_node: Optional[BranchNode] = None
        self.nodes_explored: int = 0
        self.nodes_pruned: int = 0

    def get_next_node(self) -> Optional[BranchNode]:
        """
        Get the next node to process (best-first strategy).

        Returns:
            Node with best LP bound, or None if tree is empty
        """
        if not self.open_nodes:
            return None

        # Sort by LP bound (descending for maximization)
        self.open_nodes.sort(
            key=lambda n: n.lp_bound if n.lp_bound is not None else float("-inf"),
            reverse=True,
        )

        return self.open_nodes.pop(0)

    def add_node(self, node: BranchNode) -> None:
        """
        Add a node to the open list.

        Args:
            node: Node to add
        """
        self.open_nodes.append(node)

    def prune_by_bound(self) -> int:
        """
        Prune nodes whose LP bound is worse than current best integer solution.

        Returns:
            Number of nodes pruned
        """
        if self.best_integer_solution is None:
            return 0

        original_count = len(self.open_nodes)

        # Keep only nodes with LP bound better than incumbent
        self.open_nodes = [
            node for node in self.open_nodes if node.lp_bound is None or node.lp_bound > self.best_integer_solution
        ]

        pruned = original_count - len(self.open_nodes)
        self.nodes_pruned += pruned

        return pruned

    def update_incumbent(self, node: BranchNode, solution_value: float) -> bool:
        """
        Update the best integer solution if this is an improvement.

        Args:
            node: Node with integer solution
            solution_value: Objective value

        Returns:
            True if incumbent was updated
        """
        if self.best_integer_solution is None or solution_value > self.best_integer_solution:
            self.best_integer_solution = solution_value
            self.best_integer_node = node
            return True
        return False

    def is_empty(self) -> bool:
        """Check if there are no more nodes to explore."""
        return len(self.open_nodes) == 0

    def get_statistics(self) -> Dict:
        """Get tree statistics."""
        return {
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
            "nodes_remaining": len(self.open_nodes),
            "best_bound": self.open_nodes[0].lp_bound if self.open_nodes else None,
            "best_integer": self.best_integer_solution,
        }
