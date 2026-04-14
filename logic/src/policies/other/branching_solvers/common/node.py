"""
Branch-and-Bound node data structure for VRPP.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from logic.src.policies.other.branching_solvers.branching.constraints import AnyBranchingConstraint
    from logic.src.policies.other.branching_solvers.common.route import Route


class BranchNode:
    """
    A single node in the Branch-and-Price-and-Cut search tree.

    Holds the branch-specific constraints and the resulting LP relaxation
    bound discovered during column generation at this node.

    Attributes:
        constraints: List of EdgeBranchingConstraint or
            RyanFosterBranchingConstraint objects active at this node.
        lp_bound: Objective value of the RMP linear relaxation (z_LP).
            Serves as a dual bound for all child nodes.
        lp_basis: Tuple of (vbasis, cbasis) Gurobi attributes used to warm-start
            the master problem solve from the parent's solution.
        routes: Optional subset of columns (Route objects) maintained
            locally to accelerate RMP convergence.
    """

    def __init__(
        self,
        constraints: Optional[List["AnyBranchingConstraint"]] = None,
        parent: Optional["BranchNode"] = None,
        depth: int = 0,
        lp_bound_hint: Optional[float] = None,
        branching_rule: str = "none",
    ) -> None:
        """
        Initialise a branch node.

        Args:
            constraints: Constraints added *at this node only* (not inherited).
            parent: Parent node (None for root).
            depth: Tree depth (root = 0).
            lp_bound_hint: Optional initial LP bound (inherited from parent).
            branching_rule: Name of the rule used to create this node's branch.
        """
        self.constraints: List["AnyBranchingConstraint"] = constraints or []
        self.parent: Optional["BranchNode"] = parent
        self.depth: int = depth
        self.branching_rule = branching_rule

        # State tracking
        self.lp_bound: Optional[float] = lp_bound_hint
        self.ip_solution: Optional[float] = None
        self.is_integer: bool = False
        self.is_infeasible: bool = False
        self.is_explored: bool = False
        self.is_pruned: bool = False

        # Solution data
        self.route_values: Optional[Dict[int, float]] = None
        self.routes: Optional[List["Route"]] = None
        self.lp_basis: Optional[Any] = None

        self.children: List["BranchNode"] = []

    def get_all_constraints(self) -> List["AnyBranchingConstraint"]:
        """
        Return all active constraints from the root down to this node.

        Returns:
            Flat list in root-to-leaf order.
        """
        constraints: List["AnyBranchingConstraint"] = []
        node: Optional["BranchNode"] = self
        while node is not None:
            # Reverse node local constraints to keep consistent order if multiple exist
            constraints.extend(node.constraints[::-1])
            node = node.parent
        return constraints[::-1]

    def is_route_feasible(self, route: "Route") -> bool:
        """
        Return True if *route* satisfies every inherited constraint.

        Args:
            route: Route to validate.
        """
        return all(c.is_route_feasible(route) for c in self.get_all_constraints())

    def __lt__(self, other: "BranchNode") -> bool:
        """Priority for best-first search (higher bound is better for maximization)."""
        b1 = self.lp_bound if self.lp_bound is not None else -1e10
        b2 = other.lp_bound if other.lp_bound is not None else -1e10
        return b1 > b2

    def __repr__(self) -> str:
        bound_str = f"{self.lp_bound:.2f}" if self.lp_bound is not None else "None"
        return f"BranchNode(depth={self.depth}, bound={bound_str}, constraints={len(self.constraints)})"
