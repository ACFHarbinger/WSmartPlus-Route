"""
Branching strategies for Branch-and-Price on Vehicle Routing Problems.

Provides two constraint types and the supporting B&B tree infrastructure:

EdgeBranchingConstraint
    Operates on directed arcs (u → v).  Integrates cleanly with the DP label
    extension step: forbidden / required arcs are enforced in O(1) per
    extension without any post-hoc filtering.

    Reference: Barnhart et al. (1998), Section 4.

RyanFosterBranchingConstraint
    Operates on *node pairs* (r, s).  Requires routes to either always contain
    both nodes in the same route (`together=True`) or never contain them
    together (`together=False`).

    Reference: Ryan & Foster (1981), Proposition 1.

Both constraint classes expose a common ``is_route_feasible(route)`` method
used by the master problem to filter its existing column pool whenever a new
B&B node is created.

Note on VRPP Selection:
----------------------
Ryan-Foster branching is utilized for VRPP because it appropriately modifies
the Resource-Constrained Shortest Path Problem (RCSPP) used for pricing by
enforcing or forbidding node pairs. This differs from Barnhart et al. (1998),
who primarily used divergence branching to maintain simple shortest paths
for the Origin-Destination Integer Multicommodity Flow (ODIMCF) problem.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

# Forward reference resolved at runtime — avoids a circular import with
# master_problem.py while still enabling full type annotations.
from .master_problem import Route

# ---------------------------------------------------------------------------
# Constraint classes
# ---------------------------------------------------------------------------


class EdgeBranchingConstraint:
    """
    Directed-arc branching constraint for edge-based B&P branching.

    Fixes or forbids a single arc (u → v) in the solution.  The pricing DP
    enforces this eagerly inside the label extension loop; the heuristic
    pricer also respects it during greedy insertion.

    Attributes:
        u: Origin node of the constrained arc.
        v: Destination node of the constrained arc.
        must_use: True  → arc (u, v) MUST appear in every route that visits u
                          (x_{uv} = 1 branch).
                  False → arc (u, v) is FORBIDDEN in all routes
                          (x_{uv} = 0 branch).
    """

    def __init__(self, u: int, v: int, must_use: bool) -> None:
        """
        Initialise an edge branching constraint.

        Args:
            u: Arc origin node.
            v: Arc destination node.
            must_use: Whether the arc must (True) or must not (False) be used.
        """
        self.u = u
        self.v = v
        self.must_use = must_use

    # ------------------------------------------------------------------
    # Feasibility check (used for legacy column-pool filtering)
    # ------------------------------------------------------------------

    def is_route_feasible(self, route: Route) -> bool:
        """
        Return True if *route* satisfies this edge constraint.

        Args:
            route: A Route object whose node sequence is validated.

        Returns:
            True when the constraint is not violated.
        """
        edge_present = self._edge_in_route(route.nodes)
        return edge_present if self.must_use else not edge_present

    def _edge_in_route(self, nodes: List[int]) -> bool:
        """Check whether arc (u, v) appears consecutively in the full path."""
        full_path = [0] + nodes + [0]
        return any(full_path[i] == self.u and full_path[i + 1] == self.v for i in range(len(full_path) - 1))

    def __repr__(self) -> str:
        relation = "MUST_USE" if self.must_use else "FORBIDDEN"
        return f"EdgeBranchingConstraint({self.u} -> {self.v}: {relation})"


class RyanFosterBranchingConstraint:
    """
    Node-pair branching constraint for Ryan-Foster B&P branching.

    Enforces co-occurrence or separation of two customer nodes across routes.

    Attributes:
        node_r: First node in the branching pair.
        node_s: Second node in the branching pair.
        together: True  → r and s MUST appear in the same route.
                  False → r and s MUST NOT appear in the same route.
    """

    def __init__(self, node_r: int, node_s: int, together: bool) -> None:
        """
        Initialise a Ryan-Foster branching constraint.

        Args:
            node_r: First node in the pair.
            node_s: Second node in the pair.
            together: Whether the two nodes must be co-visited (True) or
                separated (False).
        """
        self.node_r = node_r
        self.node_s = node_s
        self.together = together

    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------

    def is_route_feasible(self, route: Route) -> bool:
        """
        Return True if *route* satisfies this Ryan-Foster constraint.

        Args:
            route: A Route object to validate.

        Returns:
            True when the constraint is not violated.
        """
        r_in = self.node_r in route.node_coverage
        s_in = self.node_s in route.node_coverage

        if self.together:
            # Both must appear in every route that contains either one.
            # A route containing only one of them violates the constraint.
            if r_in != s_in:
                return False
        else:
            # The two nodes must never appear in the same route.
            if r_in and s_in:
                return False

        return True

    def __repr__(self) -> str:
        relation = "TOGETHER" if self.together else "SEPARATE"
        return f"RyanFosterBranchingConstraint({self.node_r}, {self.node_s}: {relation})"


# Backward-compatibility alias — existing code that imports BranchingConstraint
# by name continues to work without modification.
BranchingConstraint = EdgeBranchingConstraint

# Union type for callers that handle either constraint flavour.
AnyBranchingConstraint = Union[EdgeBranchingConstraint, RyanFosterBranchingConstraint]


# ---------------------------------------------------------------------------
# Branch node
# ---------------------------------------------------------------------------


class BranchNode:
    """A single node in the branch-and-bound tree."""

    def __init__(
        self,
        constraints: Optional[List[AnyBranchingConstraint]] = None,
        parent: Optional["BranchNode"] = None,
        depth: int = 0,
    ) -> None:
        """
        Initialise a branch node.

        Args:
            constraints: Constraints added *at this node only* (not inherited).
            parent: Parent node (None for root).
            depth: Tree depth (root = 0).
        """
        self.constraints: List[AnyBranchingConstraint] = constraints or []
        self.parent: Optional["BranchNode"] = parent
        self.depth: int = depth

        self.lp_bound: Optional[float] = None
        self.ip_solution: Optional[float] = None
        self.is_integer: bool = False
        self.is_infeasible: bool = False
        self.route_values: Optional[Dict[int, float]] = None

    def get_all_constraints(self) -> List[AnyBranchingConstraint]:
        """
        Return all active constraints from the root down to this node.

        Returns:
            Flat list in root-to-leaf order.
        """
        constraints: List[AnyBranchingConstraint] = []
        node: Optional["BranchNode"] = self
        while node is not None:
            constraints.extend(node.constraints)
            node = node.parent
        return constraints

    def is_route_feasible(self, route: Route) -> bool:
        """
        Return True if *route* satisfies every inherited constraint.

        Args:
            route: Route to validate.
        """
        return all(c.is_route_feasible(route) for c in self.get_all_constraints())


# ---------------------------------------------------------------------------
# Branching strategy helpers
# ---------------------------------------------------------------------------


class EdgeBranching:
    """
    Edge-based branching: select the most-fractional arc and split on it.

    Produces two child nodes:
        left  → x_{uv} = 1  (arc MUST be used)
        right → x_{uv} = 0  (arc is FORBIDDEN)
    """

    @staticmethod
    def compute_arc_flow(
        routes: List[Route],
        route_values: Dict[int, float],
    ) -> Dict[Tuple[int, int], float]:
        """
        Aggregate fractional arc flows from the LP solution.

        Args:
            routes: All routes in the master problem column pool.
            route_values: LP solution values {route_index: λ_k}.

        Returns:
            Mapping (u, v) → aggregated fractional flow.
        """
        arc_flow: Dict[Tuple[int, int], float] = {}
        for idx, lam in route_values.items():
            if lam < 1e-9:
                continue
            full_path = [0] + routes[idx].nodes + [0]
            for i in range(len(full_path) - 1):
                arc = (full_path[i], full_path[i + 1])
                arc_flow[arc] = arc_flow.get(arc, 0.0) + lam
        return arc_flow

    @staticmethod
    def find_branching_arc(
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-6,
    ) -> Optional[Tuple[int, int]]:
        """
        Select the arc with fractional flow closest to 0.5.

        Args:
            routes: All routes in the master problem.
            route_values: Current LP solution {route_index: λ_k}.
            tol: Integrality tolerance.

        Returns:
            Arc (u, v) to branch on, or None if the solution is already integer.
        """
        arc_flow = EdgeBranching.compute_arc_flow(routes, route_values)
        best_arc: Optional[Tuple[int, int]] = None
        best_frac = -1.0

        for arc, flow in arc_flow.items():
            frac = min(flow, 1.0 - flow)
            if frac > tol and frac > best_frac:
                best_frac = frac
                best_arc = arc

        return best_arc

    @staticmethod
    def create_child_nodes(
        parent: BranchNode,
        u: int,
        v: int,
    ) -> Tuple[BranchNode, BranchNode]:
        """
        Create left (must-use) and right (forbidden) child nodes.

        Args:
            parent: The node being branched.
            u: Arc origin.
            v: Arc destination.

        Returns:
            (left_child, right_child)
        """
        left = BranchNode(
            constraints=[EdgeBranchingConstraint(u, v, must_use=True)],
            parent=parent,
            depth=parent.depth + 1,
        )
        right = BranchNode(
            constraints=[EdgeBranchingConstraint(u, v, must_use=False)],
            parent=parent,
            depth=parent.depth + 1,
        )
        return left, right


class MultiEdgePartitionBranching:
    """
    Multi-edge partition branching (single-commodity adaptation of divergence branching).

    For single-commodity VRP, this partitions the outgoing edges of a node
    to break fractional solutions. In multicommodity contexts, this is
    equivalent to Divergence Node Branching (Barnhart et al. 1998).
    branching on divergence nodes is more effective than Ryan-Foster branching.

    Concept:
    --------
    A divergence node d for commodity k is a node where the fractional flow
    splits among multiple outgoing arcs. Instead of branching on arc usage,
    we partition the outgoing arcs into two sets A(d, a1) and A(d, a2) and
    create two children:
        - Child 1: Forbid commodity k from using arcs in A(d, a1)
        - Child 2: Forbid commodity k from using arcs in A(d, a2)

    Enforcement:
    ------------
    This is enforced in the pricing problem by setting infinite costs for
    forbidden arcs rather than explicitly fixing variables. The shortest
    path pricing problem will naturally avoid these arcs.

    For VRP (single commodity), we adapt this by:
    1. Identifying a node where multiple fractional routes "diverge"
    2. Partitioning the outgoing edges into two sets
    3. Creating child nodes that forbid routes from using specific edge sets

    References:
    -----------
    Barnhart, C., Hane, C. A., & Vance, P. H. (1998).
    "Using Branch-and-Price-and-Cut to Solve Origin-Destination Integer
    Multicommodity Flow Problems." Operations Research, 48(2), 318-326.
    Section 4: "Branching Strategy"
    """

    @staticmethod
    def find_divergence_node(
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-6,
    ) -> Optional[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """
        Find a divergence node and partition its outgoing arcs.

        A divergence node is one where multiple fractional routes leave
        the node via different arcs.

        Args:
            routes: All routes in the master problem.
            route_values: Current LP solution {route_index: λ_k}.
            tol: Integrality tolerance.

        Returns:
            Tuple of (divergence_node, arc_set_1, arc_set_2) or None.
            Each arc_set is a list of (from_node, to_node) tuples.
        """
        from collections import defaultdict

        # Build outgoing arc flow for each node
        node_outflow: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))

        for idx, lam in route_values.items():
            if abs(lam - round(lam)) <= tol:
                continue  # Skip integer routes

            route = routes[idx]
            full_path = [0] + route.nodes + [0]

            for i in range(len(full_path) - 1):
                u, v = full_path[i], full_path[i + 1]
                node_outflow[u][(u, v)] += lam

        # Find a node with fractional divergence
        for node, outgoing_arcs in node_outflow.items():
            if len(outgoing_arcs) < 2:
                continue  # No divergence

            # Check if flow is fractional
            total_flow = sum(outgoing_arcs.values())
            if abs(total_flow - round(total_flow)) <= tol:
                continue  # Integer flow at this node

            # Partition arcs into two sets
            # Strategy: sort by flow and split in half
            sorted_arcs = sorted(outgoing_arcs.items(), key=lambda x: x[1], reverse=True)
            mid = len(sorted_arcs) // 2

            arc_set_1 = [arc for arc, _ in sorted_arcs[:mid]]
            arc_set_2 = [arc for arc, _ in sorted_arcs[mid:]]

            if arc_set_1 and arc_set_2:
                return (node, arc_set_1, arc_set_2)

        return None

    @staticmethod
    def create_child_nodes(
        parent: BranchNode,
        divergence_node: int,
        arc_set_1: List[Tuple[int, int]],
        arc_set_2: List[Tuple[int, int]],
    ) -> Tuple[BranchNode, BranchNode]:
        """
        Create two child nodes that forbid different arc sets.

        Args:
            parent: The node being branched.
            divergence_node: The node where divergence occurs.
            arc_set_1: First set of outgoing arcs to forbid.
            arc_set_2: Second set of outgoing arcs to forbid.

        Returns:
            (left_child, right_child)
        """
        # Child 1: Forbid arcs in arc_set_1
        constraints_1 = [EdgeBranchingConstraint(u, v, must_use=False) for u, v in arc_set_1]

        # Child 2: Forbid arcs in arc_set_2
        constraints_2 = [EdgeBranchingConstraint(u, v, must_use=False) for u, v in arc_set_2]

        left = BranchNode(
            constraints=constraints_1,  # type: ignore[arg-type]
            parent=parent,
            depth=parent.depth + 1,
        )
        right = BranchNode(
            constraints=constraints_2,  # type: ignore[arg-type]
            parent=parent,
            depth=parent.depth + 1,
        )

        return left, right


class RyanFosterBranching:
    """
    Ryan-Foster branching: select a fractional node-pair co-occurrence.

    Produces two child nodes:
        left  → r and s MUST be in the same route  (together = True)
        right → r and s MUST NOT be in the same route (together = False)

    **WARNING:** Ryan-Foster branching loses its theoretical exactness
    guarantee when applied to a Set Covering master problem (>= 1), as it
    can erroneously prune optimal over-covering solutions. Use 'edge'
    branching for rigorous proofs of optimality.

    Reference: Ryan & Foster (1981), Proposition 1.
    """

    @staticmethod
    def find_branching_pair(
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-6,
    ) -> Optional[Tuple[int, int]]:
        """
        Find a node pair (r, s) whose fractional co-occurrence lies in (0, 1).

        Args:
            routes: All routes in the master problem.
            route_values: Current LP solution {route_index: λ_k}.
            tol: Integrality tolerance.

        Returns:
            (node_r, node_s) to branch on, or None if the solution is integer.
        """
        # Find any fractional route variable.
        frac_idx: Optional[int] = None
        for idx, val in route_values.items():
            if abs(val - round(val)) > tol:
                frac_idx = idx
                break

        if frac_idx is None:
            return None

        nodes_in_frac = sorted(routes[frac_idx].node_coverage)
        if len(nodes_in_frac) < 2:
            return None

        # Search all pairs for a fractional co-occurrence sum.
        for i, r in enumerate(nodes_in_frac):
            for s in nodes_in_frac[i + 1 :]:
                together_sum = sum(
                    val
                    for idx, val in route_values.items()
                    if r in routes[idx].node_coverage and s in routes[idx].node_coverage
                )
                frac = together_sum % 1.0
                if tol < frac < 1.0 - tol:
                    return (r, s)

        return None

    @staticmethod
    def create_child_nodes(
        parent: BranchNode,
        node_r: int,
        node_s: int,
    ) -> Tuple[BranchNode, BranchNode]:
        """
        Create together (left) and separate (right) child nodes.

        Args:
            parent: The node being branched.
            node_r: First node in the pair.
            node_s: Second node in the pair.

        Returns:
            (left_child, right_child)
        """
        left = BranchNode(
            constraints=[RyanFosterBranchingConstraint(node_r, node_s, together=True)],
            parent=parent,
            depth=parent.depth + 1,
        )
        right = BranchNode(
            constraints=[RyanFosterBranchingConstraint(node_r, node_s, together=False)],
            parent=parent,
            depth=parent.depth + 1,
        )
        return left, right


# ---------------------------------------------------------------------------
# Branch-and-Bound tree
# ---------------------------------------------------------------------------


class BranchAndBoundTree:
    """
    Manages the branch-and-bound tree with pluggable branching strategy.

    Supports two strategies via the ``strategy`` constructor parameter:

    ``"edge"``
        Branches on the most-fractional directed arc using
        :class:`EdgeBranching`.  Integrates natively with the DP pricing
        subproblem.

    ``"ryan_foster"``
        Branches on a fractional node-pair co-occurrence using
        :class:`RyanFosterBranching`.  Compatible with set-partitioning
        master problems.

    The tree uses best-first search (highest LP bound) to minimise the number
    of nodes explored before proving optimality.
    """

    def __init__(self, strategy: str = "edge", search_strategy: str = "best_first") -> None:
        """
        Initialise the B&B tree.

        Args:
            strategy: Branching strategy — ``"edge"`` (default),
                ``"ryan_foster"``, or ``"divergence"``.
            search_strategy: Node selection strategy — ``"best_first"`` (default)
                or ``"depth_first"``.

        Raises:
            ValueError: If an unsupported strategy string is provided.
        """
        if strategy not in ("edge", "ryan_foster", "divergence"):
            raise ValueError(
                f"Unsupported branching strategy '{strategy}'. Choose 'edge', 'ryan_foster', or 'divergence'."
            )
        if search_strategy not in ("best_first", "depth_first"):
            raise ValueError(f"Unsupported search strategy '{search_strategy}'. Choose 'best_first' or 'depth_first'.")

        self.strategy: str = strategy
        self.search_strategy: str = search_strategy
        self.root: BranchNode = BranchNode()
        self.open_nodes: List[BranchNode] = [self.root]
        self.best_integer_solution: Optional[float] = None
        self.best_integer_node: Optional[BranchNode] = None
        self.nodes_explored: int = 0
        self.nodes_pruned: int = 0

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def get_next_node(self) -> Optional[BranchNode]:
        """
        Pop and return the next open node based on the search strategy.

        - "best_first": Picks node with highest LP bound.
        - "depth_first": Picks latest added node (LIFO).

        Returns:
            Next node to process, or None if the frontier is empty.
        """
        if not self.open_nodes:
            return None

        if self.search_strategy == "best_first":
            self.open_nodes.sort(
                key=lambda n: n.lp_bound if n.lp_bound is not None else float("-inf"),
                reverse=True,
            )
            return self.open_nodes.pop(0)
        else:
            # depth_first (LIFO)
            return self.open_nodes.pop()

    def add_node(self, node: BranchNode) -> None:
        """Enqueue a new open node."""
        self.open_nodes.append(node)

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def branch(
        self,
        node: BranchNode,
        routes: List[Route],
        route_values: Dict[int, float],
    ) -> Optional[Tuple[BranchNode, BranchNode]]:
        """
        Apply the active branching strategy and return two child nodes.

        This is the single dispatch point for branching logic.  Callers no
        longer need to import strategy classes directly.

        Args:
            node: The fractional B&B node to branch from.
            routes: All routes in the master problem at this node.
            route_values: Current LP solution {route_index: λ_k}.

        Returns:
            ``(left_child, right_child)`` if a branching decision was found,
            or ``None`` if the solution is already integer (no fractional
            variable / arc found).
        """
        if self.strategy == "edge":
            arc = EdgeBranching.find_branching_arc(routes, route_values)
            if arc is None:
                return None
            u, v = arc
            return EdgeBranching.create_child_nodes(node, u, v)
        elif self.strategy == "multi_edge_partition":
            res = MultiEdgePartitionBranching.find_divergence_node(routes, route_values)
            if res is not None:
                div_node, arc_set_1, arc_set_2 = res
                return MultiEdgePartitionBranching.create_child_nodes(node, div_node, arc_set_1, arc_set_2)
            return None
        else:  # ryan_foster
            pair = RyanFosterBranching.find_branching_pair(routes, route_values)
            if pair is None:
                return None
            r, s = pair
            return RyanFosterBranching.create_child_nodes(node, r, s)

    # ------------------------------------------------------------------
    # Pruning and incumbent management
    # ------------------------------------------------------------------

    def prune_by_bound(self) -> int:
        """
        Remove nodes whose LP bound cannot improve the current incumbent.

        Returns:
            Number of nodes pruned.
        """
        if self.best_integer_solution is None:
            return 0
        before = len(self.open_nodes)
        self.open_nodes = [n for n in self.open_nodes if n.lp_bound is None or n.lp_bound > self.best_integer_solution]
        pruned = before - len(self.open_nodes)
        self.nodes_pruned += pruned
        return pruned

    def update_incumbent(self, node: BranchNode, value: float) -> bool:
        """
        Update the best known integer solution if *value* improves it.

        Args:
            node: Node where the integer solution was found.
            value: Objective value of the integer solution.

        Returns:
            True if the incumbent was improved.
        """
        if self.best_integer_solution is None or value > self.best_integer_solution:
            self.best_integer_solution = value
            self.best_integer_node = node
            return True
        return False

    def is_empty(self) -> bool:
        """Return True when no open nodes remain."""
        return len(self.open_nodes) == 0

    def get_statistics(self) -> Dict[str, Any]:
        """Return a snapshot of tree-search statistics."""
        return {
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
            "nodes_remaining": len(self.open_nodes),
            "best_bound": self.open_nodes[0].lp_bound if self.open_nodes else None,
            "best_integer": self.best_integer_solution,
            "branching_strategy": self.strategy,
        }
