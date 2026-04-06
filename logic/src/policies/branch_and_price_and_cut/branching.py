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

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from .master_problem import Route
    from .vrpp_model import VRPPModel

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
        lp_bound_hint: Optional[float] = None,
    ) -> None:
        """
        Initialise a branch node.

        Args:
            constraints: Constraints added *at this node only* (not inherited).
            parent: Parent node (None for root).
            depth: Tree depth (root = 0).
            lp_bound_hint: Optional initial LP bound (inherited from parent).
        """
        self.constraints: List[AnyBranchingConstraint] = constraints or []
        self.parent: Optional["BranchNode"] = parent
        self.depth: int = depth
        self.lp_bound: Optional[float] = lp_bound_hint
        self.ip_solution: Optional[float] = None
        self.is_integer: bool = False
        self.is_infeasible: bool = False
        self.route_values: Optional[Dict[int, float]] = None
        self.routes: Optional[List[Route]] = None
        self.lp_basis: Optional[Any] = None

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
        return constraints[::-1]

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
        tol: float = 1e-5,
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
        hint = parent.lp_bound
        left = BranchNode(
            constraints=[EdgeBranchingConstraint(u, v, must_use=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
        )
        right = BranchNode(
            constraints=[EdgeBranchingConstraint(u, v, must_use=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
        )
        return left, right


class MultiEdgePartitionBranching:
    """
    Advanced Divergence Branching with Spatial (Polar-Angle) Partitioning.

    This strategy extends the standard Divergence Branching (Barnhart et al. 1998)
    by using node coordinates to create spatially cohesive arc sets.

    Mechanism:
        1. Identify a 'divergence node' (d) where the fractional flow splits
           into multiple outgoing arcs.
        2. Sort ALL outgoing arcs from (d) by the polar angle of their
           destination nodes relative to (d).
        3. Partition the sorted arcs into two sets (A1, A2) using a median split.
        4. Create two child nodes:
           - Left: Must use an arc in A1 (if leaving d).
           - Right: Must use an arc in A2 (if leaving d).

    Theoretical Advantage:
        Spatial partitioning is highly effective for VRP variants because it
        tends to separate the problem into geographic sectors, leading to
        more balanced and deeper search trees compared to arbitrary splitting.
    """

    @staticmethod
    def find_divergence_node(  # noqa: C901
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-5,
        node_coords: Optional[Union[np.ndarray, Dict[int, Tuple[float, float]]]] = None,
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
        # 1. Collect fractional routes
        fractional_routes = [(idx, val) for idx, val in route_values.items() if tol < abs(val - round(val))]
        if len(fractional_routes) < 2:
            return None

        # 2. Sort fractional routes by lambda value descending and take the top two
        fractional_routes.sort(key=lambda x: x[1], reverse=True)
        idx_a, val_a = fractional_routes[0]
        idx_b, val_b = fractional_routes[1]

        # 3. Construct the full arc sequences for both routes (including depot 0)
        path_a = [0] + routes[idx_a].nodes + [0]
        path_b = [0] + routes[idx_b].nodes + [0]

        # 4. Walk both paths to find first divergence node d
        d = 0
        a1_v = path_a[1]
        a2_v = path_b[1]

        min_len = min(len(path_a), len(path_b))
        for i in range(min_len - 1):
            if path_a[i] == path_b[i]:
                if path_a[i + 1] != path_b[i + 1]:
                    d = path_a[i]
                    a1_v = path_a[i + 1]
                    a2_v = path_b[i + 1]
                    break
            else:
                # Paths differ at the very first step or earlier
                break

        a1 = (d, a1_v)
        a2 = (d, a2_v)

        # 5. Build arc sets by partitioning all outgoing arcs of d across all routes
        all_outgoing = set()
        for idx, lam in route_values.items():
            if lam > tol:
                nodes = [0] + routes[idx].nodes + [0]
                for i in range(len(nodes) - 1):
                    if nodes[i] == d:
                        all_outgoing.add((d, nodes[i + 1]))

        arc_set_1 = {a1}
        arc_set_2 = {a2}

        # 5. Partition remaining outgoing arcs based on polar angle relative to d
        remaining_arcs = sorted(list(all_outgoing - {a1, a2}))
        if node_coords is not None:
            # Type-agnostic check for coordinate presence
            has_d = (isinstance(node_coords, dict) and d in node_coords) or (
                isinstance(node_coords, np.ndarray) and d < len(node_coords)
            )

            if has_d:
                d_coord = node_coords[d]

                def get_polar_angle(v: int) -> float:
                    has_v = (isinstance(node_coords, dict) and v in node_coords) or (
                        isinstance(node_coords, np.ndarray) and v < len(node_coords)
                    )
                    if not has_v:
                        return 0.0
                    v_coord = node_coords[v]
                    return math.atan2(v_coord[1] - d_coord[1], v_coord[0] - d_coord[0])

                # Sort by polar angle
                remaining_arcs.sort(key=lambda a: get_polar_angle(a[1]))

                # Split at the median to form two sets
                mid = len(remaining_arcs) // 2
                arc_set_1.update(remaining_arcs[:mid])
                arc_set_2.update(remaining_arcs[mid:])
            else:
                # Fallback to naive alternating split if coordinates for d are missing
                for i, arc in enumerate(remaining_arcs):
                    if i % 2 == 0:
                        arc_set_1.add(arc)
                    else:
                        arc_set_2.add(arc)
        else:
            # Fallback to naive alternating split if coordinates are missing entirely
            for i, arc in enumerate(remaining_arcs):
                if i % 2 == 0:
                    arc_set_1.add(arc)
                else:
                    arc_set_2.add(arc)

        # 6. Safety check
        if not arc_set_1 or not arc_set_2:
            return None

        # 7. Return Tuple of (node, arc_set_1, arc_set_2)
        return (d, list(arc_set_1), list(arc_set_2))

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

        hint = parent.lp_bound
        left = BranchNode(
            constraints=constraints_1,  # type: ignore[arg-type]
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
        )
        right = BranchNode(
            constraints=constraints_2,  # type: ignore[arg-type]
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
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

    Specifically, the `together=True` branch disables any route that visits
    only one of the pair (r, s). In a Set Partitioning (== 1) master, this
    is always safe because each node is covered by exactly one route. In a
    Set Covering (>= 1) master — as used in VRPPMasterProblem — a route
    visiting only r is a valid column (s may be covered by another route),
    so disabling it can remove optimal columns and cause the solver to miss
    the true optimum.

    To use Ryan-Foster branching with full exactness guarantees, either:
      1. Switch VRPPMasterProblem to Set Partitioning (== 1) coverage
         constraints for mandatory nodes, OR
      2. Use EdgeBranching ('edge' strategy) instead, which does not have
         this limitation.

    Reference: Ryan & Foster (1981), Proposition 1.
    """

    @staticmethod
    def find_branching_pair(
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-5,
    ) -> Optional[Tuple[int, int]]:
        """
        Find a node pair (r, s) whose fractional co-occurrence lies in (0, 1).

        Args:
            routes: All routes in the master problem.
            route_values: Current LP solution {route_index: λ_k}.
            tol: Integrality tolerance.

        Returns:
            (node_r, node_s) to branch on, or None if the solution is already integer.
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
        hint = parent.lp_bound
        left = BranchNode(
            constraints=[RyanFosterBranchingConstraint(node_r, node_s, together=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
        )
        right = BranchNode(
            constraints=[RyanFosterBranchingConstraint(node_r, node_s, together=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
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

    Node selection strategy is injected externally via `NodeSelectionStrategy` objects
    (see `search_strategy.py`). `BranchAndBoundTree` does not perform node selection
    itself — call `strategy.select_node(bb_tree.open_nodes)` from the solver loop.
    """

    node_coords: Optional[np.ndarray]

    def __init__(
        self,
        v_model: VRPPModel,
        max_nodes: int = 1000,
        strategy: str = "edge",
        search_strategy: str = "best_first",
    ):
        """
        Initialize the Branch-and-Bound tree for BPC.

        Args:
            v_model: The underlying VRPP problem model.
            max_nodes: Maximum number of nodes to explore.
            strategy: Branching strategy ('divergence_spatial', 'edge', 'ryan_foster').
            search_strategy: Search strategy ('best_first', 'depth_first').
        """
        self.v_model = v_model

        # Extract coordinates from the injected model
        node_coords = v_model.node_coords

        # Convert Dict coords to Array if provided in that format
        if isinstance(node_coords, dict):
            coords_arr = np.zeros((len(node_coords) + 1, 2))
            for i, (x, y) in node_coords.items():
                coords_arr[i] = [x, y]
            self.node_coords = coords_arr  # type: ignore[assignment]
        else:
            self.node_coords = node_coords

        self.max_nodes = max_nodes
        self.strategy = strategy
        self.search_strategy = search_strategy

        # Root node
        self.root = BranchNode()
        self.open_nodes: List[BranchNode] = [self.root]
        self.best_integer_solution: Optional[float] = None
        self.best_integer_node: Optional[BranchNode] = None
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def add_node(self, node: BranchNode) -> None:
        """Enqueue a new open node."""
        self.open_nodes.append(node)

    def get_next_node(self) -> Optional[BranchNode]:
        """
        Select and remove the next node to process from the open list.

        Uses the configured search strategy:
        - "best_first": pop the node with the highest LP bound (best-bound-first).
        - "depth_first": pop the most recently added node (LIFO).

        Returns:
            The selected BranchNode, or None if the open list is empty.
        """
        if not self.open_nodes:
            return None

        if self.search_strategy == "depth_first":
            return self.open_nodes.pop()

        # Default: best-first — select the node with the highest LP bound.
        # Nodes with lp_bound=None (only the root) are treated as +inf
        # so they are explored first.
        best_idx = max(
            range(len(self.open_nodes)),
            key=lambda i: self.open_nodes[i].lp_bound if self.open_nodes[i].lp_bound is not None else float("inf"),  # type: ignore[arg-type,return-value]
        )
        return self.open_nodes.pop(best_idx)

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
        elif self.strategy in ("divergence", "multi_edge_partition"):
            # "divergence" is the documented public name; "multi_edge_partition" is the
            # internal implementation name. Both route to MultiEdgePartitionBranching.
            res = MultiEdgePartitionBranching.find_divergence_node(routes, route_values, node_coords=self.node_coords)
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
        self.open_nodes = [
            n
            for n in self.open_nodes
            if not n.is_infeasible  # exclude known infeasible
            and (n.lp_bound is None or n.lp_bound > self.best_integer_solution)
        ]
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
            "best_bound": (
                max(
                    (n.lp_bound for n in self.open_nodes if n.lp_bound is not None),
                    default=None,
                )
            ),
            "best_integer": self.best_integer_solution,
            "branching_strategy": self.strategy,
        }
