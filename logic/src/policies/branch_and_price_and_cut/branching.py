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

FleetSizeBranchingConstraint
    Operates on the total number of vehicles used (sum of lambda variables).
    Enforces a floor or ceiling on the total fleet size.

NodeVisitationBranchingConstraint
    Operates on a single node i. Enforces v_i = 0 (forbidden) or v_i = 1 (forced).

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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from .master_problem import Route
    from .vrpp_model import VRPPModel

# Forward reference resolved at runtime — avoids a circular import with
# master_problem.py while still enabling full type annotations.
from .master_problem import Route
from .params import BPCParams

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

    def __init__(
        self,
        node_r: int,
        node_s: int,
        together: bool,
        mandatory_nodes: Optional[Set[int]] = None,
    ) -> None:
        """
        Initialise a Ryan-Foster branching constraint.

        Args:
            node_r: First node in the pair.
            node_s: Second node in the pair.
            together: Whether the two nodes must be co-visited (True) or
                separated (False).
            mandatory_nodes: Optional set of mandatory nodes to restrict over-pruning.
        """
        self.node_r = node_r
        self.node_s = node_s
        self.together = together
        # Store mandatory set so feasibility check can skip optional-only routes.
        # This prevents 'together' branching from over-pruning routes that only
        # visit an optional node without its partner.
        self._mandatory: Set[int] = mandatory_nodes or {node_r, node_s}

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
            # Only enforce co-occurrence when the route touches a mandatory node
            # from the pair; visiting an optional node alone is always valid.
            r_mandatory = self.node_r in self._mandatory
            s_mandatory = self.node_s in self._mandatory

            if (r_mandatory or s_mandatory) and (r_in != s_in):
                return False
        else:
            # The two nodes must never appear in the same route.
            if r_in and s_in:
                return False

        return True

    def __repr__(self) -> str:
        relation = "TOGETHER" if self.together else "SEPARATE"
        return f"RyanFosterBranchingConstraint({self.node_r}, {self.node_s}: {relation})"


# Backward-compatibility alias
BranchingConstraint = EdgeBranchingConstraint


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
    ) -> Optional[Tuple[Tuple[int, int], float]]:
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
        best_arc_flow = 0.0

        for arc, flow in arc_flow.items():
            frac = min(flow, 1.0 - flow)
            if frac > tol and frac > best_frac:
                best_frac = frac
                best_arc = arc
                best_arc_flow = flow

        if best_arc is None:
            return None
        return best_arc, best_arc_flow

    @staticmethod
    def create_child_nodes(
        parent: BranchNode,
        u: int,
        v: int,
        arc_flow: float = 0.5,
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
        hint = parent.lp_bound if parent.lp_bound is not None else 0.0
        # In maximization, the forbidden branch (right) is expected to have a lower
        # bound estimate than the parent. We subtract a small tie-breaker.
        right_hint = hint - (arc_flow * 1e-4)

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
            lp_bound_hint=right_hint,
        )
        return left, right


class MultiEdgePartitionBranching:
    r"""
    Advanced Divergence Branching with Spatial (Polar-Angle) Partitioning.

    This strategy formalizes the Divergence Branching of Barnhart et al. (1998)
    by utilizing node coordinates to induce a spatially cohesive arc-set
    partition.

    Mathematical Formulation:
    -------------------------
    1. Identify a 'divergence node' $d$ where fractional flow $\bar{x}$ splits
       into multiple outgoing arcs $(d, v_j)$.
    2. Define a spatial mapping function $f(v) = \text{atan2}(y_v - y_d, x_v - x_d)$
       which returns the polar angle of node $v$ relative to $d$.
    3. Sort the set of active outgoing arcs $A_d = \{(d, v) \in E : \bar{x}_{dv} > 0\}$
       by their destination nodes' polar angles.
    4. Partition $A_d$ into two subsets $A_1$ and $A_2$ using a median split
       on the sorted angles.
    5. Generate two child nodes by imposing constraints:
       - Left Child: $\sum_{(d, v) \in A_1} x_{dv} \le 0$ (forbidding set A1)
       - Right Child: $\sum_{(d, v) \in A_2} x_{dv} \le 0$ (forbidding set A2)

    Theoretical Rationale:
    ----------------------
    Unlike arbitrary arc splitting, spatial partitioning separates the routing
    topology into geographic sectors. This is polyhedrally significant for
    VRP variants as it tends to isolate independent sub-problems, leading to
    more balanced and computationally efficient Branch-and-Bound trees.
    """

    @staticmethod
    def find_divergence_node(  # noqa: C901
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-5,
        node_coords: Optional[Union[np.ndarray, Dict[int, Tuple[float, float]]]] = None,
    ) -> Optional[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
        """
        Identify a divergence node and compute the spatial partition.
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

        # reaching completeness guarantee.
        # Build the node universe from all routes in the pool, then treat every
        # customer that is not d as a valid potential successor.
        node_universe: Set[int] = {0}  # Task 2: Explicitly include the depot
        for route in routes:
            node_universe.update(route.nodes)
        all_possible_outgoing: Set[Tuple[int, int]] = {(d, v) for v in node_universe if v != d}
        # Always include the two diverging arcs themselves
        all_possible_outgoing.add(a1)
        all_possible_outgoing.add(a2)

        arc_set_1: Set[Tuple[int, int]] = {a1}
        arc_set_2: Set[Tuple[int, int]] = {a2}

        # 6. Partition remaining outgoing arcs based on polar angle relative to d
        remaining_arcs = sorted(list(all_possible_outgoing - {a1, a2}))
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

        # 6. Check if both halves of the partition are non-empty
        if not arc_set_1 or not arc_set_2:
            return None

        # 7. Return Tuple of (node, arc_set_1, arc_set_2, strength)
        # Strength is the total flow of arcs in set 1 (the 'left' partition)
        arc_flow = EdgeBranching.compute_arc_flow(routes, route_values)
        strength = sum(arc_flow.get(a, 0.0) for a in arc_set_1)
        return (d, list(arc_set_1), list(arc_set_2), strength)

    @staticmethod
    def find_multiple_divergence_nodes(
        routes: List[Route],
        route_values: Dict[int, float],
        node_coords: Optional[Union[np.ndarray, Dict[int, Tuple[float, float]]]] = None,
        limit: int = 5,
        tol: float = 1e-5,
    ) -> List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
        """
        Identify top branching candidates (divergence nodes) for lookahead eval.
        """
        res = MultiEdgePartitionBranching.find_divergence_node(routes, route_values, tol, node_coords)
        if res:
            return [res]
        return []

    @staticmethod
    def create_child_nodes(
        parent: BranchNode,
        divergence_node: int,
        arc_set_1: List[Tuple[int, int]],
        arc_set_2: List[Tuple[int, int]],
        strength: float = 0.5,
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

        hint = parent.lp_bound if parent.lp_bound is not None else 0.0
        # Child 1 is standard, child 2 gets a small penalty for sorting
        right_hint = hint - (strength * 1e-4)

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
            lp_bound_hint=right_hint,
        )

        return left, right


class RyanFosterBranching:
    """
    Ryan-Foster branching: select a fractional node-pair co-occurrence.

    Produces two child nodes:
        left  → r and s MUST be in the same route  (together = True)
        right → r and s MUST NOT be in the same route (together = False)

    Ryan-Foster branching (1981) for the Set Partitioning Problem.

    Theoretical Exactness:
    ----------------------
    Ryan-Foster branching loses its theoretical exactness guarantee when applied
    to a Set Covering master problem (>= 1), as it can erroneously prune optimal
    over-covering solutions. Total theoretical rigor is maintained here by
    enforcing strict Set Partitioning (== 1.0) whenever this strategy is active.

    Specifically, the `together=True` branch disables any route that visits
    only one of the pair (r, s). In a Set Partitioning (== 1) master, this
    is always safe because each node is covered by exactly one route.

    To ensure exactness, the `VRPPMasterProblem` automatically enforces strict
    Set Partitioning (== 1) logic for mandatory nodes. This removes the risk of
    pruning optimal solutions that can occur with Set Covering (>= 1).

    Reference: Ryan & Foster (1981), Proposition 1.
    """

    @staticmethod
    def find_branching_pair(
        routes: List[Route],
        route_values: Dict[int, float],
        mandatory_nodes: Set[int],
        tol: float = 1e-5,
    ) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Find a node pair (r, s) whose fractional co-occurrence lies in (0, 1).
        Only considers pairs where both nodes are mandatory, fulfilling the
        strict Set Partitioning requirement for exactness.

        Args:
            routes: All routes in the master problem.
            route_values: Current LP solution {route_index: λ_k}.
            mandatory_nodes: Set of mandatory nodes needing exact partitioning.
            tol: Integrality tolerance.

        Returns:
            ((node_r, node_s), together_sum) to branch on, or None if the solution is already integer.
        """
        for frac_idx, val in route_values.items():
            if abs(val - round(val)) > tol:
                nodes_in_frac = sorted([n for n in routes[frac_idx].node_coverage if n in mandatory_nodes])
                if len(nodes_in_frac) < 2:
                    continue

                # Search all pairs for a fractional co-occurrence sum.
                for i, r in enumerate(nodes_in_frac):
                    for s in nodes_in_frac[i + 1 :]:
                        together_sum = sum(
                            v
                            for idx, v in route_values.items()
                            if r in routes[idx].node_coverage and s in routes[idx].node_coverage
                        )
                        frac = abs(together_sum - round(together_sum))
                        if tol < frac < 1.0 - tol:
                            return (r, s), together_sum

        return None

    @staticmethod
    def create_child_nodes(
        parent: BranchNode,
        node_r: int,
        node_s: int,
        together_sum: float = 0.5,
        mandatory_nodes: Optional[Set[int]] = None,
    ) -> Tuple[BranchNode, BranchNode]:
        """
        Create together (left) and separate (right) child nodes.

        Args:
            parent: The node being branched.
            node_r: First node in the pair.
            node_s: Second node in the pair.
            together_sum: Fractional co-occurrence value for tie-breaking.
            mandatory_nodes: Optional set of mandatory nodes for constraint exactness.

        Returns:
            (left_child, right_child)
        """
        hint = parent.lp_bound if parent.lp_bound is not None else 0.0
        # Separation branch gets small penalty
        right_hint = hint - (together_sum * 1e-4)

        left = BranchNode(
            constraints=[RyanFosterBranchingConstraint(node_r, node_s, together=True, mandatory_nodes=mandatory_nodes)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
        )
        right = BranchNode(
            constraints=[
                RyanFosterBranchingConstraint(node_r, node_s, together=False, mandatory_nodes=mandatory_nodes)
            ],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=right_hint,
        )

        return left, right


class FleetSizeBranchingConstraint:
    """Constraint on the total number of vehicles used (sum of route lambdas)."""

    def __init__(self, limit: int, is_upper: bool) -> None:
        self.limit = limit
        self.is_upper = is_upper

    def is_route_feasible(self, route: Route) -> bool:
        """Route-level feasibility is 1.0 (global constraint)."""
        return True

    def __repr__(self) -> str:
        rel = "<=" if self.is_upper else ">="
        return f"FleetSizeBranchingConstraint(Sum λ {rel} {self.limit})"


class NodeVisitationBranchingConstraint:
    """Constraint on the visitation frequency of a specific optional node."""

    def __init__(self, node: int, forced: bool) -> None:
        self.node = node
        self.forced = forced

    def is_route_feasible(self, route: Route) -> bool:
        """
        In the 'forced' branch (v_i = 1), we don't filter existing routes
        that don't visit i; the master model will enforce the visitation sum.
        In the 'forbidden' branch (v_i = 0), we filter out any route visiting i.
        """
        if not self.forced:
            return self.node not in route.node_coverage
        return True

    def __repr__(self) -> str:
        val = 1 if self.forced else 0
        return f"NodeVisitationBranchingConstraint(v_{self.node} = {val})"


class FleetSizeBranching:
    """Logic for branching on the total fleet size."""

    @staticmethod
    def find_fleet_branching(
        route_values: Dict[int, float],
        tol: float = 1e-4,
    ) -> Optional[float]:
        """Check if sum of lambdas is fractional."""
        fleet_usage = sum(route_values.values())
        if abs(fleet_usage - round(fleet_usage)) > tol:
            return fleet_usage
        return None

    @staticmethod
    def create_child_nodes(parent: BranchNode, fleet_usage: float) -> Tuple[BranchNode, BranchNode]:
        """Create floor (lower branch) and ceiling (upper branch) child nodes."""
        floor = math.floor(fleet_usage + 1e-6)
        ceil = math.ceil(fleet_usage - 1e-6)
        lower_branch = BranchNode(
            constraints=[FleetSizeBranchingConstraint(floor, is_upper=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        upper_branch = BranchNode(
            constraints=[FleetSizeBranchingConstraint(ceil, is_upper=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        return lower_branch, upper_branch


class NodeVisitationBranching:
    """Logic for branching on optional node visitation."""

    @staticmethod
    def find_node_branching(
        routes: List[Route],
        route_values: Dict[int, float],
        optional_nodes: Set[int],
        tol: float = 1e-4,
    ) -> Optional[Tuple[int, float]]:
        """Find an optional node with fractional visitation."""
        for node in sorted(optional_nodes):
            visitation = sum(val for idx, val in route_values.items() if node in routes[idx].node_coverage)
            if tol < visitation < 1.0 - tol:
                return node, visitation
        return None

    @staticmethod
    def create_child_nodes(parent: BranchNode, node: int, visitation: float) -> Tuple[BranchNode, BranchNode]:
        """Create v_i = 0 and v_i = 1 child nodes."""
        left = BranchNode(
            constraints=[NodeVisitationBranchingConstraint(node, forced=False)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
        )
        right = BranchNode(
            constraints=[NodeVisitationBranchingConstraint(node, forced=True)],
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=parent.lp_bound,
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
        params: Optional[BPCParams] = None,
        # Legacy positional arguments kept for backward compatibility only.
        # If params is supplied these are ignored; a DeprecationWarning is emitted
        # when they differ from their defaults to surface accidental misuse.
        max_nodes: int = 1000,
        strategy: str = "edge",
        search_strategy: str = "best_first",
    ):
        """
        Initialize the Branch-and-Bound tree for BPC.

        Args:
            v_model: The underlying VRPP problem model.
            params: Standardized BPC parameters.
            max_nodes: Maximum number of nodes to explore.
            strategy: Branching strategy ('divergence_spatial', 'edge', 'ryan_foster').
            search_strategy: Search strategy ('best_first', 'depth_first').
        """
        import warnings

        if params is not None:
            if max_nodes != 1000 or strategy != "edge" or search_strategy != "best_first":
                warnings.warn(
                    "BranchAndBoundTree: explicit 'max_nodes', 'strategy', and "
                    "'search_strategy' arguments are ignored when 'params' is supplied. "
                    "Configure these via BPCParams instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            max_nodes = params.max_bb_nodes
            strategy = params.branching_strategy
            search_strategy = params.search_strategy

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
        """Delegate to the injected search strategy. Prefer calling
        search_strategy.select_node(bb_tree.open_nodes) directly."""
        raise NotImplementedError(
            "Call search_strategy.select_node(bb_tree.open_nodes) directly. "
            "get_next_node is retained only for backward compatibility."
        )

    # ------------------------------------------------------------------
    # Branching
    # ------------------------------------------------------------------

    def branch(
        self,
        node: BranchNode,
        routes: List[Route],
        route_values: Dict[int, float],
        mandatory_nodes: Optional[Set[int]] = None,
        strong_candidate: Optional[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]] = None,  # Task 11
    ) -> Optional[Tuple[BranchNode, BranchNode]]:
        """
        Apply the hierarchical branching strategy and return two child nodes.

        Task 11 (SOTA): Strong Branching.
        If a strong_candidate is provided (lookahead-verified winner), use it.
        Otherwise, follow the default hierarchy.
        """
        # --- Level 0: Strong Branching Winner (Task 11) ---
        if strong_candidate is not None:
            div_node, arc_set_1, arc_set_2, strength = strong_candidate
            return MultiEdgePartitionBranching.create_child_nodes(node, div_node, arc_set_1, arc_set_2, strength)

        # --- Level 1: Fleet Size ---
        fleet_frac = FleetSizeBranching.find_fleet_branching(route_values)
        if fleet_frac is not None:
            return FleetSizeBranching.create_child_nodes(node, fleet_frac)

        # --- Level 2: Spatial Divergence ---
        res_div = MultiEdgePartitionBranching.find_divergence_node(routes, route_values, node_coords=self.node_coords)
        if res_div is not None:
            div_node, arc_set_1, arc_set_2, strength = res_div
            return MultiEdgePartitionBranching.create_child_nodes(node, div_node, arc_set_1, arc_set_2, strength)

        # --- Level 3: Ryan-Foster co-occurrence ---
        if mandatory_nodes is not None:
            res_rf = RyanFosterBranching.find_branching_pair(routes, route_values, mandatory_nodes)
            if res_rf is not None:
                pair, together_sum = res_rf
                return RyanFosterBranching.create_child_nodes(
                    node, pair[0], pair[1], together_sum, mandatory_nodes=mandatory_nodes
                )

        # --- Level 4: Simple Edge Branching ---
        res_edge: Optional[Tuple[Tuple[int, int], float]] = EdgeBranching.find_branching_arc(routes, route_values)
        if res_edge is not None:
            arc, flow = res_edge
            return EdgeBranching.create_child_nodes(node, arc[0], arc[1], flow)

        # --- Level 5: Node Visitation (Last Resort - Mandatory only) ---
        # Fix 7: Restrict to mandatory nodes and move to last resort.
        if mandatory_nodes:
            node_frac_res = NodeVisitationBranching.find_node_branching(routes, route_values, mandatory_nodes)
            if node_frac_res is not None:
                n, visitation = node_frac_res
                return NodeVisitationBranching.create_child_nodes(node, n, visitation)

        return None

    def find_strong_branching_candidates(
        self, routes: List[Route], route_values: Dict[int, float], max_candidates: int = 5
    ) -> List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
        """
        Task 11 (SOTA): Identify top branching candidates for lookahead eval.
        Uses Spatial Divergence strength as the primary heuristic.
        """
        # We target divergence nodes as they provide the most robust branching
        candidates = MultiEdgePartitionBranching.find_multiple_divergence_nodes(
            routes, route_values, node_coords=self.node_coords, limit=max_candidates
        )
        return candidates

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
            and (n.lp_bound is None or n.lp_bound > self.best_integer_solution + 1e-8)
        ]
        pruned = before - len(self.open_nodes)
        self.nodes_pruned += pruned
        return pruned

    def record_explored(self) -> None:
        """Increment the global nodes explored counter."""
        self.nodes_explored += 1

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


# Union type for callers that handle any constraint flavour.
AnyBranchingConstraint = Union[
    EdgeBranchingConstraint,
    RyanFosterBranchingConstraint,
    NodeVisitationBranchingConstraint,
    FleetSizeBranchingConstraint,
]
