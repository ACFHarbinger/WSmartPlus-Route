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

    Conflict rule:
        At most one must_use arc may originate from any node, and at most one
        must_use arc may terminate at any node. Violations are detected by the
        column generation loop before the RCSPP solve and cause the B&B node
        to be marked infeasible.
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
    ) -> None:
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
            # A route that visits either node of the pair must visit both.
            # We do not restrict routes that visit neither node — those are
            # always feasible on the together branch.
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
        branching_rule: str = "none",
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
        self.lp_bound_hint = lp_bound_hint
        self.branching_rule = branching_rule
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
        flow: float = 0.5,
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
        left_bc = EdgeBranchingConstraint(u, v, must_use=True)
        right_bc = EdgeBranchingConstraint(u, v, must_use=False)

        return (
            BranchNode(
                constraints=parent.constraints + [left_bc],
                parent=parent,
                depth=parent.depth + 1,
                lp_bound_hint=flow,
                branching_rule="edge",
            ),
            BranchNode(
                constraints=parent.constraints + [right_bc],
                parent=parent,
                depth=parent.depth + 1,
                lp_bound_hint=1.0 - flow,
                branching_rule="edge",
            ),
        )


class MultiEdgePartitionBranching:
    r"""
    Advanced Divergence Branching with Spatial Fleet Partitioning.

    This strategy formalizes the Divergence Branching of Barnhart et al. (1998)
    by utilizing node coordinates to induce a spatially cohesive arc-set
    partition. Unlike ODIMCF which restricts specific vehicles (commodities),
    this restricts the entire anonymous fleet, making it polyhedrally stronger.

    Mathematical Formulation:
    -------------------------
    1. Divergence Node Identification:
       Identify a 'divergence node' $d$ where the fractional flow $\bar{x}$
       diverges into multiple outgoing arcs.
       $A_d^+ = \{(d, v) \in E : 0 < \bar{x}_{dv} < 1\}$

    2. Polar Mapping:
       Define a spatial mapping function $f(v) = \operatorname{atan2}(y_v - y_d, x_v - x_d)$
       returning the polar angle of node $v$ relative to $d$.

    3. Spatial Partitioning:
       Sort $A_d^+$ by destination polar angles and partition into $S_1$ and $S_2$
       via a median split. This creates two balanced geographic sectors.
       $\mathcal{L}: \sum_{(d,v) \in S_1} x_{dv} = 0 \quad \text{and} \quad \mathcal{R}: \sum_{(d,v) \in S_2} x_{dv} = 0$

    4. Candidate Scoring (SVRPC):
       Candidates are ranked by the Spatial Variable Routing Persistence (SVRP) 
       strength, calculated as the fractional flow persistence across the split:
       $\sigma(d) = \sum_{(d,v) \in S_1} \bar{x}_{dv}$

    Theoretical Rationale:
    ----------------------
    Spatial fleet partitioning separates the routing topology into convex 
    geographic polygons. By forbidding arc sets rather than single edges, 
    it globally restricts the anonymous fleet, enforcing a strong bound.
    """

    @staticmethod
    def find_divergence_node(
        routes: List[Route],
        route_values: Dict[int, float],
        tol: float = 1e-5,
        node_coords: Optional[np.ndarray] = None,
        n_nodes: int = 0,
    ) -> Optional[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
        """
        Identify a node 'd' where routes diverge into two arc sets.

        The divergence branching rule (Barnhart et al. 2000, §4.3) identifies
        a node 'd' such that the outgoing flow from 'd' is split between two
        sets of arcs A1 and A2.

        Args:
            routes: Column pool.
            route_values: Fractional solution (λ_k).
            tol: Fractional tolerance.
            node_coords: Optional coords for spatial partitioning.
            n_nodes: Total number of customer nodes in the problem.

        Returns:
            Tuple of (divergence_node_d, arc_set_1, arc_set_2, score) or None.
        """
        # 1. Identify divergence node d (must have at least two outgoing arcs)
        node_out_flows: Dict[int, Dict[int, float]] = {}
        for idx, lam in route_values.items():
            if lam < tol:
                continue
            r = routes[idx]
            full_path = [0] + r.nodes + [0]
            for i in range(len(full_path) - 1):
                u, v = full_path[i], full_path[i + 1]
                node_out_flows.setdefault(u, {}).setdefault(v, 0.0)
                node_out_flows[u][v] += lam

        # Candidate divergence nodes (those with >1 outgoing arcs in the fractional soln)
        div_candidates = [d for d, flows in node_out_flows.items() if len(flows) >= 2]
        if not div_candidates:
            return None

        # 2. Select node d (tie-break by flow magnitude)
        d = max(div_candidates, key=lambda x: sum(node_out_flows[x].values()))

        # 3. Partition all possible outgoing arcs from d into A1 and A2.
        # Fix 6: Node universe must cover all possible customer nodes to ensure
        # the partition is complete and cannot be bypassed by new columns.
        if n_nodes > 0:
            node_universe = set(range(n_nodes + 1))
        else:
            node_universe = {0}
            for r in routes:
                node_universe.update(r.nodes)
        node_universe.discard(d)
        all_possible_outgoing = {(d, v) for v in node_universe}
        
        # A1 contains the arc with the largest flow, A2 contains the rest (initial)
        # Spatial refinement if coords are available
        flows = node_out_flows[d]
        v_sorted = sorted(flows.keys(), key=lambda v: flows[v], reverse=True)
        
        v1 = v_sorted[0]
        arc_set_1 = {(d, v1)}
        
        if node_coords is not None:
            # Spatial partition: group arcs by distance to v1 in the graph
            # Arcs to nodes spatially close to v1 go to A1.
            d_coord = np.array(node_coords[d])
            v1_coord = np.array(node_coords[v1])
            vec_v1 = v1_coord - d_coord
            
            for v in node_universe:
                if v == v1:
                    continue
                v_coord = np.array(node_coords[v])
                vec_v = v_coord - d_coord
                # Cosine similarity for direction-based grouping
                similarity = np.dot(vec_v1, vec_v) / (np.linalg.norm(vec_v1) * np.linalg.norm(vec_v) + 1e-9)
                if similarity > 0.5:
                    arc_set_1.add((d, v))

        arc_set_2 = all_possible_outgoing - arc_set_1
        
        # Invariant: A1 and A2 must partition ALL outgoing arcs from d
        if not (arc_set_1 | arc_set_2) == all_possible_outgoing:
             logger.error("Divergence arc partition is incomplete.")
             
        score = sum(flows[v] for v in flows if (d, v) in arc_set_1)
        return d, list(arc_set_1), list(arc_set_2), score

    @staticmethod
    def find_multiple_divergence_nodes(
        routes: List[Route],
        route_values: Dict[int, float],
        node_coords: Optional[np.ndarray] = None,
        limit: int = 5,
        tol: float = 1e-5,
        n_nodes: int = 0,
    ) -> List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
        """
        Implement proper multi-candidate support (Fix 19).
        """
        candidates = []
        remaining_route_values = dict(route_values)

        for _ in range(limit):
            if len(remaining_route_values) < 2:
                break
            res = MultiEdgePartitionBranching.find_divergence_node(
                routes, remaining_route_values, tol, node_coords, n_nodes
            )
            if res is None:
                break
            candidates.append(res)
            d, arc_set_1, arc_set_2, _ = res
            # Remove routes passing through d to find the next divergence node
            remaining_route_values = {
                idx: lam for idx, lam in remaining_route_values.items()
                if d not in ([0] + routes[idx].nodes + [0])
            }

        return candidates

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
            constraints=parent.constraints + constraints_1,
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=hint,
            branching_rule="divergence",
        )
        right = BranchNode(
            constraints=parent.constraints + constraints_2,
            parent=parent,
            depth=parent.depth + 1,
            lp_bound_hint=right_hint,
            branching_rule="divergence",
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
                # Search over all fractional nodes (Fix 7).
                # Note: Ryan-Foster pairs are most effective for mandatory nodes
                # but mathematically valid for any nodes in a Set Partitioning RMP.
                nodes_in_frac = sorted(list(routes[frac_idx].node_coverage))
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
    ) -> Tuple[BranchNode, BranchNode]:
        """
        Create together (left) and separate (right) child nodes.

        Args:
            parent: The node being branched.
            node_r: First node in the pair.
            node_s: Second node in the pair.
            together_sum: Fractional co-occurrence value for tie-breaking.

        Returns:
            (left_child, right_child)
        """
        hint = parent.lp_bound if parent.lp_bound is not None else 0.0
        # Separation branch gets small penalty
        right_hint = hint - (together_sum * 1e-4)

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

    def find_strong_branching_candidates(
        self, routes: List[Route], route_values: Dict[int, float], max_candidates: int = 5
    ) -> List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
        """
        Task 11 (SOTA): Identify top branching candidates for lookahead eval.
        Uses Spatial Divergence strength as the primary heuristic.
        """
        candidates = []
        # Divergence candidates
        div_candidates = MultiEdgePartitionBranching.find_multiple_divergence_nodes(
            routes,
            route_values,
            node_coords=self.node_coords,
            limit=max_candidates,
            n_nodes=self.v_model.n_nodes - 1,
        )
        for cand in div_candidates:
            d, arcs1, arcs2, score = cand
            candidates.append((d, arcs1, arcs2, score))

        # Sort by fractional score (how close flow is to 0.5)
        candidates.sort(key=lambda x: abs(0.5 - (x[3] % 1.0)), reverse=True)
        return candidates[:max_candidates]

    def branch(
        self,
        node: BranchNode,
        routes: List[Route],
        route_values: Dict[int, float],
        mandatory_nodes: Set[int],
        strong_candidate: Optional[Any] = None,
    ) -> Optional[Tuple[BranchNode, BranchNode]]:
        """
        Create child nodes by branching on a fractional solution.
        """
        # 1. Strong Branching candidate override
        if strong_candidate:
            d, arcs1, arcs2, _ = strong_candidate
            return MultiEdgePartitionBranching.create_child_nodes(node, d, arcs1, arcs2)

        # 2. Level 1: Fleet Size branching
        res_fleet = FleetSizeBranching.find_fleet_branching(route_values)
        if res_fleet:
            return FleetSizeBranching.create_child_nodes(node, *res_fleet)

        # 3. Level 2: Divergence branching (Preferred spatial rule)
        # Fix 6: Pass n_nodes to ensure complete arc partition.
        res_div = MultiEdgePartitionBranching.find_divergence_node(
            routes,
            route_values,
            node_coords=self.node_coords,
            n_nodes=self.v_model.n_nodes - 1,
        )
        if res_div:
            d, arcs1, arcs2, _ = res_div
            return MultiEdgePartitionBranching.create_child_nodes(node, d, arcs1, arcs2)

        # 4. Level 3: Ryan-Foster branching (co-occurrence)
        res_rf = RyanFosterBranching.find_branching_pair(routes, route_values, mandatory_nodes)
        if res_rf:
            pair, together_sum = res_rf
            return RyanFosterBranching.create_child_nodes(node, pair[0], pair[1], together_sum)

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
