"""
Resource-Constrained Shortest Path Problem (RCSPP) solver using Dynamic Programming.

Implements label-setting algorithm for Elementary Shortest Path Problem with Resource
Constraints (ESPPRC) as described in:
- Irnich & Desaulniers (2005): "Shortest Path Problems with Resource Constraints"
- Feillet et al. (2004): "An exact algorithm for the ESPPRC"
- Baldacci, R., Mingozzi, A., & Roberti, R. (2011). "New Route Relaxation and Pricing
  Strategies for the Vehicle Routing Problem". Operations Research, 59(5), 1269-1283.

This is the exact / ng-relaxed pricing subproblem for Branch-and-Price column generation.

Edge-branching constraint enforcement
--------------------------------------
Branching constraints from ``branching.EdgeBranchingConstraint`` are enforced
*inside* the label-extension loop rather than as a post-hoc feasibility filter
on completed routes.  This is both more efficient (infeasible partial paths are
discarded early) and mathematically correct (a post-hoc filter can mistakenly
discard a dominating label before the constraint is checked, causing the
algorithm to miss the constrained optimum).

Three rules are applied before extending from node u to node v:

1. Forbidden-arc rule (must_use = False):
   If any constraint forbids arc (u → v), skip v entirely.

2. Required-successor rule (must_use = True, source = u):
   If node u has a required outgoing arc to some node w, then the only
   permitted extension from u is to w.  Any v ≠ w is skipped.
   Exception: the depot is always a legal return destination.

3. Required-predecessor rule (must_use = True, target = v):
   If node v has a required incoming arc from some node x, then the only
   node permitted to precede v is x.  Any u ≠ x is skipped.
   Exception: v may also be reached from the depot.

ng-Route Relaxation
--------------------
When ``use_ng_routes=True`` (default), the solver uses the *ng*-route
relaxation of Baldacci et al. (2011) instead of exact ESPPRC.

Exact ESPPRC tracks the complete visited set to prevent cycles, resulting
in an exponential number of labels in the worst case.  The *ng*-route
relaxation replaces this with a compact *ng*-memory set M_v, which only
blocks revisiting nodes in the neighborhood N_v of the current node v.
This makes dominance checks much more effective (smaller memory ⇒ more
labels dominate each other) while still preventing the shortest, most
profitable cycles.

Key definitions (Baldacci et al. 2011, Section 3):
    N_i    – ng-neighborhood of node i: the k closest nodes to i by distance,
              always including i itself.
    M_v    – ng-memory of a label at node v: the subset of visited nodes
              that belong to N_v.  Computed incrementally as:
                  M_v = (M_u ∪ {v}) ∩ N_v
              where u is the predecessor node.
    Feasibility: extension from a label at v to node w is allowed iff
              w ∉ M_v  (ng mode)  or  w ∉ visited  (exact ESPPRC).
    Dominance: label L1 dominates L2 at the same node iff:
              L1.rc ≥ L2.rc  AND  L1.load ≤ L2.load  AND
              L1.ng_memory ⊆ L2.ng_memory  (ng mode)  or
              L1.visited   ⊆ L2.visited    (exact ESPPRC).

The ``visited`` set is always maintained alongside ``ng_memory`` so that the
final path can be reconstructed accurately and exact ESPPRC can be restored
by a single flag toggle (``use_ng_routes=False``).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np


@dataclass(order=True)
class Label:
    """
    Label for dynamic programming state in RCSPP / ng-RCSPP.

    Represents a partial path from the depot to the current node with
    accumulated resources.  Labels are ordered by reduced cost for efficient
    dominance checking (higher is better for the maximisation objective).

    When ng-route relaxation is active, the dominance and feasibility checks
    use ``ng_memory`` instead of ``visited``.  The ``visited`` set is always
    maintained so that the final path sequence can be reconstructed correctly
    regardless of the relaxation mode, and to ensure that toggling
    ``use_ng_routes=False`` restores exact ESPPRC without any other changes.
    """

    # Primary sort key — higher reduced cost is preferred.
    reduced_cost: float = field(compare=True)

    # State fields — excluded from the dataclass ordering.
    node: int = field(compare=False)
    cost: float = field(compare=False)
    load: float = field(compare=False)
    revenue: float = field(compare=False)
    path: List[int] = field(default_factory=list, compare=False)

    # visited: complete set of customer nodes on the partial path.
    # Always maintained for path reconstruction and exact-ESPPRC mode.
    visited: Set[int] = field(default_factory=set, compare=False)

    # ng_memory: compact relaxed state for ng-route dominance / feasibility.
    ng_memory: Set[int] = field(default_factory=set, compare=False)

    # rf_unmatched: nodes from a 'together' pair visited without their partner.
    rf_unmatched: FrozenSet[int] = field(default_factory=frozenset, compare=False)

    parent: Optional["Label"] = field(default=None, compare=False, repr=False)

    def dominates(
        self,
        other: "Label",
        use_ng: bool = False,
        epsilon: float = 1e-6,
    ) -> bool:
        """
        Check whether this label dominates *other* at the same node.

        Dominance requires all four conditions to hold simultaneously:
            1. self.reduced_cost >= other.reduced_cost
            2. self.load         <= other.load
            3. self.rf_unmatched == other.rf_unmatched (Obligation Exactness)
            4a. [ng mode]   self.ng_memory ⊆ other.ng_memory
            4b. [exact mode] self.visited  ⊆ other.visited

        Condition 3a is the key difference from exact ESPPRC.  Because
        ``ng_memory`` is a *subset* of ``visited`` (only nearby nodes are
        retained), more pairs of labels satisfy the subset relation and more
        labels get pruned.  This is the efficiency gain of the ng-relaxation:
        the label count is dramatically reduced on large instances.

        Soundness: if L1 dominates L2 under ng-memory, every ng-feasible
        extension of L2 is also ng-feasible for L1, so discarding L2 is safe.
        Cycles that exact ESPPRC would prevent but ng-routes allow involve
        nodes outside N_v — those are typically far away and thus unprofitable
        after travel-cost subtraction.

        Args:
            other: Another label at the same node.
            use_ng: If True, use ng_memory for condition 3 (ng mode).
                    If False, use visited (exact ESPPRC mode).
            epsilon: Numerical tolerance for floating-point comparisons.

        Returns:
            True if this label dominates *other*.
        """
        if self.node != other.node:
            return False
        if self.reduced_cost < other.reduced_cost - epsilon:
            return False
        if self.load > other.load + epsilon:
            return False
        if self.rf_unmatched != other.rf_unmatched:
            return False

        if use_ng:
            return self.ng_memory.issubset(other.ng_memory)
        else:
            return self.visited.issubset(other.visited)

    def is_feasible(self, capacity: float) -> bool:
        """Return True if the accumulated load does not exceed capacity."""
        return self.load <= capacity

    def reconstruct_path(self) -> List[int]:
        """
        Reconstruct the complete path by following parent pointers.

        Returns:
            Ordered list of node indices from depot to current node.
        """
        if self.parent is None:
            return [self.node]
        return self.parent.reconstruct_path() + [self.node]


class RCSPPSolver:
    """
    Exact / ng-relaxed solver for the Resource-Constrained Shortest Path Problem.

    Uses forward label-setting dynamic programming with dominance pruning to
    find elementary (or ng-feasible) paths of maximum reduced cost, subject to
    vehicle capacity.

    Two operating modes are supported:

    Exact ESPPRC (``use_ng_routes=False``)
        Tracks the complete visited set in every label.  Guarantees that every
        generated column is a true elementary route.  Exponential worst-case
        label count for dense instances.

    ng-Route relaxation (``use_ng_routes=True``, default)
        Tracks only the compact ng-memory set M_v = (M_u ∪ {v}) ∩ N_v per
        Baldacci et al. (2011).  Permits revisiting nodes outside N_v, but
        such revisits are typically unprofitable.  Dominance is far more
        effective, drastically reducing the label count on large instances.
        Setting ``use_ng_routes=False`` perfectly restores exact ESPPRC.

    Both modes accept ``EdgeBranchingConstraint`` objects from the current B&B
    node and enforce them eagerly during label extension.

    References:
        Baldacci, R., Mingozzi, A., & Roberti, R. (2011). "New Route Relaxation
        and Pricing Strategies for the Vehicle Routing Problem". Operations
        Research, 59(5), 1269-1283.

    Attributes:
        n_nodes: Number of customer nodes (depot excluded; depot index = 0).
        cost_matrix: Distance matrix of shape (n_nodes+1, n_nodes+1).
        wastes: Mapping from customer node ID to waste volume.
        capacity: Vehicle payload capacity.
        R: Revenue per unit of waste.
        C: Cost per unit of distance.
        mandatory_nodes: Nodes that must appear in every feasible route.
        depot: Depot index (always 0).
        use_ng_routes: Whether ng-route relaxation is active.
        ng_neighborhood_size: Number of closest neighbors in each N_i.
        ng_neighborhoods: Precomputed neighborhood sets N_i for every node.
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
        use_ng_routes: bool = True,
        ng_neighborhood_size: int = 8,
    ) -> None:
        """
                Initialise the RCSPP solver.

                Args:
                    n_nodes: Number of customer nodes (excluding depot).
                    cost_matrix: Distance matrix (n_nodes+1 × n_nodes+1); index 0 is
                        the depot.
                    wastes: Mapping from node ID to waste volume.
                    capacity: Vehicle payload capacity.
                    revenue_per_kg: Revenue earned per unit of waste collected.
                    cost_per_km: Operating cost per unit of distance travelled.
                    mandatory_nodes: Set of customer node This pricer implements a **Label-Correcting** algorithm (FIFO queue) to handle
        potential negative edge costs introduced by dual variables. It supports both
        exact ESPPRC and the ng-route relaxation (Baldacci et al. 2011).
                        Set False to restore exact ESPPRC behaviour.  Default True.
                    ng_neighborhood_size: Size of each node's ng-neighborhood N_i
                        (including the node itself).  Larger values produce a tighter
                        relaxation (approaching exact ESPPRC) at the cost of slower
                        dominance checks.  Default 8.
        """
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes: Set[int] = mandatory_nodes or set()
        self.depot = 0
        self.use_ng_routes = use_ng_routes
        self.ng_neighborhood_size = ng_neighborhood_size

        # Counters updated each call to solve()
        self.labels_generated: int = 0
        self.labels_dominated: int = 0
        self.labels_infeasible: int = 0

        # Precompute ng-neighborhoods once at construction time.
        # This is O(n^2 log n) but performed only once per solver instance.
        # When use_ng_routes=False the neighborhoods are never consulted, but
        # computing them keeps the initialisation path uniform.
        self.ng_neighborhoods: Dict[int, Set[int]] = self._compute_ng_neighborhoods()

    # ------------------------------------------------------------------
    # Neighborhood precomputation
    # ------------------------------------------------------------------

    def _compute_ng_neighborhoods(self) -> Dict[int, Set[int]]:
        """
        Precompute the ng-neighborhood N_i for every node including the depot.

        N_i is defined as node i itself plus the ``ng_neighborhood_size - 1``
        nodes with the smallest arc cost from i (Baldacci et al. 2011, Def. 1).
        All nodes (depot = 0, customers 1 … n_nodes) are eligible neighbors.

        The effective size is capped at the total number of nodes so that
        ``ng_neighborhood_size`` may safely exceed the instance size without
        error.

        Returns:
            Mapping node_id → frozenset-compatible set of neighbor IDs.
            Node i is always included in N_i.
        """
        all_nodes = list(range(self.n_nodes + 1))  # depot + customers
        k = min(self.ng_neighborhood_size, len(all_nodes))
        neighborhoods: Dict[int, Set[int]] = {}

        for i in all_nodes:
            # Rank all other nodes by distance from i.
            distances = sorted((self.cost_matrix[i, j], j) for j in all_nodes if j != i)
            # Take the k-1 closest; then add i itself to complete N_i.
            closest: Set[int] = {j for _, j in distances[: k - 1]}
            closest.add(i)
            neighborhoods[i] = closest

        return neighborhoods

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        dual_values: Dict[Union[int, str], float],
        max_routes: int = 10,
        branching_constraints: Optional[List[Any]] = None,
        capacity_cut_duals: Optional[Dict[FrozenSet[int], float]] = None,
    ) -> List[Tuple[List[int], float]]:
        """
        Solve the RCSPP and return routes with positive reduced cost.

        Only node-coverage dual values from the master problem are used in the
        reduced-cost formula.  Capacity constraints are enforced implicitly by
        the resource tracker in the label extension step.  Edge-branching
        and Ryan-Foster constraints are enforced during extension and at
        route completion.

        When ``use_ng_routes=True``, the ng-route relaxation (Baldacci et al.
        2011) is used in place of exact ESPPRC.  The relaxation may
        occasionally generate routes with short cycles; these are almost always
        unprofitable after dual subtraction and are rare in practice.

        Args:
            dual_values: Dual values from the master problem.  Keys are
                customer node IDs (int) and optionally ``"vehicle_limit"``
                (str).
            max_routes: Maximum number of routes to return.
            branching_constraints: Active edge branching constraints at the
                current B&B node, or None / empty list for the root.

        Returns:
            List of (route_nodes, reduced_cost) tuples sorted by descending
            reduced cost.  route_nodes excludes the depot.
        """
        self.labels_generated = 0
        self.labels_dominated = 0
        self.labels_infeasible = 0

        self.dual_values = dual_values
        self.capacity_cut_duals = capacity_cut_duals or {}
        self._compute_node_reduced_costs()

        constraints: List[Any] = branching_constraints or []
        (
            forbidden_arcs,
            required_successors,
            required_predecessors,
            rf_separate,
            rf_together,
        ) = self._preprocess_constraints(constraints)

        routes = self._label_correcting_algorithm(
            max_routes, forbidden_arcs, required_successors, required_predecessors, rf_separate, rf_together
        )
        routes.sort(key=lambda x: x[1], reverse=True)
        return routes[:max_routes]

    # ------------------------------------------------------------------
    # Constraint pre-processing
    # ------------------------------------------------------------------

    def _preprocess_constraints(
        self,
        constraints: List[Any],
    ) -> Tuple[FrozenSet[Tuple[int, int]], Dict[int, int], Dict[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Convert branching constraints into fast lookup structures.

        Uses duck-typing to handle both EdgeBranchingConstraint and
        RyanFosterBranchingConstraint without hard import dependencies.

        Args:
            constraints: Active branching constraints.

        Returns:
            Tuple (forbidden_arcs, req_successors, req_predecessors,
                   rf_separate_pairs, rf_together_pairs).
        """
        forbidden: Set[Tuple[int, int]] = set()
        req_succ: Dict[int, int] = {}
        req_pred: Dict[int, int] = {}
        rf_separate: Set[Tuple[int, int]] = set()
        rf_together: Set[Tuple[int, int]] = set()

        for c in constraints:
            if hasattr(c, "must_use"):
                # EdgeBranchingConstraint
                if not c.must_use:
                    forbidden.add((c.u, c.v))
                else:
                    if c.u in req_succ and req_succ[c.u] != c.v:
                        raise ValueError(f"Contradictory req-successor at {c.u}")
                    if c.v in req_pred and req_pred[c.v] != c.u:
                        raise ValueError(f"Contradictory req-predecessor at {c.v}")
                    req_succ[c.u] = c.v
                    req_pred[c.v] = c.u
            elif hasattr(c, "together"):
                # RyanFosterBranchingConstraint
                pair = (min(c.node_r, c.node_s), max(c.node_r, c.node_s))
                if not c.together:
                    rf_separate.add(pair)
                else:
                    rf_together.add(pair)

        return frozenset(forbidden), req_succ, req_pred, rf_separate, rf_together

    # ------------------------------------------------------------------
    # Reduced-cost helper
    # ------------------------------------------------------------------

    def _compute_node_reduced_costs(self) -> None:
        """
        Pre-compute the per-node reduced-cost contribution.

        node_rc_i = waste_i * R  −  dual_i
        """
        self.node_reduced_costs: Dict[int, float] = {}
        for node in range(1, self.n_nodes + 1):
            revenue = self.wastes.get(node, 0.0) * self.R
            dual = self.dual_values.get(node, 0.0)
            self.node_reduced_costs[node] = revenue - dual

    # ------------------------------------------------------------------
    # Label-correcting DP
    # ------------------------------------------------------------------

    def _label_correcting_algorithm(  # noqa: C901
        self,
        max_routes: int,
        forbidden_arcs: FrozenSet[Tuple[int, int]],
        required_successors: Dict[int, int],
        required_predecessors: Dict[int, int],
        rf_separate: Set[Tuple[int, int]],
        rf_together: Set[Tuple[int, int]],
    ) -> List[Tuple[List[int], float]]:
        """
        Forward label-correcting algorithm for ESPPRC / ng-RCSPP.

        Extends labels from the depot through customer nodes back to the depot.
        At each extension step (u → v), edge-branching rules and the
        appropriate elementarity / ng-feasibility check are applied before a
        label is created.

        ng-route mode vs exact ESPPRC — the *only* two divergence points:

        1. **Feasibility gate** (line in the inner loop):
              ng mode:   v ∉ current.ng_memory
              exact:     v ∉ current.visited

        2. **Dominance check** forwarded to Label.dominates(use_ng=...):
              ng mode:   self.ng_memory ⊆ other.ng_memory
              exact:     self.visited   ⊆ other.visited

        All other logic (arc costs, duals, capacity, branching constraints,
        depot return) is identical in both modes.

        Args:
            max_routes: Upper bound on the number of routes to collect.
            forbidden_arcs: Pairs (u, v) that must not be traversed.
            required_successors: u → w; from u the only valid next customer is w.
            required_predecessors: v → x; v may only be entered from x or depot.

        Returns:
            List of (route_nodes, reduced_cost) with reduced_cost > 1e-6.
        """
        use_ng = self.use_ng_routes

        initial_label = Label(
            reduced_cost=0.0,
            node=self.depot,
            cost=0.0,
            load=0.0,
            revenue=0.0,
            path=[self.depot],
            visited=set(),
            ng_memory=set(),
            rf_unmatched=frozenset(),
            parent=None,
        )

        labels_at_node: Dict[int, List[Label]] = {self.depot: [initial_label]}
        unprocessed: deque[Label] = deque([initial_label])
        completed_routes: List[Label] = []

        while unprocessed:
            current = unprocessed.popleft()
            u = current.node

            # Determine candidate next customers.
            candidate_nodes = [required_successors[u]] if u in required_successors else range(1, self.n_nodes + 1)

            for v in candidate_nodes:
                # ---- Elementarity / ng-feasibility -------------------------
                # ng mode:    block v only if it is in the current ng-memory.
                # exact mode: block v if it was visited anywhere on the path.
                if use_ng:
                    if v in current.ng_memory:
                        continue
                else:
                    if v in current.visited:
                        continue

                # ---- Ryan-Foster Separation (Rule 4) -----------------------
                # Reject v if it forms a separate-pair with any node in path.
                violated_rf = False
                for node_in_path in current.visited:
                    pair = (min(v, node_in_path), max(v, node_in_path))
                    if pair in rf_separate:
                        violated_rf = True
                        break
                if violated_rf:
                    continue

                # ---- Edge-branching Rule 1: Forbidden arc (u → v) ----------
                if (u, v) in forbidden_arcs:
                    self.labels_infeasible += 1
                    continue

                # ---- Edge-branching Rule 3: Required-predecessor for v ------
                if v in required_predecessors and required_predecessors[v] != u:
                    self.labels_infeasible += 1
                    continue

                # ---- Attempt label extension --------------------------------
                new_label = self._extend_label(current, v, rf_together)
                if new_label is None:
                    self.labels_infeasible += 1
                    continue

                # Path length cap to prevent infinite loops in zero-waste cycles.
                if len(new_label.path) > self.n_nodes + 2:
                    continue

                self.labels_generated += 1

                # ---- Dominance check ---------------------------------------
                existing = labels_at_node.get(v, [])
                if self._is_dominated(new_label, existing, use_ng):
                    self.labels_dominated += 1
                    continue

                # Remove labels that new_label now dominates.
                labels_at_node[v] = [lbl for lbl in existing if not new_label.dominates(lbl, use_ng=use_ng)]
                labels_at_node[v].append(new_label)
                unprocessed.append(new_label)

            # ---- Try returning to the depot --------------------------------
            if u != self.depot:
                req_next = required_successors.get(u)
                if req_next is not None and req_next != self.depot:
                    pass  # u must visit req_next before closing.
                else:
                    final_label = self._extend_to_depot(current)
                    # Ryan-Foster Together (Rule 5)
                    # Discard if exactly one member of a together-pair is present.
                    if final_label is not None and not final_label.rf_unmatched and final_label.reduced_cost > 1e-6:
                        completed_routes.append(final_label)

        # Collect routes with positive reduced cost.
        routes: List[Tuple[List[int], float]] = []
        for label in completed_routes:
            full_path = label.reconstruct_path()
            route_nodes = [n for n in full_path if n != self.depot]
            if label.reduced_cost > 1e-6:
                routes.append((route_nodes, label.reduced_cost))

        return routes

    # ------------------------------------------------------------------
    # Label extension primitives
    # ------------------------------------------------------------------

    def _extend_label(self, label: Label, next_node: int, rf_together: Set[Tuple[int, int]]) -> Optional[Label]:
        """
        Extend *label* by visiting *next_node*.

        Incremental Update for Ryan-Foster 'Together' obligations:
            rf_unmatched_new = rf_unmatched_old ⊕ {next_node | partner in visited}
            (We track nodes in together-pairs that don't yet have their partner).

        ng-memory update (Baldacci et al. 2011, Section 3):
            M_{next_node} = (M_u ∪ {next_node}) ∩ N_{next_node}

        The intersection with N_{next_node} is the key step: it drops nodes
        from the memory that are no longer relevant at the new position
        (i.e. nodes not in the neighborhood of next_node), keeping the
        memory compact and dominance comparisons cheap.

        The ``visited`` set is always extended regardless of mode so that
        path reconstruction and exact ESPPRC mode remain correct.

        Args:
            label: Current label (partial path ending at label.node).
            next_node: Customer node to append to the path.

        Returns:
            New extended label, or None if the extension violates capacity.
        """
        node_waste = self.wastes.get(next_node, 0.0)
        new_load = label.load + node_waste
        if new_load > self.capacity:
            return None

        edge_dist = self.cost_matrix[label.node, next_node]
        edge_cost = edge_dist * self.C
        new_cost = label.cost + edge_cost

        node_revenue = node_waste * self.R
        new_revenue = label.revenue + node_revenue

        node_dual = self.dual_values.get(next_node, 0.0)

        # Subtract duals from capacity cuts crossed by edge (label.node -> next_node)
        # An edge crosses cut boundary delta(S) if (u in S) != (v in S).
        crossing_penalty = 0.0
        u, v = label.node, next_node
        for node_set, dual in self.capacity_cut_duals.items():
            if (u in node_set) != (v in node_set):
                crossing_penalty += dual

        new_rc = label.reduced_cost + (node_revenue - edge_cost - node_dual - crossing_penalty)

        # Always maintain the full visited set for path reconstruction.
        new_visited = label.visited | {next_node}

        # Compute the ng-memory update.
        # ng mode:    M_v = (M_u ∪ {v}) ∩ N_v  (Baldacci et al. 2011)
        # exact mode: ng_memory mirrors visited (unused for dominance, but
        #             kept uniform so that _extend_to_depot can copy it
        #             without branching on mode).
        if self.use_ng_routes:
            new_ng_memory = (label.ng_memory | {next_node}) & self.ng_neighborhoods[next_node]
        else:
            new_ng_memory = new_visited

        # Update Ryan-Foster 'Together' unmatched obligations.
        new_unmatched = set(label.rf_unmatched)
        for r, s in rf_together:
            if next_node == r:
                if s in label.visited:
                    new_unmatched.discard(s)
                else:
                    new_unmatched.add(r)
            elif next_node == s:
                if r in label.visited:
                    new_unmatched.discard(r)
                else:
                    new_unmatched.add(s)

        return Label(
            reduced_cost=new_rc,
            node=next_node,
            cost=new_cost,
            load=new_load,
            revenue=new_revenue,
            path=label.path + [next_node],
            visited=new_visited,
            ng_memory=new_ng_memory,
            rf_unmatched=frozenset(new_unmatched),
            parent=label,
        )

    def _extend_to_depot(self, label: Label) -> Optional[Label]:
        """
        Close a partial route by returning from *label.node* to the depot.

        Args:
            label: Current label at a customer node.

        Returns:
            Final label at the depot representing the completed route.
        """
        edge_dist = self.cost_matrix[label.node, self.depot]
        edge_cost = edge_dist * self.C
        new_cost = label.cost + edge_cost

        vehicle_dual = self.dual_values.get("vehicle_limit", 0.0)  # type: ignore[call-overload]

        # Subtract duals from capacity cuts crossed by return edge (label.node -> depot)
        crossing_penalty = 0.0
        u, v = label.node, self.depot
        for node_set, dual in self.capacity_cut_duals.items():
            if (u in node_set) != (v in node_set):
                crossing_penalty += dual

        new_rc = label.reduced_cost - edge_cost - vehicle_dual - crossing_penalty

        return Label(
            reduced_cost=new_rc,
            node=self.depot,
            cost=new_cost,
            load=label.load,
            revenue=label.revenue,
            path=label.path + [self.depot],
            visited=label.visited,
            ng_memory=label.ng_memory,
            rf_unmatched=label.rf_unmatched,
            parent=label,
        )

    # ------------------------------------------------------------------
    # Dominance helpers
    # ------------------------------------------------------------------

    def _is_dominated(
        self,
        label: Label,
        existing_labels: List[Label],
        use_ng: bool,
    ) -> bool:
        """
        Return True if *label* is dominated by any label in *existing_labels*.

        Args:
            label: Candidate label to check.
            existing_labels: Non-dominated labels already stored at the node.
            use_ng: Forwarded to Label.dominates to select the correct
                state-field for the subset check.

        Returns:
            True if at least one existing label dominates *label*.
        """
        return any(existing.dominates(label, use_ng=use_ng) for existing in existing_labels)

    # ------------------------------------------------------------------
    # Route-detail helper
    # ------------------------------------------------------------------

    def compute_route_details(
        self,
        route: List[int],
    ) -> Tuple[float, float, float, Set[int]]:
        """
        Compute cost, revenue, load, and node coverage for a given route.

        Args:
            route: Ordered list of customer nodes (depot excluded).

        Returns:
            Tuple of (cost, revenue, load, node_coverage).
        """
        total_distance = 0.0
        prev = self.depot
        for node in route:
            total_distance += self.cost_matrix[prev, node]
            prev = node
        total_distance += self.cost_matrix[prev, self.depot]

        total_waste = sum(self.wastes.get(node, 0.0) for node in route)
        revenue = total_waste * self.R
        cost = total_distance * self.C

        return cost, revenue, total_waste, set(route)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, int]:
        """
        Return label-processing statistics from the most recent solve.

        Returns:
            Dictionary with keys:
                labels_generated  – total labels created and accepted,
                labels_dominated  – labels discarded by dominance,
                labels_infeasible – labels discarded for infeasibility.
        """
        return {
            "labels_generated": self.labels_generated,
            "labels_dominated": self.labels_dominated,
            "labels_infeasible": self.labels_infeasible,
        }
