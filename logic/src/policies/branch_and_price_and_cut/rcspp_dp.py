r"""
RCSPP Pricing Subproblem Solver.

Solves the Resource-Constrained Shortest Path Problem (RCSPP) to identify
profitable columns (routes) for the BPC Master Problem.

Theoretical Deviation Note:
---------------------------
Unlike the pure ODIMCF formulation in Barnhart, Hane, and Vance (2000) which
manages state-space explosion strictly via branching, this implementation intentionally
incorporates Subset-Row Inequalities (SRIs) and accommodates Ryan-Foster branching.
These algorithmic choices fundamentally expand the DP state space by requiring auxiliary
resource dimensions (e.g., tracking SRI parity constraints). To mitigate this explosion
and maintain tractability, we rely heavily on the ng-route relaxation framework,
balancing bounding strength with computational viability for the VRPP.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union, cast

if TYPE_CHECKING:
    from .branching import AnyBranchingConstraint

import numpy as np

from .master_problem import Route


@dataclass(order=True)
class Label:
    """
    Label for dynamic programming state in RCSPP / ng-RCSPP.
    """

    # Primary sort key — higher reduced cost is preferred.
    reduced_cost: float = field(compare=True)

    # State fields
    node: int = field(compare=False)
    load: float = field(compare=False)
    visited: Set[int] = field(default_factory=set, compare=False)
    ng_memory: Set[int] = field(default_factory=set, compare=False)
    rf_unmatched: FrozenSet[int] = field(default_factory=frozenset, compare=False)
    parent: Optional["Label"] = field(default=None, compare=False, repr=False)
    sri_state: Tuple[int, ...] = field(default_factory=tuple, compare=False)

    def dominates(
        self,
        other: "Label",
        use_ng: bool = False,
        epsilon: float = 1e-6,
        sri_dual_values: Optional[List[float]] = None,
    ) -> bool:
        if self.node != other.node:
            return False
        if self.load > other.load + epsilon:
            return False
        if not self.rf_unmatched.issubset(other.rf_unmatched):
            return False
        if len(self.sri_state) != len(other.sri_state):
            return False

        total_potential_penalty = 0.0
        if sri_dual_values is not None:
            for s, o, dual in zip(self.sri_state, other.sri_state, sri_dual_values):
                if s == 1 and o in (0, 2):
                    total_potential_penalty += dual
        else:
            if any(s > o for s, o in zip(self.sri_state, other.sri_state)):
                return False

        if self.reduced_cost - total_potential_penalty < other.reduced_cost - epsilon:
            return False

        if use_ng:
            return self.ng_memory.issubset(other.ng_memory)
        else:
            return self.visited.issubset(other.visited)

    def is_feasible(self, capacity: float) -> bool:
        return self.load <= capacity

    def reconstruct_path(self) -> List[int]:
        if self.parent is None:
            return [self.node]
        return self.parent.reconstruct_path() + [self.node]


class RCSPPSolver:
    """
    Exact / ng-relaxed solver for the Resource-Constrained Shortest Path Problem.
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
        ng_neighborhoods: Optional[Dict[int, Set[int]]] = None,
    ) -> None:
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

        self.labels_generated: int = 0
        self.labels_dominated: int = 0
        self.labels_infeasible: int = 0
        self.last_max_rc: float = -float("inf")
        self.dual_values: Dict[int, float] = {}
        self.bounds_to: np.ndarray = np.zeros(self.n_nodes + 1)
        self.bounds_from: np.ndarray = np.zeros(self.n_nodes + 1)
        self.fixed_arcs: Set[Tuple[int, int]] = set()

        if ng_neighborhoods is not None:
            self.ng_neighborhoods = ng_neighborhoods
        else:
            self.ng_neighborhoods = self._compute_ng_neighborhoods()

        # Fix 8: Precompute distance-sorted neighbor lists
        self._sorted_neighbors: Dict[int, List[int]] = self._precompute_sorted_neighbors()

    def _precompute_sorted_neighbors(self) -> Dict[int, List[int]]:
        """Precompute distance-sorted neighbor lists for all nodes."""
        result: Dict[int, List[int]] = {}
        for i in range(self.n_nodes + 1):
            dists = [(self.cost_matrix[i, j], j) for j in range(self.n_nodes + 1) if j != i]
            dists.sort()
            result[i] = [j for _, j in dists]
        return result

    def _compute_ng_neighborhoods(self) -> Dict[int, Set[int]]:
        customer_nodes = list(range(1, self.n_nodes + 1))
        k = min(self.ng_neighborhood_size, len(customer_nodes))
        neighborhoods: Dict[int, Set[int]] = {}
        for i in customer_nodes:
            distances = sorted((self.cost_matrix[i, j], j) for j in customer_nodes if j != i)
            closest: Set[int] = {j for _, j in distances[: k - 1]}
            closest.add(i)
            neighborhoods[i] = closest
        neighborhoods[self.depot] = set()
        return neighborhoods

    def expand_ng_neighborhoods(self, cycles: List[Tuple[int, ...]]) -> int:
        added_count = 0
        for cycle in cycles:
            cycle_set = set(cycle)
            for i in cycle:
                if i == self.depot:
                    continue
                before = len(self.ng_neighborhoods[i])
                self.ng_neighborhoods[i].update(cycle_set)
                added_count += len(self.ng_neighborhoods[i]) - before
        return added_count

    def save_ng_snapshot(self) -> Dict[int, Set[int]]:
        """Return a deep copy of the current ng-neighborhoods."""
        return {k: set(v) for k, v in self.ng_neighborhoods.items()}

    def restore_ng_snapshot(self, snapshot: Dict[int, Set[int]]) -> None:
        """Restore ng-neighborhoods from a previously saved snapshot."""
        self.ng_neighborhoods = {k: set(v) for k, v in snapshot.items()}

    def enforce_elementarity(self, nodes: List[int]) -> int:
        added_count = 0
        nodes_to_enforce = [n for n in nodes if n != self.depot]
        if not nodes_to_enforce:
            return 0
        for i in range(self.n_nodes + 1):
            if i == self.depot:
                continue
            before = len(self.ng_neighborhoods[i])
            self.ng_neighborhoods[i].update(nodes_to_enforce)
            added_count += len(self.ng_neighborhoods[i]) - before
        return added_count

    def solve(
        self,
        dual_values: Union[Dict[int, float], Dict[str, Any]],
        max_routes: int = 10,
        branching_constraints: Optional[List[AnyBranchingConstraint]] = None,
        capacity_cut_duals: Optional[Dict[FrozenSet[int], float]] = None,
        sri_cut_duals: Optional[Dict[FrozenSet[int], float]] = None,
        edge_clique_cut_duals: Optional[Dict[Tuple[int, int], float]] = None,
        forced_nodes: Optional[Set[int]] = None,
        rf_conflicts: Optional[Dict[int, Set[int]]] = None,
        is_farkas: bool = False,
        exact_mode: bool = False,
    ) -> List[Route]:
        self._rcc_duals_for_bounds = (
            cast(Dict[str, Any], dual_values).get("rcc_duals", {})
            if isinstance(dual_values, dict)
            else capacity_cut_duals or {}
        )
        # 1. Dual handling
        if isinstance(dual_values, dict) and "node_duals" in dual_values:
            # Composite duals from Master Problem
            complex_duals = cast(Dict[str, Any], dual_values)
            node_duals = complex_duals.get("node_duals", {})
            rcc_duals = complex_duals.get("rcc_duals", {})
            sri_duals = complex_duals.get("sri_duals", {})
            # Edge-clique duals are keyed by canonical (min, max) edge tuples.
            # They penalise the reduced cost whenever the DP traverses a cut edge.
            edge_clique_duals: Dict[Tuple[int, int], float] = complex_duals.get("edge_clique_duals", {})
            self.vehicle_dual = complex_duals.get("vehicle_limit", 0.0)
            # LCI duals (γ_S) and node-level lifting coefficients (α_i).
            # Per Barnhart et al. (2000) §4.2, when traversing a node i in cover S:
            #   rc_delta -= α_i · γ_S
            lci_duals_raw: Dict[FrozenSet[int], float] = complex_duals.get("lci_duals", {})
            lci_node_alphas_raw: Dict[FrozenSet[int], Dict[int, float]] = complex_duals.get("lci_node_alphas", {})
        else:
            # Simple node duals only
            node_duals = dual_values  # type: ignore[assignment]
            rcc_duals = capacity_cut_duals or {}
            sri_duals = sri_cut_duals or {}
            edge_clique_duals = edge_clique_cut_duals or {}
            self.vehicle_dual = 0.0
            lci_duals_raw = {}
            lci_node_alphas_raw = {}

        # Build LCI cover items for DP extension: list of (cover_set, node_alpha, dual_value).
        # Only include cuts with non-negligible dual to avoid wasted iteration in the inner loop.
        lci_cover_items: List[Tuple[FrozenSet[int], Dict[int, float], float]] = [
            (cover_set, lci_node_alphas_raw.get(cover_set, {}), dual)
            for cover_set, dual in lci_duals_raw.items()
            if dual > 1e-8
        ]

        # 2. Reset state
        self.labels_generated = 0
        self.labels_dominated = 0
        self.labels_infeasible = 0
        self.dual_values = node_duals
        self.is_farkas = is_farkas
        self.forced_nodes = forced_nodes or set()
        self.rf_conflicts = rf_conflicts or {}

        # Task 2: Completion Bounds
        self._compute_completion_bounds()

        # 3. SRI pre-processing
        active_sri_items = [(k, v) for k, v in sri_duals.items() if v > 1e-4]
        active_sri_items.sort(key=lambda x: x[1], reverse=True)
        active_sri_subsets = sorted([k for k, v in active_sri_items], key=hash)
        sri_dual_values = [sri_duals[s] for s in active_sri_subsets]
        self.sri_dual_values = sri_dual_values
        node_to_sri: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes + 1)}
        for idx, s in enumerate(active_sri_subsets):
            for node in s:
                if node in node_to_sri:
                    node_to_sri[node].append(idx)

        # 4. Reduced costs
        if not self.is_farkas:
            self.node_reduced_costs = {
                n: self.wastes.get(n, 0.0) * self.R - node_duals.get(n, 0.0) for n in range(1, self.n_nodes + 1)
            }
        else:
            self.node_reduced_costs = {n: node_duals.get(n, 0.0) for n in range(1, self.n_nodes + 1)}

        # 5. Constraints
        constraints: List[Any] = branching_constraints or []
        (forbidden_arcs, req_succ, req_pred, rf_sep, rf_tog) = self._preprocess_constraints(constraints)

        original_use_ng = self.use_ng_routes
        try:
            routes = self._label_correcting_algorithm(
                max_routes,
                forbidden_arcs,
                req_succ,
                req_pred,
                rf_sep,
                rf_tog,
                rcc_duals=rcc_duals,
                active_sri_subsets=active_sri_subsets,
                sri_dual_values=sri_dual_values,
                node_to_sri=node_to_sri,
                forced_nodes=self.forced_nodes,
                edge_clique_duals=edge_clique_duals,
                lci_cover_items=lci_cover_items,
                exact_mode=exact_mode,
            )
        finally:
            self.use_ng_routes = original_use_ng

        routes.sort(key=lambda x: getattr(x, "reduced_cost", 0.0), reverse=True)
        return routes[:max_routes]

    def _compute_completion_bounds(self):
        # Fix 11: Incorporate RCC duals into completion bounds for tighter pruning.
        # RCC duals are arc-crossing penalties with a clear per-edge interpretation.
        self.bounds_to = np.zeros(self.n_nodes + 1)
        self.bounds_from = np.zeros(self.n_nodes + 1)
        rcc_duals = getattr(self, "_rcc_duals_for_bounds", {})
        nodes = list(range(self.n_nodes + 1))
        for _ in range(self.n_nodes):
            changed = False
            for i in nodes:
                # To Depot
                for j in nodes:
                    if i == j:
                        continue
                    edge_cost = self.cost_matrix[i, j] * self.C
                    node_dual = self.dual_values.get(j, 0.0)
                    node_rev = self.wastes.get(j, 0.0) * self.R

                    # Add RCC dual contributions: for each cut set S containing j,
                    # subtract the dual if crossing the cut boundary.
                    rcc_penalty = sum(mu for S, mu in rcc_duals.items() if j in S and i not in S)

                    val = (node_rev - edge_cost - node_dual - rcc_penalty) + self.bounds_to[j]
                    if val > self.bounds_to[i]:
                        self.bounds_to[i] = val
                        changed = True
                # From Depot
                for j in nodes:
                    if i == j:
                        continue
                    edge_cost = self.cost_matrix[j, i] * self.C
                    node_dual = self.dual_values.get(i, 0.0)
                    node_rev = self.wastes.get(i, 0.0) * self.R

                    rcc_penalty = sum(mu for S, mu in rcc_duals.items() if i in S and j not in S)

                    val = (node_rev - edge_cost - node_dual - rcc_penalty) + self.bounds_from[j]
                    if val > self.bounds_from[i]:
                        self.bounds_from[i] = val
                        changed = True
            if not changed:
                break

    def compute_route_details(self, route_nodes: List[int]) -> Route:
        """
        Compute cost/revenue/load for a customer-only node list (no depot bookends).
        Depot arcs are added explicitly: depot→route_nodes[0] and
        route_nodes[-1]→depot.
        """
        if not route_nodes:
            return Route(nodes=[], cost=0.0, revenue=0.0, load=0.0, node_coverage=set())

        cost = 0.0
        rev = 0.0

        # Depot → first customer
        cost += self.cost_matrix[self.depot, route_nodes[0]] * self.C

        # Customer arcs
        for i in range(len(route_nodes) - 1):
            cost += self.cost_matrix[route_nodes[i], route_nodes[i + 1]] * self.C
            rev += self.wastes.get(route_nodes[i], 0.0) * self.R

        # Last customer revenue + last customer → depot
        rev += self.wastes.get(route_nodes[-1], 0.0) * self.R
        cost += self.cost_matrix[route_nodes[-1], self.depot] * self.C

        load = sum(self.wastes.get(n, 0.0) for n in route_nodes)

        # Reduced cost under current duals
        rc = rev - cost
        for node in route_nodes:
            rc -= self.dual_values.get(node, 0.0)
        rc -= self.vehicle_dual

        route = Route(
            nodes=route_nodes,
            cost=cost,
            revenue=rev,
            load=load,
            node_coverage=set(route_nodes),
        )
        route.reduced_cost = rc
        return route

    def _preprocess_constraints(self, constraints: List[Any]):
        forbidden: Set[Tuple[int, int]] = set()
        req_succ: Dict[int, int] = {}
        req_pred: Dict[int, int] = {}
        rf_separate: Set[Tuple[int, int]] = set()
        rf_together: Set[Tuple[int, int]] = set()
        for c in constraints:
            if hasattr(c, "must_use"):
                if not c.must_use:
                    forbidden.add((c.u, c.v))
                else:
                    req_succ[c.u] = c.v
                    req_pred[c.v] = c.u
            elif hasattr(c, "together"):
                pair = (min(c.node_r, c.node_s), max(c.node_r, c.node_s))
                if not c.together:
                    rf_separate.add(pair)
                else:
                    rf_together.add(pair)
        return frozenset(forbidden), req_succ, req_pred, rf_separate, rf_together

    def _label_correcting_algorithm(  # noqa: C901
        self,
        max_routes: int,
        forbidden_arcs: FrozenSet[Tuple[int, int]],
        required_successors: Dict[int, int],
        required_predecessors: Dict[int, int],
        rf_separate: Set[Tuple[int, int]],
        rf_together: Set[Tuple[int, int]],
        rcc_duals: Dict[FrozenSet[int], float],
        active_sri_subsets: List[FrozenSet[int]],
        sri_dual_values: List[float],
        node_to_sri: Dict[int, List[int]],
        forced_nodes: Set[int],
        sri_memory_nodes: Optional[List[Set[int]]] = None,
        edge_clique_duals: Optional[Dict[Tuple[int, int], float]] = None,
        lci_cover_items: Optional[List[Tuple[FrozenSet[int], Dict[int, float], float]]] = None,
        exact_mode: bool = False,
    ) -> List[Route]:
        """Forward label-correcting algorithm (priority-queue order)."""
        use_ng = self.use_ng_routes
        initial_sri = tuple([0] * len(active_sri_subsets)) if active_sri_subsets else ()

        start = Label(
            reduced_cost=0.0,
            node=self.depot,
            load=0.0,
            visited=set(),
            ng_memory=set(),
            rf_unmatched=frozenset(),
            parent=None,
            sri_state=initial_sri,
        )

        # Max-heap via negated reduced_cost; integer counter breaks ties.
        _counter = 0
        queue: list = []
        heapq.heappush(queue, (-start.reduced_cost, _counter, start))

        labels_at_node: Dict[int, List[Label]] = {self.depot: [start]}
        completed_routes: List[Label] = []
        global_max_rc = -float("inf")

        while queue:
            _, _, current = heapq.heappop(queue)
            u = current.node

            # Candidate successors
            neighbor_limit = self.n_nodes if exact_mode else 20
            candidate_nodes = (
                [required_successors[u]] if u in required_successors else self._get_neighbors(u, neighbor_limit)
            )

            for v in candidate_nodes:
                is_required = u in required_successors and required_successors[u] == v

                # Elementarity
                if v in current.visited:
                    continue

                # ng-feasibility
                if not is_required and use_ng and v in current.ng_memory:
                    continue

                # Ryan-Foster separation
                if any((min(v, n), max(v, n)) in rf_separate for n in current.visited):
                    continue

                # Edge constraints
                if (u, v) in forbidden_arcs:
                    continue
                if v in required_predecessors and required_predecessors[v] != u:
                    continue

                new_label = self._extend_label(
                    label=current,
                    next_node=v,
                    forbidden=forbidden_arcs,
                    rcc_duals=rcc_duals,
                    active_sri=active_sri_subsets,
                    sri_duals=sri_dual_values,
                    node_to_sri=node_to_sri,
                    sri_memory_nodes=sri_memory_nodes,
                    edge_clique_duals=edge_clique_duals,
                    lci_cover_items=lci_cover_items,
                )
                if new_label is None:
                    continue

                # Fix 7: Early exit for fixed arcs before dominance check
                if (u, v) in self.fixed_arcs:
                    continue

                if len(new_label.visited) > self.n_nodes:
                    continue

                self.labels_generated += 1
                existing = labels_at_node.get(v, [])
                if any(e.dominates(new_label, use_ng=use_ng, sri_dual_values=sri_dual_values) for e in existing):
                    self.labels_dominated += 1
                    continue

                labels_at_node[v] = [lbl for lbl in existing if not new_label.dominates(lbl, use_ng=use_ng)]
                labels_at_node[v].append(new_label)

                _counter += 1
                heapq.heappush(queue, (-new_label.reduced_cost, _counter, new_label))
                global_max_rc = max(global_max_rc, new_label.reduced_cost)

            # Attempt depot return
            if u != self.depot:
                can_return = not (u in required_successors and required_successors[u] != self.depot)
                # Completion bounds pruning
                bound = current.reduced_cost + self.bounds_to[u]
                if bound < 1e-4:
                    continue

                if can_return:
                    final = self._extend_to_depot(current)
                    if final and not final.rf_unmatched:
                        completed_routes.append(final)
                        global_max_rc = max(global_max_rc, final.reduced_cost)

        self.last_max_rc = global_max_rc
        completed_routes.sort(key=lambda x: x.reduced_cost, reverse=True)

        routes: List[Route] = []
        for label in completed_routes:
            if label.reduced_cost <= 1e-6:
                continue
            full_path = label.reconstruct_path()
            route_nodes = [n for n in full_path if n != self.depot]
            cost, rev, load, coverage = self._route_details_from_path(route_nodes)
            rt = Route(route_nodes, cost, rev, load, coverage)
            rt.reduced_cost = label.reduced_cost
            routes.append(rt)
            if len(routes) >= max_routes:
                break

        return routes

    def _get_neighbors(self, node: int, limit: int) -> List[int]:
        return self._sorted_neighbors[node][:limit]

    def _extend_to_depot(self, label: Label) -> Optional[Label]:
        edge_cost = self.cost_matrix[label.node, self.depot] * self.C
        # Extract vehicle limit dual
        vehicle_dual = getattr(self, "vehicle_dual", 0.0)

        # RCC crossing penalty for the depot-return arc.
        # The depot (node 0) is never inside a customer cut set S, so an arc
        # from label.node ∈ S back to the depot is always a boundary exit crossing
        # and must carry the RCC dual penalty.  Omitting this caused the pricing DP
        # to over-value depot returns from within a cut set, generating columns that
        # violated RCC constraints.
        crossing_penalty = 0.0
        rcc_duals_depot = getattr(self, "_rcc_duals_for_bounds", {})
        for subset, dual in rcc_duals_depot.items():
            if label.node in subset and self.depot not in subset:
                crossing_penalty += dual

        new_rc = label.reduced_cost - edge_cost - vehicle_dual - crossing_penalty

        return Label(
            reduced_cost=new_rc,
            node=self.depot,
            load=label.load,
            visited=label.visited,
            ng_memory=label.ng_memory,
            rf_unmatched=label.rf_unmatched,
            parent=label,
            sri_state=label.sri_state,
        )

    def _route_details_from_path(self, route_nodes: List[int]):
        """Compute (cost, revenue, load, coverage) for a customer-only node list."""
        prev = self.depot
        cost = 0.0
        for n in route_nodes:
            cost += self.cost_matrix[prev, n] * self.C
            prev = n
        cost += self.cost_matrix[prev, self.depot] * self.C
        waste = sum(self.wastes.get(n, 0.0) for n in route_nodes)
        return cost, waste * self.R, waste, set(route_nodes)

    def _extend_label(
        self,
        label: Label,
        next_node: int,
        forbidden: FrozenSet[Tuple[int, int]],
        rcc_duals: Dict[FrozenSet[int], float],
        active_sri: List[FrozenSet[int]],
        sri_duals: List[float],
        node_to_sri: Dict[int, List[int]],
        sri_memory_nodes: Optional[List[Set[int]]] = None,
        edge_clique_duals: Optional[Dict[Tuple[int, int], float]] = None,
        lci_cover_items: Optional[List[Tuple[FrozenSet[int], Dict[int, float], float]]] = None,
    ) -> Optional[Label]:
        edge = (label.node, next_node)
        if edge in forbidden:
            return None
        load = label.load + self.wastes.get(next_node, 0.0)
        if load > self.capacity + 1e-6:
            return None

        # Task 1: Basic BC calculation (profit - dist - duals)
        dist = self.cost_matrix[edge[0], edge[1]]
        cost = dist * self.C
        rev = self.wastes.get(next_node, 0.0) * self.R

        # Duals
        rc_delta = rev - cost - self.dual_values.get(next_node, 0.0)
        # RCC duals: penalise each boundary CROSSING, not each node visit.
        # A crossing occurs when the arc (label.node → next_node) transitions
        # between the interior and exterior of cut set S.
        # Applying the dual per node visit would double-count arcs within S
        # and miss exit crossings entirely, producing incorrect reduced costs.
        for subset, dual in rcc_duals.items():
            current_in_set = label.node in subset
            next_in_set = next_node in subset
            if current_in_set != next_in_set:
                rc_delta -= dual

        # Edge-Clique dual penalty (Barnhart et al. 2000, §4.2).
        # For a cut on edge (u, v) with dual γ, every route traversing (u, v)
        # must have its reduced cost decreased by γ.  The canonical key is
        # (min(u,v), max(u,v)) to match the master problem's storage convention.
        if edge_clique_duals:
            canonical_edge = (min(label.node, next_node), max(label.node, next_node))
            ec_dual = edge_clique_duals.get(canonical_edge, 0.0)
            if ec_dual > 0.0:
                rc_delta -= ec_dual

        # Fix 4: SRI Dual Penalty
        new_sri = list(label.sri_state)
        for idx in node_to_sri.get(next_node, []):
            if sri_memory_nodes is None or next_node in sri_memory_nodes[idx]:
                curr = new_sri[idx]
                if curr == 0:
                    # 1st visit to a node in S: no penalty yet.
                    new_sri[idx] = 1
                elif curr == 1:
                    # 2nd visit: ⌊2/2⌋ = 1 — apply the dual penalty exactly once.
                    new_sri[idx] = 2
                    rc_delta -= sri_duals[idx]
                # 3rd visit: ⌊3/2⌋ = 1 still — no additional penalty; keep state at 2.

        # LCI Dual Penalty — Barnhart, Hane, Vance (2000) §4.2
        if lci_cover_items:
            for cover_set, node_alpha, dual in lci_cover_items:
                if next_node in cover_set:  # Fix 2: Strict guard for cover set members
                    alpha = node_alpha.get(next_node, 1.0)
                    if alpha > 1e-9:
                        rc_delta -= alpha * dual

        new_rc = label.reduced_cost + rc_delta
        new_visited = label.visited | {next_node}

        # NG-Memory transition
        new_ng = (label.ng_memory & self.ng_neighborhoods[next_node]) | {next_node}

        return Label(
            node=next_node,
            load=load,
            reduced_cost=new_rc,
            visited=new_visited,
            parent=label,
            ng_memory=new_ng,
            sri_state=tuple(new_sri),
            rf_unmatched=label.rf_unmatched,
        )
