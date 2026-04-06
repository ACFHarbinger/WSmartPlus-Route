"""
Resource-Constrained Shortest Path Problem (RCSPP) Solver with ng-Route Relaxation.

This module provides the core pricing engine for the Branch-and-Price-and-Cut (BPC)
algorithm. It implements a forward label-correcting dynamic programming
algorithm to find profitable routes (columns) to add to the Master Problem.

Theoretical Foundation:
    - ng-Route Relaxation (Baldacci et al. 2011): Provides a compromise between
      the high complexity of exact ESPPRC and the weak relaxation of SPPRC.
      Cycles are permitted as long as they don't involve "recently visited"
      nodes within a local memory neighborhood.
    - Subset-Row Inequalities (SRI) (Jepsen et al. 2008): Strengthens the set
      partitioning relaxation by adding valid inequalities for subsets of size 3.
    - Rounded Capacity Cuts (RCC) (Lysgaard et al. 2004): Standard VRP capacity
      constraints based on bin-packing bounds.
    - exact ESPPRC: High-fidelity pricing achieved when ng-neighborhoods
      encompass all nodes, or when using strict elementary visit tracking.

Key Algorithmic Features:
    - Bidirectional Dual Processing: Supports standard maximization duals
      and Farkas duals for infeasibility proof in B&B nodes.
    - Parity Tracking: Tracks visit counts per SRI subset (modulo 2) to correctly
      apply dual penalties for the 3-SRI family.
    - Edge-Based Duals: Dynamic subtraction of penalties for edge-capacity
      clique cuts (LCIs) during label extension.
    - Structural Constraints: Enforces branching decisions (Arc-based or Ryan-Foster)
      to maintain search tree integrity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .master_problem import Route


@dataclass(order=True)
class Label:
    """
    Label for dynamic programming state in RCSPP / ng-RCSPP.

    Represents a partial path from the depot to the current node with
    accumulated resources.  Labels are ordered by reduced cost for efficient
    dominance checking (higher is better for the maximisation objective).
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
    visited: Set[int] = field(default_factory=set, compare=False)

    # ng_memory: compact relaxed state for ng-route dominance / feasibility.
    ng_memory: Set[int] = field(default_factory=set, compare=False)

    # rf_unmatched: nodes from a 'together' pair visited without their partner.
    # Used for enforcing Ryan-Foster branching constraints during pricing.
    rf_unmatched: FrozenSet[int] = field(default_factory=frozenset, compare=False)

    parent: Optional["Label"] = field(default=None, compare=False, repr=False)

    # Subset-Row Inequalities (SRI) state:
    # A tuple where each entry corresponds to an active SRI subset S.
    # State values:
    #   0: No nodes in S visited yet.
    #   1: One node in S visited (potential penalty on next visit).
    #   2: Two nodes in S visited (dual penalty applied, resets on 3rd visit if allowed).
    # This state enables the exact calculation of ⌊ 1/2 * Σ a_{ik} ⌋ dual penalties.
    sri_state: Tuple[int, ...] = field(default_factory=tuple, compare=False)

    def dominates(
        self,
        other: "Label",
        use_ng: bool = False,
        epsilon: float = 1e-6,
    ) -> bool:
        """
        Check whether this label dominates *other* at the same node.
        """
        if self.node != other.node:
            return False
        if self.reduced_cost < other.reduced_cost - epsilon:
            return False
        if self.load > other.load + epsilon:
            return False
        if self.rf_unmatched != other.rf_unmatched:
            return False

        # Exact ESPPRC requirement for SRI states:
        # To maintain mathematical exactness in the presence of Subset-Row Inequalities,
        # two labels are only comparable if they have identical SRI visit counts (parity).
        # If sri_state differs, one label might incur a penalty that the other already
        # paid, or vice versa, making them globally incomparable in the state-space.
        if self.sri_state != other.sri_state:
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

        # Counters updated each call to solve()
        self.labels_generated: int = 0
        self.labels_dominated: int = 0
        self.labels_infeasible: int = 0
        self.last_max_rc: float = -float("inf")

        if ng_neighborhoods is not None:
            self.ng_neighborhoods = ng_neighborhoods
        else:
            self.ng_neighborhoods = self._compute_ng_neighborhoods()

    def _compute_ng_neighborhoods(self) -> Dict[int, Set[int]]:
        all_nodes = list(range(self.n_nodes + 1))
        k = min(self.ng_neighborhood_size, len(all_nodes))
        neighborhoods: Dict[int, Set[int]] = {}

        for i in all_nodes:
            distances = sorted((self.cost_matrix[i, j], j) for j in all_nodes if j != i)
            closest: Set[int] = {j for _, j in distances[: k - 1]}
            closest.add(i)
            neighborhoods[i] = closest

        return neighborhoods

    def solve(
        self,
        dual_values: Dict[Any, Any],
        max_routes: int = 10,
        branching_constraints: Optional[List[Any]] = None,
        capacity_cut_duals: Optional[Dict[FrozenSet[int], float]] = None,
        sri_cut_duals: Optional[Dict[FrozenSet[int], float]] = None,
        lci_cut_duals: Optional[Dict[Tuple[int, int], float]] = None,
        is_farkas: bool = False,
    ) -> List[Route]:
        """
        Solve the Resource-Constrained Shortest Path Problem (RCSPP).

        Executes a forward label-correcting algorithm to identify columns with
        positive reduced cost in the Master Problem. Supports composite dual
        inputs to facilitate simultaneous pricing of node revenues and various
        cutting plane families.

        Args:
            dual_values: Dictionary of dual values. Supports two formats:
                1. Flat: mapping node ID (int) -> dual value.
                2. Composite: Sub-dictionaries "node_duals", "rcc_duals",
                   "sri_duals", and "lci_duals".
            max_routes: Maximum number of profitable routes to return.
            branching_constraints: List of branching objects to enforce.
            capacity_cut_duals: Explicit RCC duals (if not in composite).
            sri_cut_duals: Explicit SRI duals (if not in composite).
            lci_cut_duals: Explicit LCI duals (mapped by (u,v) edges).
            is_farkas: Whether to solve the dual of the Farkas lemma.
                True: proof of node infeasibility (minimizes violation).
                False: standard maximization of reduced cost profit.

        Returns:
            List of generated Route objects with positive reduced cost.
        """
        self.is_farkas = is_farkas

        # Distinguish between old-style flat duals and new-style composite duals
        if isinstance(dual_values, dict) and "node_duals" in dual_values:
            node_duals = dual_values["node_duals"]
            rcc_duals: Dict[FrozenSet[int], float] = dual_values.get("rcc_duals", {})  # type: ignore[assignment]
            sri_duals: Dict[FrozenSet[int], float] = dual_values.get("sri_duals", {})  # type: ignore[assignment]
            lci_duals: Dict[Tuple[int, int], float] = dual_values.get("lci_duals", {})  # type: ignore[assignment]
        else:
            node_duals = dual_values  # type: ignore[assignment]
            rcc_duals: Dict[FrozenSet[int], float] = capacity_cut_duals or {}  # type: ignore[no-redef]
            sri_duals: Dict[FrozenSet[int], float] = sri_cut_duals or {}  # type: ignore[no-redef]
            lci_duals: Dict[Tuple[int, int], float] = lci_cut_duals or {}  # type: ignore[no-redef]

        # Reset statistics
        self.labels_generated = 0
        self.labels_dominated = 0
        self.labels_infeasible = 0
        self.last_max_rc = -float("inf")
        self.dual_values = node_duals  # for _extend_label
        self.capacity_cut_duals = rcc_duals  # for _extend_label
        self.lci_cut_duals = lci_duals  # for _extend_label

        # Pre-compute reduced costs
        if not self.is_farkas:
            self.node_reduced_costs = {
                n: self.wastes.get(n, 0.0) * self.R - node_duals.get(n, 0.0)  # type: ignore[attr-defined]
                for n in range(1, self.n_nodes + 1)
            }
        else:
            self.node_reduced_costs = {n: node_duals.get(n, 0.0) for n in range(1, self.n_nodes + 1)}  # type: ignore[attr-defined]

        # SRI pre-processing
        active_sri_subsets = sorted(list(sri_duals.keys()), key=hash)
        sri_dual_values = [sri_duals[s] for s in active_sri_subsets]
        node_to_sri: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes + 1)}
        for idx, s in enumerate(active_sri_subsets):
            for node in s:
                if node in node_to_sri:
                    node_to_sri[node].append(idx)

        constraints: List[Any] = branching_constraints or []
        (
            forbidden_arcs,
            required_successors,
            required_predecessors,
            rf_separate,
            rf_together,
        ) = self._preprocess_constraints(constraints)

        routes = self._label_correcting_algorithm(
            max_routes,
            forbidden_arcs,
            required_successors,
            required_predecessors,
            rf_separate,
            rf_together,
            rcc_duals=rcc_duals,
            active_sri_subsets=active_sri_subsets,
            sri_dual_values=sri_dual_values,
            node_to_sri=node_to_sri,
        )
        routes.sort(key=lambda x: getattr(x, "reduced_cost", 0.0), reverse=True)
        return routes[:max_routes]

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
        rcc_duals: Optional[Dict[FrozenSet[int], float]] = None,
        active_sri_subsets: Optional[List[FrozenSet[int]]] = None,
        sri_dual_values: Optional[List[float]] = None,
        node_to_sri: Optional[Dict[int, List[int]]] = None,
    ) -> List[Route]:
        use_ng = self.use_ng_routes
        initial_sri_state = tuple([0] * len(active_sri_subsets)) if active_sri_subsets else ()

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
            sri_state=initial_sri_state,
        )

        labels_at_node: Dict[int, List[Label]] = {self.depot: [initial_label]}
        unprocessed: deque[Label] = deque([initial_label])
        completed_routes: List[Label] = []

        while unprocessed:
            current = unprocessed.popleft()
            u = current.node

            # Candidate nodes
            candidate_nodes = [required_successors[u]] if u in required_successors else range(1, self.n_nodes + 1)

            for v in candidate_nodes:
                # Elementarity/ng-feasibility
                if use_ng:
                    if v in current.ng_memory:
                        continue
                else:
                    if v in current.visited:
                        continue

                # Ryan-Foster Separation
                if any((min(v, n), max(v, n)) in rf_separate for n in current.visited):
                    continue

                # Edge constraints
                if (u, v) in forbidden_arcs:
                    continue
                if v in required_predecessors and required_predecessors[v] != u:
                    continue

                # Extend
                new_label = self._extend_label(
                    current, v, rf_together, active_sri_subsets, sri_dual_values, node_to_sri
                )
                if new_label is None:
                    self.labels_infeasible += 1
                    continue

                if len(new_label.path) > self.n_nodes + 2:
                    continue

                self.labels_generated += 1
                existing = labels_at_node.get(v, [])
                if self._is_dominated(new_label, existing, use_ng):
                    self.labels_dominated += 1
                    continue

                labels_at_node[v] = [lbl for lbl in existing if not new_label.dominates(lbl, use_ng=use_ng)]
                labels_at_node[v].append(new_label)
                unprocessed.append(new_label)

            # Depot return
            if u != self.depot:
                if u in required_successors and required_successors[u] != self.depot:
                    pass
                else:
                    final_label = self._extend_to_depot(current)
                    if final_label and not final_label.rf_unmatched:
                        completed_routes.append(final_label)

        # 7. Finalize routes and track bounds
        completed_routes.sort(key=lambda x: x.reduced_cost, reverse=True)

        # RECORD ABSOLUTE MAXIMUM REDUCED COST (even if negative)
        # This is critical for Lagrangian relaxation bounds and proving optimality.
        self.last_max_rc = completed_routes[0].reduced_cost if completed_routes else -float("inf")

        # Build Route objects for positive reduced cost only (column generation)
        routes: List[Route] = []
        for label in completed_routes:
            if label.reduced_cost <= 1e-6:
                continue

            full_path = label.reconstruct_path()
            route_nodes = [n for n in full_path if n != self.depot]

            # Compute physical parameters (cost, rev, load) for Master Problem
            cost, rev, load, coverage = self.compute_route_details(route_nodes)
            rt = Route(route_nodes, cost, rev, load, coverage)
            rt.reduced_cost = label.reduced_cost
            routes.append(rt)

            if len(routes) >= max_routes:
                break

        return routes

    def _extend_label(
        self,
        label: Label,
        next_node: int,
        rf_together: Set[Tuple[int, int]],
        active_sri_subsets: Optional[List[FrozenSet[int]]] = None,
        sri_dual_values: Optional[List[float]] = None,
        node_to_sri: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[Label]:
        node_waste = self.wastes.get(next_node, 0.0)
        new_load = label.load + node_waste
        if new_load > self.capacity:
            return None

        edge_dist = self.cost_matrix[label.node, next_node]
        edge_cost = edge_dist * self.C
        new_cost = label.cost + edge_cost

        node_revenue = node_waste * self.R
        new_revenue = label.revenue + node_revenue
        node_dual = self.dual_values.get(next_node, 0.0)  # type: ignore[attr-defined]

        # 1. SRI Parity Tracking (3-SRIs)
        # For each active SRI subset S, we track the number of nodes visited {0, 1, 2}.
        # The dual penalty γ_S is applied specifically when the visit count transitions
        # from 1 to 2, implementing the floor function ⌊ 1/2 * Σ a_{ik} ⌋.
        # If we visit a 3rd node, the penalty is not applied again (since ⌊3/2⌋ = 1),
        # modeled here by keeping the state at 2 (or resetting if elementary constraints allow).
        new_sri_state = list(label.sri_state)
        sri_penalty = 0.0
        if node_to_sri and next_node in node_to_sri:
            for sri_idx in node_to_sri[next_node]:
                curr = new_sri_state[sri_idx]
                if curr == 1:
                    # Transition 1 -> 2: Subject to dual penalty
                    sri_penalty += sri_dual_values[sri_idx]  # type: ignore[index]
                    new_sri_state[sri_idx] = 2
                elif curr == 0:
                    # Transition 0 -> 1: No penalty yet
                    new_sri_state[sri_idx] = 1

        # 2. LCI Duals (Edge Capacity Cuts)
        # For edges (u,v) with active LCI cuts, we subtract the dual γ_{uv} whenever
        # traversing that specific arc. Note: edge_tuple is canonical (sorted).
        lci_penalty = 0.0
        edge_tuple = tuple(sorted((label.node, next_node)))
        if hasattr(self, "lci_cut_duals") and edge_tuple in self.lci_cut_duals:
            lci_penalty = self.lci_cut_duals[edge_tuple]  # type: ignore[index]

        # 3. Capacity Cuts (RCC/SEC)
        # Penalties are applied when the vehicle crosses the boundary of the set S.
        crossing_penalty = 0.0
        for s, dual in self.capacity_cut_duals.items():
            if (label.node in s) != (next_node in s):
                crossing_penalty += dual

        # 4. Total Step Objective Calculation
        # Standard: Revenue - (Transp Cost + Duals + Cut Penalties)
        # Farkas: Minimize Dual Violation (Reduced cost based strictly on duals)
        step_obj = (
            (node_revenue - edge_cost - node_dual - crossing_penalty - sri_penalty - lci_penalty)
            if not self.is_farkas
            else node_dual
        )
        new_rc = label.reduced_cost + step_obj

        new_visited = label.visited | {next_node}
        if self.use_ng_routes:
            new_ng = (label.ng_memory | {next_node}) & self.ng_neighborhoods[next_node]
        else:
            new_ng = new_visited

        # Together unmatched
        new_unmatched = set(label.rf_unmatched)
        for r, s in rf_together:  # type: ignore[assignment]
            if next_node == r:
                if s in label.visited:
                    new_unmatched.discard(r)
                else:
                    new_unmatched.add(r)
            elif next_node == s:
                if r in label.visited:
                    new_unmatched.discard(r)
                else:
                    new_unmatched.add(s)  # type: ignore[arg-type]

        return Label(
            reduced_cost=new_rc,
            node=next_node,
            cost=new_cost,
            load=new_load,
            revenue=new_revenue,
            path=label.path + [next_node],
            visited=new_visited,
            ng_memory=new_ng,
            rf_unmatched=frozenset(new_unmatched),
            parent=label,
            sri_state=tuple(new_sri_state),
        )

    def _extend_to_depot(self, label: Label) -> Optional[Label]:
        edge_dist = self.cost_matrix[label.node, self.depot]
        edge_cost = edge_dist * self.C
        vehicle_dual = self.dual_values.get("vehicle_limit", 0.0)  # type: ignore[attr-defined]

        crossing_penalty = 0.0
        for s, dual in self.capacity_cut_duals.items():
            if (label.node in s) != (self.depot in s):
                crossing_penalty += dual

        if not self.is_farkas:
            new_rc = label.reduced_cost - edge_cost - vehicle_dual - crossing_penalty
        else:
            new_rc = label.reduced_cost

        return Label(
            reduced_cost=new_rc,
            node=self.depot,
            cost=label.cost + edge_cost,
            load=label.load,
            revenue=label.revenue,
            path=label.path + [self.depot],
            visited=label.visited,
            ng_memory=label.ng_memory,
            rf_unmatched=label.rf_unmatched,
            parent=label,
            sri_state=label.sri_state,
        )

    def _is_dominated(self, label: Label, existing: List[Label], use_ng: bool) -> bool:
        return any(e.dominates(label, use_ng=use_ng) for e in existing)

    def compute_route_details(self, route: List[int]):
        dist = 0.0
        prev = self.depot
        for n in route:
            dist += self.cost_matrix[prev, n]
            prev = n
        dist += self.cost_matrix[prev, self.depot]
        waste = sum(self.wastes.get(n, 0.0) for n in route)
        return dist * self.C, waste * self.R, waste, set(route)

    def get_statistics(self):
        return {
            "labels_generated": self.labels_generated,
            "labels_dominated": self.labels_dominated,
            "labels_infeasible": self.labels_infeasible,
        }
