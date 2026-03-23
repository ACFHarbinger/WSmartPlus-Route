"""
Resource-Constrained Shortest Path Problem (RCSPP) solver using Dynamic Programming.

Implements label-setting algorithm for Elementary Shortest Path Problem with Resource
Constraints (ESPPRC) as described in:
- Irnich & Desaulniers (2005): "Shortest Path Problems with Resource Constraints"
- Feillet et al. (2004): "An exact algorithm for the ESPPRC"

This is the exact pricing subproblem for Branch-and-Price algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .ryan_foster_branching import BranchingConstraint


@dataclass(order=True)
class Label:
    """
    Label for dynamic programming state in RCSPP.

    Represents a partial path from depot to current node with accumulated resources.
    Labels are ordered by reduced cost for efficient dominance checking.
    """

    # Primary sort key (reduced cost, higher is better for maximization)
    reduced_cost: float = field(compare=True)

    # State information (not compared for ordering)
    node: int = field(compare=False)
    cost: float = field(compare=False)
    load: float = field(compare=False)
    revenue: float = field(compare=False)
    path: List[int] = field(default_factory=list, compare=False)
    visited: Set[int] = field(default_factory=set, compare=False)
    parent: Optional["Label"] = field(default=None, compare=False, repr=False)

    def dominates(self, other: "Label", epsilon: float = 1e-6) -> bool:
        """
        Check if this label dominates another label at the same node.

        Label L1 dominates L2 if:
        - L1.reduced_cost >= L2.reduced_cost (better objective)
        - L1.load <= L2.load (less resource consumption)
        - L1.visited ⊆ L2.visited (elementarity)

        Args:
            other: Another label at the same node
            epsilon: Tolerance for numerical comparison

        Returns:
            True if this label dominates the other
        """
        if self.node != other.node:
            return False

        # Check objective (reduced cost for maximization)
        if self.reduced_cost < other.reduced_cost - epsilon:
            return False

        # Check resource (load)
        if self.load > other.load + epsilon:
            return False

        # Check elementarity (visited set must be subset)
        return self.visited.issubset(other.visited)

    def is_feasible(self, capacity: float) -> bool:
        """Check if label respects capacity constraint."""
        return self.load <= capacity

    def reconstruct_path(self) -> List[int]:
        """Reconstruct full path from depot through parent pointers."""
        if self.parent is None:
            return [self.node]
        return self.parent.reconstruct_path() + [self.node]


class RCSPPSolver:
    """
    Exact solver for Resource-Constrained Shortest Path Problem.

    Uses label-setting dynamic programming with dominance rules to find
    elementary paths with maximum reduced cost subject to capacity constraints.

    This is the pricing subproblem for Branch-and-Price column generation.
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
    ):
        """
        Initialize RCSPP solver.

        Args:
            n_nodes: Number of customer nodes (excluding depot)
            cost_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 is depot
            wastes: Dictionary mapping node ID to waste volume
            capacity: Vehicle capacity
            revenue_per_kg: Revenue per unit of waste collected
            cost_per_km: Cost per unit of distance traveled
            mandatory_nodes: Set of mandatory node indices (optional)
        """
        self.n_nodes = n_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.mandatory_nodes = mandatory_nodes or set()
        self.depot = 0

        # Statistics
        self.labels_generated = 0
        self.labels_dominated = 0
        self.labels_infeasible = 0

    def solve(
        self,
        dual_values: Dict[int, float],
        capacity_cut_duals: Optional[List[Tuple[Set[int], float, float]]] = None,
        max_routes: int = 10,
        branching_constraints: Optional[List] = None,
    ) -> List[Tuple[List[int], float]]:
        """
        Solve RCSPP to find routes with positive reduced cost.

        Args:
            dual_values: Dual values from master problem (node coverage constraints)
            max_routes: Maximum number of routes to return
            branching_constraints: Optional branching constraints from Ryan-Foster

        Returns:
            List of (route_nodes, reduced_cost) tuples sorted by descending reduced cost
        """
        # Initialize statistics
        self.labels_generated = 0
        self.labels_dominated = 0
        self.labels_infeasible = 0

        # Store dual values
        self.dual_values = dual_values
        self.capacity_cut_duals = capacity_cut_duals or []

        # Compute modified costs with dual values
        self._compute_reduced_costs()

        # Run label-setting algorithm
        routes = self._label_setting_algorithm(max_routes, branching_constraints)

        # Sort by descending reduced cost
        routes.sort(key=lambda x: x[1], reverse=True)

        return routes[:max_routes]

    def _compute_reduced_costs(self) -> None:
        """
        Compute reduced costs for each node visit.

        For node i:
            reduced_cost_i = revenue_i - dual_i
        """
        self.node_reduced_costs = {}
        for node in range(1, self.n_nodes + 1):
            revenue = self.wastes.get(node, 0.0) * self.R
            dual = self.dual_values.get(node, 0.0)

            # Additional duals from capacity cuts
            # If node i is in S, and the cut is Σ internal edges <= |S|-k(S),
            # then an edge (i,j) with both in S contributes to the dual.
            # However, it's easier to handle this during extension.
            self.node_reduced_costs[node] = revenue - dual

    def _label_setting_algorithm(
        self,
        max_routes: int,
        branching_constraints: Optional[List] = None,
    ) -> List[Tuple[List[int], float]]:
        """
        Label-setting dynamic programming algorithm for ESPPRC.

        Implements forward labeling with dominance rules.

        Args:
            max_routes: Maximum number of routes to generate
            branching_constraints: Optional branching constraints

        Returns:
            List of (route_nodes, reduced_cost) tuples
        """
        # Initialize labels at depot
        initial_label = Label(
            reduced_cost=0.0,
            node=self.depot,
            cost=0.0,
            load=0.0,
            revenue=0.0,
            path=[self.depot],
            visited=set(),
            parent=None,
        )

        # Label storage: node -> list of non-dominated labels
        labels_at_node: Dict[int, List[Label]] = {self.depot: [initial_label]}

        # Priority queue for processing (process highest reduced cost first)
        unprocessed_labels = [initial_label]

        # Track best routes found (as labels reaching depot)
        completed_routes: List[Label] = []

        while unprocessed_labels:
            # Get label with highest reduced cost
            current_label = max(unprocessed_labels, key=lambda l: l.reduced_cost)
            unprocessed_labels.remove(current_label)

            # Try extending to all successor nodes
            for next_node in range(1, self.n_nodes + 1):
                # Skip if already visited (elementarity)
                if next_node in current_label.visited:
                    continue

                # Create extended label
                new_label = self._extend_label(current_label, next_node)

                if new_label is None:
                    self.labels_infeasible += 1
                    continue

                self.labels_generated += 1

                # Check branching constraints
                if branching_constraints and not self._satisfies_constraints(new_label, branching_constraints):
                    self.labels_infeasible += 1
                    continue

                # Check dominance
                if self._is_dominated(new_label, labels_at_node.get(next_node, [])):
                    self.labels_dominated += 1
                    continue

                # Remove dominated labels
                if next_node in labels_at_node:
                    labels_at_node[next_node] = [
                        lbl for lbl in labels_at_node[next_node] if not new_label.dominates(lbl)
                    ]

                # Add to label set
                if next_node not in labels_at_node:
                    labels_at_node[next_node] = []
                labels_at_node[next_node].append(new_label)
                unprocessed_labels.append(new_label)

            # Try returning to depot (complete route)
            if current_label.node != self.depot:
                final_label = self._extend_to_depot(current_label)
                if final_label is not None and final_label.reduced_cost > 1e-6:
                    completed_routes.append(final_label)

        # Convert completed routes to output format
        routes = []
        for label in completed_routes:
            # Reconstruct path (exclude depot at start and end)
            full_path = label.reconstruct_path()
            route_nodes = [node for node in full_path if node != self.depot]
            reduced_cost = label.reduced_cost

            if reduced_cost > 1e-6:  # Only positive reduced cost
                routes.append((route_nodes, reduced_cost))

        return routes

    def _extend_label(self, label: Label, next_node: int) -> Optional[Label]:
        """
        Extend a label to a successor node.

        Args:
            label: Current label
            next_node: Node to extend to

        Returns:
            New extended label, or None if infeasible
        """
        # Calculate new resource consumption
        node_waste = self.wastes.get(next_node, 0.0)
        new_load = label.load + node_waste

        # Check capacity constraint
        if new_load > self.capacity:
            return None

        # Calculate cost and revenue
        edge_distance = self.cost_matrix[label.node, next_node]
        edge_cost = edge_distance * self.C
        new_cost = label.cost + edge_cost

        node_revenue = node_waste * self.R
        new_revenue = label.revenue + node_revenue

        # Calculate reduced cost contribution
        node_dual = self.dual_values.get(next_node, 0.0)

        # Contribution from capacity cuts (RCC)
        # If both current node and next node are in the cut set S,
        # the edge (label.node, next_node) is internal to S.
        cut_dual_total = 0.0
        for cut_set, _, dual_val in self.capacity_cut_duals:
            if label.node in cut_set and next_node in cut_set:
                # The constraint is Σ a_k^S λ_k <= |S|-k(S), so dual Pi <= 0.
                # Reduced cost = profit - Σ dual_j * a_j - dual_cut * internal_edges
                # Since Pi is non-positive for <= constraint in maximization,
                # we SUBTRACT dual_val * 1.
                cut_dual_total += dual_val

        new_reduced_cost = label.reduced_cost + (node_revenue - edge_cost - node_dual - cut_dual_total)

        # Create new label
        new_path = label.path + [next_node]
        new_visited = label.visited | {next_node}

        new_label = Label(
            reduced_cost=new_reduced_cost,
            node=next_node,
            cost=new_cost,
            load=new_load,
            revenue=new_revenue,
            path=new_path,
            visited=new_visited,
            parent=label,
        )

        return new_label

    def _extend_to_depot(self, label: Label) -> Optional[Label]:
        """
        Extend a label back to depot to complete the route.

        Args:
            label: Current label at customer node

        Returns:
            Final label at depot, or None if infeasible
        """
        # Calculate return cost
        edge_distance = self.cost_matrix[label.node, self.depot]
        edge_cost = edge_distance * self.C

        new_cost = label.cost + edge_cost

        # Check if the return edge (label.node, 0) is in any capacity cuts
        cut_dual_total = 0.0
        for cut_set, _, dual_val in self.capacity_cut_duals:
            if label.node in cut_set and self.depot in cut_set:
                cut_dual_total += dual_val

        new_reduced_cost = label.reduced_cost - edge_cost - cut_dual_total

        # Create final label
        new_path = label.path + [self.depot]

        final_label = Label(
            reduced_cost=new_reduced_cost,
            node=self.depot,
            cost=new_cost,
            load=label.load,
            revenue=label.revenue,
            path=new_path,
            visited=label.visited,
            parent=label,
        )

        return final_label

    def _is_dominated(self, label: Label, existing_labels: List[Label]) -> bool:
        """
        Check if a label is dominated by any existing label at the same node.

        Args:
            label: Label to check
            existing_labels: List of existing labels at the node

        Returns:
            True if label is dominated
        """
        return any(existing.dominates(label) for existing in existing_labels)

    def _satisfies_constraints(self, label: Label, constraints: List) -> bool:
        """
        Check if a label satisfies branching constraints.

        Args:
            label: Label to check
            constraints: List of BranchingConstraint objects

        Returns:
            True if all constraints are satisfied
        """
        for constraint in constraints:
            if not isinstance(constraint, BranchingConstraint):
                continue

            node_r_visited = constraint.node_r in label.visited
            node_s_visited = constraint.node_s in label.visited

            if constraint.together:
                # Both must be visited together or both not visited
                if node_r_visited != node_s_visited:
                    return False
            else:
                # At most one can be visited
                if node_r_visited and node_s_visited:
                    return False

        return True

    def compute_route_details(
        self,
        route: List[int],
    ) -> Tuple[float, float, float, Set[int]]:
        """
        Compute detailed information about a route.

        Args:
            route: List of nodes in route

        Returns:
            Tuple of (cost, revenue, load, node_coverage)
        """
        # Calculate distance
        total_distance = 0.0
        prev = self.depot

        for node in route:
            total_distance += self.cost_matrix[prev, node]
            prev = node

        total_distance += self.cost_matrix[prev, self.depot]

        # Calculate revenue and load
        total_waste = sum(self.wastes.get(node, 0.0) for node in route)
        revenue = total_waste * self.R
        cost = total_distance * self.C

        # Node coverage
        node_coverage = set(route)

        return cost, revenue, total_waste, node_coverage

    def get_statistics(self) -> Dict[str, int]:
        """Get solving statistics."""
        return {
            "labels_generated": self.labels_generated,
            "labels_dominated": self.labels_dominated,
            "labels_infeasible": self.labels_infeasible,
        }
