"""
VRPP Valid Inequality Definitions.
"""

from typing import List, Set

import numpy as np


class Inequality:
    """Base class for valid inequalities."""

    def __init__(self, inequality_type: str, degree_of_violation: float):
        self.type = inequality_type
        self.violation = degree_of_violation
        self.rhs = 0.0  # Right-hand side of inequality

    def __lt__(self, other):
        """Sort by violation (descending)."""
        return self.violation > other.violation


class PCSubtourEliminationCut(Inequality):
    """
    Prize-Collecting Subtour Elimination Constraint (PC-SEC).
    """

    def __init__(
        self, node_set: Set[int], violation: float, facet_form: str = "2.1", node_i: int = -1, node_j: int = -1
    ):
        super().__init__("SEC", violation)
        self.node_set = node_set
        self.facet_form = facet_form
        self.node_i = node_i
        self.node_j = node_j
        self.local_only = False
        self.rhs = 2.0


class CapacityCut(Inequality):
    """
    Rounded Capacity Cut (RCC).
    """

    def __init__(self, node_set: Set[int], total_demand: float, capacity: float, violation: float):
        super().__init__("CAPACITY", violation)
        self.node_set = node_set
        self.total_demand = total_demand
        self.min_vehicles = int(np.ceil(total_demand / capacity))
        self.rhs = 2.0 * self.min_vehicles


class CombInequality(Inequality):
    """
    Comb Inequality for TSP/VRP.
    """

    def __init__(self, handle: Set[int], teeth: List[Set[int]], violation: float):
        super().__init__("COMB", violation)
        self.handle = handle
        self.teeth = teeth
        self.rhs = len(handle) + sum(len(t) for t in teeth) - (len(teeth) + 1) / 2.0
