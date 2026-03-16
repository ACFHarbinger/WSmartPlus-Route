"""
State-space search tree node for Branch-and-Bound.

Each node in the tree represents a subproblem where a subset of binary decision
variables (edges and node visits) has been fixed to specific integer values.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(order=True)
class Node:
    """Represents a discrete node in the Branch-and-Bound exploration tree.

    Nodes are prioritized in the search queue based on their lower bound (LB)
    value, implementing a best-bound-first search strategy to minimize tree size.

    Attributes:
        bound (float): The objective value of the Linear Programming relaxation
            at this node. Serves as a deterministic lower bound for the subproblem.
        fixed_x (Dict[Tuple[int, int], int]): A dictionary mapping edge tuples (i, j)
            to their fixed binary values (0 or 1).
        fixed_y (Dict[int, int]): A dictionary mapping customer IDs to their
            fixed binary visit statuses (0 or 1).
        depth (int): The distance of the node from the root of the search tree.
    """

    bound: float = field(compare=True)  # LP relaxation value (lower bound for minimization)
    fixed_x: Dict[Tuple[int, int], int] = field(compare=False, default_factory=dict)
    fixed_y: Dict[int, int] = field(compare=False, default_factory=dict)
    depth: int = field(compare=False, default=0)

    def __post_init__(self):
        """Standardization of floating point precision for the bound if necessary."""
        pass
