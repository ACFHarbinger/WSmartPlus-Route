"""
Operator enum for WSmart-Route.

Attributes:
    OperatorTag: Enum for operator tags

Example:
    >>> from logic.src.enums import OperatorTag
    >>> OperatorTag.DESTRUCTIVE
    <OperatorTag.DESTRUCTIVE: 1>
"""

from enum import Enum, auto


class OperatorTag(Enum):
    """
    Operator tags for WSmart-Route.

    Attributes:
        DESTRUCTIVE: Ruin (e.g., Random, Shaw, Worst)
        CONSTRUCTIVE: Repair/Recreate (e.g., Greedy, Regret-K)
        IMPROVEMENT: Local Search (e.g., 2-Opt, Relocate)
        RECOMBINATION: Crossover (e.g., OX, Edge Recombination)
        INTRA_ROUTE: Operates within a single route
        INTER_ROUTE: Swaps between different routes
        O_N: Linear time
        O_N2: Quadratic (standard for most edge-exchanges)
        O_N3: Cubic (e.g., full 3-Opt)
        HEURISTIC: Fast approximation
        EXACT_SUBPROBLEM: Solves a sub-graph optimally (DP or Set Partitioning)
    """

    # Topology Scope
    INTRA_ROUTE = auto()  # Operates within a single route
    INTER_ROUTE = auto()  # Swaps between different routes

    # Computational Complexity
    O_N = auto()  # Linear time
    O_N2 = auto()  # Quadratic (standard for most edge-exchanges)
    O_N3 = auto()  # Cubic (e.g., full 3-Opt)

    # Mathematical Rigor
    HEURISTIC = auto()  # Fast approximation
    EXACT_SUBPROBLEM = auto()  # Solves a sub-graph optimally (DP or Set Partitioning)
