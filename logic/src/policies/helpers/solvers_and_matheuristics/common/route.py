"""Route data structure for Branch-and-Price-and-Cut VRPP.

Attributes:
    Route: Data structure representing a vehicle route (column) in the BPC process.

Example:
    >>> route = Route(nodes=[1, 2], cost=10.0, revenue=20.0, load=5.0, node_coverage={1, 2})
"""

from typing import List, Optional, Set


class Route:
    """Represents a single route (column) in the master problem.

    Attributes:
        nodes (List[int]): Sequence of customer nodes visited.
        cost (float): Total distance cost of the route.
        revenue (float): Total revenue from collected waste.
        load (float): Total waste collected on the route.
        profit (float): Net profit (revenue - cost).
        node_coverage (Set[int]): Set of customer node IDs covered.
        reduced_cost (Optional[float]): Reduced cost relative to duals.

    Example:
        >>> r = Route([1, 2], 10.0, 15.0, 5.0, {1, 2})
    """

    def __init__(
        self,
        nodes: List[int],
        cost: float,
        revenue: float,
        load: float,
        node_coverage: Set[int],
    ) -> None:
        """
        Initialise a route.

        Args:
            nodes: Sequence of customer nodes visited (excluding depot returns).
            cost: Total distance cost of the route.
            revenue: Total revenue from collected waste.
            load: Total waste collected on the route.
            node_coverage: Set of customer node IDs covered by this route.
        """
        self.nodes = nodes
        self.cost = cost
        self.revenue = revenue
        self.load = load
        self.profit = revenue - cost
        self.node_coverage = node_coverage
        self.reduced_cost: Optional[float] = None

    def __repr__(self) -> str:
        """Returns string representation of the route.

        Returns:
            str: String representation including nodes and profit.
        """

        return f"Route(nodes={self.nodes}, profit={self.profit:.2f})"
