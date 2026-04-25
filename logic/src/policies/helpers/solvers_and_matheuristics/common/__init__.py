"""Shared types and data structures for Branch-and-Price-and-Cut.

This module provides common data structures like BranchNode and Route that are used
across the various components of the exact solver.

Attributes:
    BranchNode: Represents a node in the Branch-and-Bound search tree.
    Route: Represents a single vehicle route with cost and profit information.

Example:
    >>> node = BranchNode(depth=1)
    >>> route = Route(nodes=[1, 2, 3], cost=10.0, profit=5.0)
"""

from .node import BranchNode
from .route import Route

__all__ = ["BranchNode", "Route"]
