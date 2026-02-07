"""
Destroy operators for the Adaptive Large Neighborhood Search (ALNS).

This module contains various removal heuristics used to discard nodes
from the current routing solution to explore the search space.

(Refactored to point to `logic.src.policies.operators.destroy` package)
"""

from .destroy import (
    cluster_removal,
    random_removal,
    shaw_removal,
    string_removal,
    worst_removal,
)

__all__ = [
    "random_removal",
    "worst_removal",
    "cluster_removal",
    "shaw_removal",
    "string_removal",
]
