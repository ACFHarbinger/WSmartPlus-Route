"""
Vectorized Split Algorithms for Vehicle Routing Problems.

This module is now a facade for `logic.src.models.policies.classical.shared.split`.
"""

from logic.src.models.policies.classical.shared.split import (
    _reconstruct_limited,
    _reconstruct_routes,
    _vectorized_split_limited,
    vectorized_linear_split,
)

__all__ = [
    "vectorized_linear_split",
    "_vectorized_split_limited",
    "_reconstruct_routes",
    "_reconstruct_limited",
]
