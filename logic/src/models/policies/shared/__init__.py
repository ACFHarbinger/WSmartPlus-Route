"""Shared policy utilities package.

This package provides shared algorithmic components used across different
routing policies, including split algorithms for transforming giant tours
into multiple routes and utilities for route reconstruction.

Attributes:
    vectorized_linear_split: Linear time split algorithm.
    _vectorized_split_limited: Limited capacity split algorithm.
    _reconstruct_routes: Reconstructs tour sequences from split indices.
    _reconstruct_limited: Reconstruction utility for limited-capacity scenarios.

Example:
    None
"""

from __future__ import annotations

from .limited import vectorized_split_limited
from .linear import vectorized_linear_split
from .reconstruction import reconstruct_limited, reconstruct_routes

# Aliases for backward compatibility
_vectorized_split_limited = vectorized_split_limited
_reconstruct_routes = reconstruct_routes
_reconstruct_limited = reconstruct_limited

__all__ = [
    "vectorized_linear_split",
    "_vectorized_split_limited",
    "_reconstruct_routes",
    "_reconstruct_limited",
]
