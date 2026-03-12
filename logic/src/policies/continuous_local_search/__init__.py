"""
Continuous Local Search with Trigonometric Perturbations for VRPP.

Rigorous implementation replacing metaphor-based "Sine Cosine Algorithm".
"""

from .params import ContinuousLocalSearchParams
from .solver import ContinuousLocalSearchSolver

__all__ = ["ContinuousLocalSearchSolver", "ContinuousLocalSearchParams"]
