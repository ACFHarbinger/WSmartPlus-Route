"""
Hybrid Memetic Search (HMS) for VRPP.

Rigorous implementation replacing "Hybrid Volleyball Premier League (HVPL)".
Multi-phase hybrid solver combining ACO, GA, and ALNS.

Attributes:
    HybridMemeticSearchParams: Parameters for HMS solver.
    HybridMemeticSearchSolver: Core HMS solver logic.

Example:
    >>> from logic.src.policies.route_construction.meta_heuristics.hybrid_memetic_search import HybridMemeticSearchSolver
"""

from .params import HybridMemeticSearchParams
from .solver import HybridMemeticSearchSolver

__all__ = ["HybridMemeticSearchSolver", "HybridMemeticSearchParams"]
