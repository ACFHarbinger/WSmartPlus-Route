"""
Separation algorithms for cutting planes in Branch-and-Price-and-Cut.

Finds violated inequalities (RCC, SEC, SRI, LCI) from fractional LP solutions.
These inequalities strengthen the Lower Bound (z_LP) by cutting off fractional
points that cannot be represented as convex combinations of integer feasible
routes.

Reference:
    - Lysgaard, Letchford, and Eglese (2004) for RCC/SEC.
    - Jepsen et al. (2008) for Subset-Row Inequalities.
"""

from logic.src.policies.other.branching_solvers.separation.engine import SeparationEngine
from logic.src.policies.other.branching_solvers.separation.inequality import (
    CapacityCut,
    CombInequality,
    Inequality,
    PCSubtourEliminationCut,
)

__all__ = [
    "SeparationEngine",
    "Inequality",
    "PCSubtourEliminationCut",
    "CapacityCut",
    "CombInequality",
]
