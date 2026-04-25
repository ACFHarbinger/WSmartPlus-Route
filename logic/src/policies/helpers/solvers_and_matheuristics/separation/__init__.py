"""Separation algorithms for cutting planes in Branch-and-Price-and-Cut.

Finds violated inequalities (RCC, SEC, SRI, LCI) from fractional LP solutions.
These inequalities strengthen the Lower Bound (z_LP) by cutting off fractional
points that cannot be represented as convex combinations of integer feasible
routes.

Attributes:
    SeparationEngine: Core engine for identifying violated inequalities.
    Inequality: Base class for valid inequalities.
    PCSubtourEliminationCut: Subtour elimination constraint for profitable VRP.
    CapacityCut: Rounded Capacity Cut (RCC) representation.
    CombInequality: Comb inequality representation.

Example:
    >>> engine = SeparationEngine(n_nodes=50, capacity=100)
    >>> cuts = engine.separate(x_star, y_star)
"""

from __future__ import annotations

from logic.src.policies.helpers.solvers_and_matheuristics.separation.engine import SeparationEngine
from logic.src.policies.helpers.solvers_and_matheuristics.separation.inequality import (
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
