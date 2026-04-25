"""Lagrangian Relaxation shared infrastructure for VRPP exact solvers.

Provides the subgradient optimisation loop and the uncapacitated Orienteering
Problem (UOP) inner solver used by both the Branch-and-Bound (LR-UOP formulation)
and the Branch-and-Price-and-Cut pre-pruning integration.

Attributes:
    run_subgradient: Polyak-step subgradient loop for minimizing dual functions.
    solve_uncapacitated_op: Exact Gurobi solver for the uncapacitated OP at a fixed λ.
    _nearest_neighbour_tour_cost: Greedy tour evaluator for lower bounds.

Example:
    >>> lam, ub, lb, hist = run_subgradient(env, initial_lam=0.5)
"""

from .subgradient_optimization import _nearest_neighbour_tour_cost, run_subgradient
from .uncapacitated_orienteering_problem import solve_uncapacitated_op

__all__ = [
    "run_subgradient",
    "solve_uncapacitated_op",
    "_nearest_neighbour_tour_cost",
]
