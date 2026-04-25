r"""Augmented Hybrid Volleyball Premier League (AHVPL) matheuristic package.

This module implements the AHVPL algorithm, a population-based meta-heuristic
inspired by the competition structure of volleyball leagues, augmented with
local search for solving the VRPP.

Attributes:
    AHVPLPolicy: Adapter class for the AHVPL solver.
    AHVPLSolver: The main solver implementation.
    AHVPLParams: Parameter configuration for AHVPL.

Example:
    >>> from logic.src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league import AHVPLPolicy
    >>> policy = AHVPLPolicy()
"""

from .policy_ahvpl import AHVPLPolicy

__all__ = ["AHVPLPolicy"]
