"""
Init documentation for Population-based Hyper-Heuristic.

Exposes the PHHPolicy class for use in route construction.

Attributes:
    PHHPolicy: Adapter for the PHH solver.

Example:
    >>> from logic.src.policies.route_construction.hyper_heuristics import PHHPolicy
    >>> policy = PHHPolicy()
"""

from .policy_phh import PHHPolicy

__all__ = ["PHHPolicy"]
