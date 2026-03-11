"""
Repair Operators Package.

This package contains operator implementations for repairing (re-inserting nodes into)
partial solutions. Includes greedy and regret-based heuristics.

Attributes:
    greedy_insertion (function): Best single-move insertion.
    greedy_insertion_with_blinks (function): Randomized greedy insertion.
    regret_2_insertion (function): Insertion maximizing difference between best and 2nd best.
    regret_k_insertion (function): General k-regret insertion.

Example:
    >>> from logic.src.policies.other.operators.repair import greedy_insertion
    >>> new_routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from ..unstringing_stringing import apply_type_i_s, apply_type_ii_s, apply_type_iii_s, apply_type_iv_s
from .greedy import greedy_insertion
from .greedy_blink import greedy_insertion_with_blinks
from .regret import regret_2_insertion, regret_k_insertion

__all__ = [
    "greedy_insertion",
    "regret_2_insertion",
    "regret_k_insertion",
    "greedy_insertion_with_blinks",
    # Stringing repair
    "apply_type_i_s",
    "apply_type_ii_s",
    "apply_type_iii_s",
    "apply_type_iv_s",
]
