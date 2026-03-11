"""
Repair Operators Package.

This package contains operator implementations for repairing (re-inserting nodes into)
partial solutions. Includes greedy, regret-based, savings-based, and deep heuristics.

Example:
    >>> from logic.src.policies.other.operators.repair import greedy_insertion
    >>> routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from ..unstringing_stringing import apply_type_i_s, apply_type_ii_s, apply_type_iii_s, apply_type_iv_s
from .deep import deep_insertion
from .greedy import greedy_insertion
from .greedy_blink import greedy_insertion_with_blinks
from .regret import regret_2_insertion, regret_k_insertion
from .savings import savings_insertion

__all__ = [
    "greedy_insertion",
    "regret_2_insertion",
    "regret_k_insertion",
    "greedy_insertion_with_blinks",
    "savings_insertion",
    "deep_insertion",
    # Stringing repair
    "apply_type_i_s",
    "apply_type_ii_s",
    "apply_type_iii_s",
    "apply_type_iv_s",
]
