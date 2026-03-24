"""
Repair Operators Package.

This package contains operator implementations for repairing (re-inserting nodes into)
partial solutions. Includes greedy, regret-based, savings-based, and deep heuristics.

Example:
    >>> from logic.src.policies.other.operators.repair import greedy_insertion
    >>> routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from .deep import deep_insertion, deep_profit_insertion
from .farthest import farthest_insertion, farthest_profit_insertion
from .geni import geni_insertion, geni_profit_insertion
from .greedy import greedy_insertion, greedy_profit_insertion
from .greedy_blink import greedy_insertion_with_blinks, greedy_profit_insertion_with_blinks
from .nearest import nearest_insertion, nearest_profit_insertion
from .regret import regret_2_insertion, regret_2_profit_insertion, regret_k_insertion, regret_k_profit_insertion
from .savings import savings_insertion, savings_profit_insertion

__all__ = [
    "greedy_insertion",
    "greedy_profit_insertion",
    "farthest_insertion",
    "farthest_profit_insertion",
    "geni_insertion",
    "geni_profit_insertion",
    "regret_2_insertion",
    "regret_2_profit_insertion",
    "regret_k_insertion",
    "regret_k_profit_insertion",
    "greedy_insertion_with_blinks",
    "greedy_profit_insertion_with_blinks",
    "nearest_insertion",
    "nearest_profit_insertion",
    "savings_insertion",
    "savings_profit_insertion",
    "deep_insertion",
    "deep_profit_insertion",
    # Stringing repair
    "apply_type_i_s",
    "apply_type_ii_s",
    "apply_type_iii_s",
    "apply_type_iv_s",
    "apply_type_i_s_profit",
    "apply_type_ii_s_profit",
    "apply_type_iii_s_profit",
    "apply_type_iv_s_profit",
    "stringing_insertion",
    "stringing_profit_insertion",
]
