"""
Vectorized Local Search Operators for Vehicle Routing Problems.

This module implements several local search heuristics optimized for parallel execution
on GPU using PyTorch. These operators are used to improve routing solutions in both
genetic algorithms (like HGS) and as post-processing steps for neural models.

Implemented Operators:
- vectorized_two_opt: Intra-route segment reversal.
- vectorized_swap: Intra-route node exchange.
- vectorized_relocate: Intra-route node relocation.
- vectorized_two_opt_star: Inter-route tail swap.
- vectorized_swap_star: Inter-route node exchange with re-insertion optimization.
- vectorized_three_opt: Intra-route 3-opt moves.
"""

from logic.src.models.policies.operators.relocate import vectorized_relocate
from logic.src.models.policies.operators.swap import vectorized_swap
from logic.src.models.policies.operators.swap_star import vectorized_swap_star
from logic.src.models.policies.operators.three_opt import vectorized_three_opt
from logic.src.models.policies.operators.two_opt import vectorized_two_opt
from logic.src.models.policies.operators.two_opt_star import vectorized_two_opt_star

__all__ = [
    "vectorized_two_opt",
    "vectorized_swap",
    "vectorized_relocate",
    "vectorized_two_opt_star",
    "vectorized_swap_star",
    "vectorized_three_opt",
]
