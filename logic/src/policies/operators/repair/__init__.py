from .greedy import greedy_insertion
from .greedy_blink import greedy_insertion_with_blinks
from .regret import regret_2_insertion, regret_k_insertion

__all__ = [
    "greedy_insertion",
    "regret_2_insertion",
    "regret_k_insertion",
    "greedy_insertion_with_blinks",
]
