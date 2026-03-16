"""
Improving and Equal (IE) parameters.
"""

from dataclasses import dataclass


@dataclass
class IEParams:
    """
    Configuration for the Improving and Equal (IE) solver.
    """

    max_iterations: int = 1000
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
