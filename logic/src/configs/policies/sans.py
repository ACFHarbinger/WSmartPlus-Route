"""
SANS (Simulated Annealing Neighborhood Search) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SANSConfig:
    """Configuration for Simulated Annealing Neighborhood Search (SANS) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        perc_bins_can_overflow: Percentage of bins allowed to overflow.
        T_min: Minimum temperature for simulated annealing.
        T_init: Initial temperature for simulated annealing.
        iterations_per_T: Number of iterations at each temperature level.
        alpha: Cooling rate (temperature multiplier).
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    time_limit: float = 60.0
    perc_bins_can_overflow: float = 0.0
    T_min: float = 0.01
    T_init: float = 75.0
    iterations_per_T: int = 5000
    alpha: float = 0.95
    must_go: Optional[List[str]] = None
    post_processing: Optional[List[str]] = None
