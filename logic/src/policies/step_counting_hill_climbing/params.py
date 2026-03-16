"""
Step Counting Hill Climbing (SCHC) parameters.
"""

from dataclasses import dataclass


@dataclass
class SCHCParams:
    """
    Configuration for the Step Counting Hill Climbing (SCHC) solver.

    Attributes:
        max_iterations: Total LLH applications.
        step_size: Number of steps before updating the threshold.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        time_limit: Wall-clock time limit in seconds.
    """

    max_iterations: int = 1000
    step_size: int = 100
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
