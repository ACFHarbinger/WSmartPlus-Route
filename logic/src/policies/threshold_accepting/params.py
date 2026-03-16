"""
Threshold Accepting (TA) parameters.
"""

from dataclasses import dataclass


@dataclass
class TAParams:
    """
    Configuration for the Threshold Accepting (TA) solver.

    Attributes:
        max_iterations: Total LLH applications.
        initial_threshold: Starting absolute tolerance.
        time_limit: Wall-clock time limit in seconds.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
    """

    max_iterations: int = 1000
    initial_threshold: float = 100.0
    time_limit: float = 60.0
    n_removal: int = 2
    n_llh: int = 5
