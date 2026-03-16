"""
Only Improving (OI) parameters.
"""

from dataclasses import dataclass


@dataclass
class OIParams:
    """
    Configuration for the Only Improving (OI) solver.

    Attributes:
        max_iterations: Total LLH applications.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        time_limit: Wall-clock time limit in seconds.
    """

    max_iterations: int = 1000
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
