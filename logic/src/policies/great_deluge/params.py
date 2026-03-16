"""
Great Deluge (GD) parameters.
"""

from dataclasses import dataclass


@dataclass
class GDParams:
    """
    Configuration for the Great Deluge (GD) solver.

    Attributes:
        max_iterations: Total LLH applications.
        target_fitness_multiplier: Initial target = initial_profit * multiplier.
        time_limit: Wall-clock time limit in seconds.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
    """

    max_iterations: int = 1000
    target_fitness_multiplier: float = 1.1
    time_limit: float = 60.0
    n_removal: int = 2
    n_llh: int = 5
