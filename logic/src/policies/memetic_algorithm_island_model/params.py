"""
Configuration parameters for Memetic Island Model Genetic Algorithm.

This matches the "Soccer League Competition (SLC)" algorithm structure
but uses rigorous Genetic Algorithm terminology.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemeticAlgorithmIslandModelParams:
    """
    Parameters for Memetic Algorithm with Island Model (MA-IM).

    Matches the Soccer League Competition (SLC) logic:
    - n_islands (n_teams): Number of sub-populations.
    - island_size (team_size): Individuals per sub-population.
    - max_generations (max_iterations): Evolution cycles.
    - n_removal: Nodes removed per mutation/perturbation step.
    - stagnation_limit: Generations without improvement before island regeneration.
    - local_search_iterations: Number of local search iterations.
    - time_limit: Wall-clock time limit in seconds.

    Reference:
        Moosavian, N., & Rppdsarou, B. K. (2014).
        "Soccer league competition algorithm: A novel meta-heuristic
        algorithm for optimal design of water distribution networks."
        Whitley, D., Rana, S., & Heckendorn, R. B. (1998).
        "The island model genetic algorithm: On separability, population size and convergence."
    """

    n_islands: int = 5  # K islands (analogous to teams)
    island_size: int = 4  # N individuals per island
    max_generations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    local_search_iterations: int = 100
    time_limit: float = 60.0
