"""
Configuration parameters for Pure Island Model Genetic Algorithm.

This replaces "Soccer League Competition (SLC)" with standard GA terminology.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PureIslandModelGAParams:
    """
    Parameters for Pure Island Model Genetic Algorithm (without local search).

    Multi-population GA where isolated sub-populations evolve independently
    with periodic migration. Uses only genetic operators (crossover, mutation)
    without local search refinement.

    Replaces "Soccer League Competition" sports metaphors:
    - "Soccer Teams" → Sub-populations (islands)
    - "Players" → Chromosomes/decision variables
    - "Matches" → Fitness evaluations
    - "Player Transfers" → Migration operators
    - "Relegation" → Worst-solution replacement

    Algorithm Structure:
        1. Initialize K islands with N chromosomes each
        2. For each generation:
            a. Selection: Tournament selection within each island
            b. Crossover: Recombine selected parents
            c. Mutation: Perturb offspring
            d. Replacement: Replace worst individuals
            e. Migration: Periodically exchange best solutions between islands

    Attributes:
        n_islands: Number of independent sub-populations (K).
        island_size: Population size per island (N).
        max_generations: Number of evolution cycles.
        crossover_rate: Probability of applying crossover operator.
        mutation_rate: Probability of applying mutation operator.
        tournament_size: Number of individuals in tournament selection.
        elitism_count: Top individuals preserved per island.
        migration_interval: Generations between inter-island migration.
        migration_size: Number of individuals migrated per interval.
        time_limit: Wall-clock time limit in seconds (0 = no limit).

    Complexity:
        - Space: O(K × N × n) for island populations
        - Time per generation: O(K × N × n²) for genetic operators

    Reference:
        Whitley, D., Rana, S., & Heckendorn, R. B. (1998).
        "The island model genetic algorithm: On separability, population size and convergence."
    """

    n_islands: int = 10  # K parameter
    island_size: int = 20  # N parameter
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elitism_count: int = 2
    migration_interval: int = 10
    migration_size: int = 2
    time_limit: float = 60.0
