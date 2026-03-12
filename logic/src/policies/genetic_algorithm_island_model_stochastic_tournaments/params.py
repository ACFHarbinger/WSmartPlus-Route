"""
Configuration parameters for the Island Model Genetic Algorithm with Stochastic Tournaments (IMGA-ST).

Replaces the metaphor-based Volleyball Premier League (VPL) with a mathematically
rigorous multi-population genetic algorithm.

Architecture:
    - Island Model: Multi-population structure with periodic migration.
    - Stochastic Tournament Selection: Pairwise selection with sigmoid probability.
    - Local Improvement: ALNS applied to individuals for intensification.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IslandModelSTGAParams:
    """
    Configuration parameters for IMGA-ST.

    Attributes:
        n_islands: Number of independent sub-populations (islands).
        island_size: Number of individuals per island.
        max_generations: Maximum number of evolution cycles.
        migration_interval: Generations between migration events.
        migration_size: Number of best individuals to migrate per event.
        tournament_size: Number of competitors in stochastic tournament.
        selection_pressure: Beta coefficient for sigmoid win probability.
        crossover_rate: Probability of recombination between parents.
        mutation_rate: Probability of random perturbation.
        alns_iterations: Local improvement (ALNS) iterations per individual.
        elitism_size: Number of best individuals preserved per island.
        time_limit: Overall wall-clock time limit in seconds.
    """

    # Population structure
    n_islands: int = 4
    island_size: int = 10
    max_generations: int = 100

    # Migration (substitution)
    migration_interval: int = 10
    migration_size: int = 2

    # Selection (competition)
    tournament_size: int = 2
    selection_pressure: float = 0.5  # Beta for sigmoid win prob

    # Evolution (coaching/learning)
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    alns_iterations: int = 50
    elitism_size: int = 2

    # Resource constraints
    time_limit: float = 300.0

    def __post_init__(self):
        """Validate parameter constraints."""
        assert self.n_islands > 0, "n_islands must be positive"
        assert self.island_size > 0, "island_size must be positive"
        assert self.migration_size < self.island_size, "migration_size must be less than island_size"
        assert self.selection_pressure >= 0, "selection_pressure must be non-negative"
        assert 0 <= self.crossover_rate <= 1, "crossover_rate must be in [0, 1]"
        assert 0 <= self.mutation_rate <= 1, "mutation_rate must be in [0, 1]"
