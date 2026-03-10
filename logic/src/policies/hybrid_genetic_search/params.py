"""
Configuration parameters for Hybrid Genetic Search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.src.configs.policies import HGSConfig


@dataclass
class HGSParams:
    """
    Configuration parameters for Hybrid Genetic Search.

    Attributes:
        time_limit: Maximum search time in seconds.
        population_size: Target population size.
        elite_size: Number of elite individuals for survivor selection.
        mutation_rate: Probability of applying local search improvement.
        max_vehicles: Maximum number of vehicles allowed (0 = unlimited).
        n_generations: Number of generations to run the algorithm.
        alpha_diversity: Weight for diversity in fitness evaluation.
        min_diversity: Minimum diversity threshold for triggering diversity maintenance.
        diversity_change_rate: Rate at which alpha diversity changes.
        min_diversity_threshold: Threshold for minimum diversity.
        survivor_threshold: Threshold for survivor selection.
        no_improvement_threshold: Number of generations without improvement to trigger diversity maintenance.
        neighbor_list_size: Number of nearest neighbors to consider.
        local_search_iterations: Number of iterations to run local search.
        max_vehicles: Maximum number of vehicles allowed (0 = unlimited).
    """

    time_limit: float = 60.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    n_generations: int = 100
    alpha_diversity: float = 0.5
    min_diversity: float = 0.2
    diversity_change_rate: float = 0.05
    survivor_threshold: int = 2
    no_improvement_threshold: int = 20
    min_diversity_threshold: float = 0.2
    neighbor_list_size: int = 15
    local_search_iterations: int = 100
    max_vehicles: int = 0

    @classmethod
    def from_config(cls, config: HGSConfig) -> HGSParams:
        """Create HGSParams from a HGSConfig dataclass.

        Args:
            config: HGSConfig dataclass with solver parameters.

        Returns:
            HGSParams instance with values from config.
        """
        return cls(
            time_limit=config.time_limit,
            population_size=config.population_size,
            elite_size=config.elite_size,
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            n_generations=config.n_generations,
            max_vehicles=config.max_vehicles,
            alpha_diversity=config.alpha_diversity,
            min_diversity=config.min_diversity,
            neighbor_list_size=config.neighbor_list_size,
            local_search_iterations=config.local_search_iterations,
        )
