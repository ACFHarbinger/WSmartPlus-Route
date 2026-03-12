"""
Stochastic Tournament GA configuration.

Replaces LCA with rigorous GA terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class StochasticTournamentGAConfig:
    """Configuration for Stochastic Tournament GA policy.

    GA with sigmoid-based pairwise tournament selection.
    """

    population_size: int = 50
    tournament_competitors: int = 5
    selection_pressure: float = 0.1
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elitism_count: int = 2
    max_generations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
