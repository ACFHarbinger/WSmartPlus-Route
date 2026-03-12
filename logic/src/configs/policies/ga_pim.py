"""
Pure Island Model GA configuration.

Replaces SLC with rigorous GA terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class PureIslandModelGAConfig:
    """Configuration for Pure Island Model GA policy.

    Multi-population GA with genetic operators only (no local search).
    """

    n_islands: int = 10
    island_size: int = 20
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elitism_count: int = 2
    migration_interval: int = 10
    migration_size: int = 2
    time_limit: float = 60.0
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
