"""
Island Model Genetic Algorithm with Stochastic Tournaments (IMGA-ST) Configuration.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class IslandModelSTGAConfig:
    """
    Configuration for the Island Model STGA policy.
    """

    engine: str = "ga_im_st"

    # IMGA Parameters
    n_islands: int = 4
    island_size: int = 10
    max_generations: int = 100

    # Migration
    migration_interval: int = 10
    migration_size: int = 2

    # Selection
    tournament_size: int = 2
    selection_pressure: float = 0.5

    # Evolution
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    alns_iterations: int = 50
    elitism_size: int = 2

    # Global Parameters
    time_limit: float = 300.0
    seed: Optional[int] = None

    # Common policy fields
    vrpp: bool = True
    must_go: List[str] = field(default_factory=list)
    post_processing: List[Any] = field(default_factory=list)
