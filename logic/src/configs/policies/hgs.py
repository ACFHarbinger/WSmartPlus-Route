"""
HGS (Hybrid Genetic Search) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..other.must_go import MustGoConfig
from ..other.post_processing import PostProcessingConfig


@dataclass
class HGSConfig:
    """Configuration for Hybrid Genetic Search (HGS) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        population_size: Size of the genetic population.
        elite_size: Number of elite individuals to preserve.
        mutation_rate: Probability of mutation.
        n_generations: Number of generations to evolve.
        max_vehicles: Maximum number of vehicles (0 for unlimited).
        engine: Solver engine to use ('custom', 'pyvrp').
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    time_limit: float = 60.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.2
    n_generations: int = 100
    max_vehicles: int = 0
    engine: str = "custom"
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
