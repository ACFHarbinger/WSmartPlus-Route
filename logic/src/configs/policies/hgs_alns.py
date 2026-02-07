"""
HGS-ALNS (Hybrid Genetic Search with ALNS Education) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HGSALNSConfig:
    """Configuration for Hybrid Genetic Search with Adaptive Large Neighborhood Search Education (HGS-ALNS) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        population_size: Size of the genetic population.
        elite_size: Number of elite individuals to preserve.
        mutation_rate: Probability of mutation.
        alns_education_iterations: Number of ALNS iterations for education phase.
        n_generations: Number of generations to evolve.
        max_vehicles: Maximum number of vehicles (0 for unlimited).
        engine: Solver engine to use.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    time_limit: float = 60.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.2
    alns_education_iterations: int = 50
    n_generations: int = 100
    max_vehicles: int = 0
    engine: str = "hgs_alns"
    must_go: Optional[List[str]] = None
    post_processing: Optional[List[str]] = None
