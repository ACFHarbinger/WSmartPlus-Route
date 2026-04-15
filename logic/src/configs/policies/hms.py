"""
Memetic Island Model GA configuration.

Replaces HVPL with rigorous GA terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class HybridMemeticSearchConfig:
    """Configuration for Hybrid Memetic Search (HMS) policy.

    Functionally equivalent to HVPL.
    """

    population_size: int = 30
    max_generations: int = 50
    substitution_rate: float = 0.2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_count: int = 3
    aco_init_iterations: int = 50
    time_limit: float = 300.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
