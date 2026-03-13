"""
Stochastic Tournament GA configuration.

Replaces LCA with rigorous GA terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class MemeticAlgorithmToleranceBasedConfig:
    """Configuration for Memetic Algorithm Tolerance Based (MA-TB) policy.

    GA with sigmoid-based pairwise tournament selection.
    """

    population_size: int = 10
    max_iterations: int = 100
    tolerance_pct: float = 0.05
    recombination_rate: float = 0.6
    perturbation_strength: int = 2
    time_limit: float = 60.0
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
