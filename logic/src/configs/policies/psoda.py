"""
Distance-Based PSO configuration.

Replaces Firefly Algorithm with rigorous PSO terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class DistancePSOConfig:
    """Configuration for Distance-Based PSO policy.

    Standard PSO with distance-dependent attraction weights.
    """

    population_size: int = 20
    initial_attraction: float = 1.0
    distance_decay: float = 0.01
    exploration_rate: float = 0.1
    n_removal: int = 3
    max_iterations: int = 500
    local_search_iterations: int = 500
    time_limit: float = 60.0
    alpha_profit: float = 1.0
    beta_will: float = 0.5
    gamma_cost: float = 0.3
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
