"""
Continuous Local Search configuration.

Replaces Sine Cosine Algorithm with rigorous terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class ContinuousLocalSearchConfig:
    """Configuration for Continuous Local Search (CLS) policy.

    Gradient-free search with trigonometric perturbations.
    """

    population_size: int = 30
    max_step_size: float = 2.0
    max_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
