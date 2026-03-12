"""
(μ,λ) Evolution Strategy configuration.

Replaces Artificial Bee Colony with rigorous ES terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class MuCommaLambdaESConfig:
    """Configuration for (μ,λ) Evolution Strategy policy.

    Multi-phase ES with random restart mechanism.
    """

    population_size: int = 20
    offspring_per_parent: int = 1
    n_removal: int = 3
    stagnation_limit: int = 10
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
