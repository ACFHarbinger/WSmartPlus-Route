"""
(μ+1) Evolution Strategy configuration.

Replaces Harmony Search with rigorous ES terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class MuPlusLambdaESConfig:
    """Configuration for (μ+λ) Evolution Strategy policy.

    Canonical evolution strategy with recombination and mutation.

    Attributes:
        population_size: Number of parent solutions (μ parameter).
        offspring_size: Number of offspring generated per cycle (λ parameter).
        recombination_rate: Probability of archive recombination (vs random).
        mutation_rate: Probability of local mutation after recombination.
        max_iterations: Maximum number of evolution cycles.
        local_search_iterations: Number of local search improvement steps.
        time_limit: Wall-clock time limit in seconds (0 = no limit).
        seed: Random seed for reproducibility.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    population_size: int = 10
    offspring_size: int = 5
    recombination_rate: float = 0.95
    mutation_rate: float = 0.3
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
