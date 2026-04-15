"""
Pure Island Model GA configuration.

Replaces SLC with rigorous GA terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class MemeticAlgorithmIslandModelConfig:
    """Configuration for Memetic Algorithm Island Model (MA-IM) policy.

    Functionally equivalent to SLC.
    """

    n_islands: int = 5
    island_size: int = 4
    max_generations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
