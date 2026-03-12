"""
Memetic Island Model GA configuration.

Replaces HVPL with rigorous GA terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class MemeticIslandModelGAConfig:
    """Configuration for Memetic Island Model GA policy.

    Multi-population GA with ALNS local search (memetic component).
    """

    n_islands: int = 10
    island_size: int = 10
    max_generations: int = 50
    time_limit: float = 60.0
    replacement_rate: float = 0.2
    tournament_size: int = 3
    migration_interval: int = 5
    migration_size: int = 1
    aco_n_ants: int = 10
    aco_k_sparse: int = 10
    alns_max_iterations: int = 100
    alns_start_temp: float = 100.0
    alns_cooling_rate: float = 0.95
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
