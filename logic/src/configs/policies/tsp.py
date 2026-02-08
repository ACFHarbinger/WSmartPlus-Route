"""
TSP (Traveling Salesman Problem) policy configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class TSPConfig:
    """Configuration for Traveling Salesman Problem (TSP) policy.

    Attributes:
        cache: Whether to cache solutions.
        time_limit: Maximum time in seconds for the solver.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    cache: bool = True
    time_limit: float = 60.0
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
