"""
LKH (Lin-Kernighan-Helsgaun) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class LKHConfig:
    """Configuration for Lin-Kernighan-Helsgaun Heuristic (LKH) policy.

    Attributes:
        check_capacity: Whether to check vehicle capacity constraints.
        max_iterations: Maximum number of LKH iterations.
        engine: Solver engine to use.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    check_capacity: bool = True
    max_iterations: int = 100
    engine: str = "custom"
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
