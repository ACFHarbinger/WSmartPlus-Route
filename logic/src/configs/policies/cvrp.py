"""
CVRP (Capacitated Vehicle Routing Problem) policy configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..other.must_go import MustGoConfig
from ..other.post_processing import PostProcessingConfig


@dataclass
class CVRPConfig:
    """Configuration for Capacitated Vehicle Routing Problem (CVRP) policy.

    Attributes:
        cache: Whether to cache solutions.
        time_limit: Maximum time in seconds for the solver.
        engine: Solver engine to use ('ortools', 'gurobi').
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    cache: bool = False
    time_limit: float = 60.0
    engine: str = "ortools"
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
