"""
BPC (Branch-and-Price-and-Cut) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class BPCConfig:
    """Configuration for Branch-and-Price-and-Cut (BPC) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        engine: Solver engine to use ('ortools', 'gurobi').
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    time_limit: float = 60.0
    engine: str = "ortools"
    vrpp: bool = True
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
