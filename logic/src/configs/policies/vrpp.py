"""
VRPP (Vehicle Routing Problem with Profits) policy configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class VRPPConfig:
    """Configuration for Vehicle Routing Problem with Profits (VRPP) policy.

    Attributes:
        Omega: Profit weight parameter.
        delta: Distance weight parameter.
        psi: Penalty parameter.
        time_limit: Maximum time in seconds for the solver.
        engine: Solver engine to use ('gurobi', 'ortools').
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    Omega: float = 0.1
    delta: float = 0.0
    psi: float = 1.0
    time_limit: float = 600.0
    engine: str = "gurobi"
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
