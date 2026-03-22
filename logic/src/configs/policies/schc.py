"""
SCHC (Step Counting Hill Climbing) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SCHCConfig:
    """Configuration for the Step Counting Hill Climbing policy."""

    engine: str = "schc"
    max_iterations: int = 1000
    step_size: int = 100
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = True
    vrpp: bool = True
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
