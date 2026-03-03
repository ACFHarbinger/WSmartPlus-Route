"""
SA (Simulated Annealing) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SAConfig:
    """Configuration for the Simulated Annealing policy."""

    engine: str = "sa"
    initial_temp: float = 100.0
    alpha: float = 0.995
    min_temp: float = 0.01
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
