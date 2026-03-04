"""
GLS (Guided Local Search) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GLSConfig:
    """Configuration for the Guided Local Search policy."""

    engine: str = "gls"
    lambda_param: float = 1.0
    alpha_param: float = 0.3
    max_restarts: int = 50
    n_removal: int = 2
    n_llh: int = 5
    inner_iterations: int = 20
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
