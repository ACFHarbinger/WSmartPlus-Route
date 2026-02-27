"""
OBA (Old Bachelor Acceptance) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class OBAConfig:
    """Configuration for the Old Bachelor Acceptance policy."""

    engine: str = "oba"
    dilation: float = 5.0
    contraction: float = 2.0
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    vrpp: bool = True
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
