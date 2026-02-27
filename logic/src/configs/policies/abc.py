"""
ABC (Artificial Bee Colony) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ABCConfig:
    """
    Configuration for the Artificial Bee Colony policy.

    Attributes:
        n_sources: Number of food sources (employed bees).
        limit: Abandonment threshold for scout bee phase.
        max_iterations: Maximum ABC cycles.
        n_removal: Nodes removed per neighbourhood perturbation.
        time_limit: Wall-clock time limit in seconds.
        vrpp: If True, solver operates in full VRPP mode.
        must_go: Must-go selection strategy config list.
        post_processing: Post-processing operation config list.
    """

    engine: str = "abc"
    n_sources: int = 20
    limit: int = 10
    max_iterations: int = 200
    n_removal: int = 1
    time_limit: float = 60.0
    vrpp: bool = True
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
