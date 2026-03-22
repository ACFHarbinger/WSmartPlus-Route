"""
LAHC (Late Acceptance Hill-Climbing) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class LAHCConfig:
    """
    Configuration for the LAHC policy.

    Attributes:
        engine: Policy engine identifier.
        queue_size: Length of the circular history queue (L).
        max_iterations: Total LLH applications.
        n_removal: Nodes removed per destroy step.
        n_llh: LLH pool size.
        time_limit: Wall-clock time limit in seconds.
        profit_aware_operators: If True, uses profit-aware destroy and repair operators.
        vrpp: If True, solver operates in full VRPP mode.
        must_go: Must-go selection strategy config list.
        post_processing: Post-processing operation config list.
    """

    engine: str = "lahc"
    queue_size: int = 50
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = True
    vrpp: bool = True
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
