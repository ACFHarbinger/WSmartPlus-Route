"""
HS (Harmony Search) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class HSConfig:
    """
    Configuration for the Harmony Search policy.

    Attributes:
        hm_size: Harmony Memory size (archive capacity).
        HMCR: Harmony Memory Considering Rate ∈ [0, 1].
        PAR: Pitch Adjusting Rate ∈ [0, 1].
        max_iterations: Total improvisation cycles.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        vrpp: If True, solver operates in full VRPP mode.
        must_go: Must-go selection strategy config list.
        post_processing: Post-processing operation config list.
    """

    engine: str = "hs"
    hm_size: int = 10
    HMCR: float = 0.9
    PAR: float = 0.3
    max_iterations: int = 500
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
