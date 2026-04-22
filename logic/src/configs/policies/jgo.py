"""
Joint Greedy Orienteering Configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class JointGreedyConfig:
    """
    Hydra configuration for the Joint Greedy Orienteering policy.
    """

    k_best: int = 3
    n_starts: int = 10
    distance_weight: float = 1.0
    seed: Optional[int] = 42
    time_limit: float = 30.0

    # Hydra compatibility placeholders
    vrpp: bool = True
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
