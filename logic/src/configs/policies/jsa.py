"""
Joint Simulated Annealing Configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class JointSAConfig:
    """
    Hydra configuration for the Joint Simulated Annealing policy.
    """

    start_temp: float = 1000.0
    cooling_rate: float = 0.995
    max_steps: int = 2000
    restart_limit: int = 5
    prob_bit_flip: float = 0.7
    prob_route_swap: float = 0.3
    overflow_penalty: float = 1000.0
    seed: Optional[int] = 42
    time_limit: float = 60.0

    # Hydra compatibility placeholders
    vrpp: bool = True
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
