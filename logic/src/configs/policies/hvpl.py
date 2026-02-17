"""
HVPL Configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List

from .aco import ACOConfig
from .alns import ALNSConfig


@dataclass
class HVPLConfig:
    """
    Configuration for the Hybrid Volleyball Premier League policy.
    """

    engine: str = "hvpl"

    # League Parameters
    n_teams: int = 10
    max_iterations: int = 50
    sub_rate: float = 0.2
    time_limit: float = 60.0

    # Nested component configs
    aco: ACOConfig = field(default_factory=ACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # Common policy fields
    must_go: List[str] = field(default_factory=list)
    post_processing: List[Any] = field(default_factory=list)
