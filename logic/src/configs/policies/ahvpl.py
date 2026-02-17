"""
AHVPL (Augmented Hybrid Volleyball Premier League) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List

from .aco import ACOConfig
from .alns import ALNSConfig
from .hgs import HGSConfig


@dataclass
class AHVPLConfig:
    """
    Configuration for the Augmented Hybrid Volleyball Premier League policy.

    Extends HVPL with HGS integration parameters for diversity-driven
    crossover and bi-criteria fitness evaluation.
    """

    engine: str = "ahvpl"

    # VPL League Parameters
    n_teams: int = 10
    max_iterations: int = 50
    sub_rate: float = 0.2
    time_limit: float = 60.0

    # Nested component configs
    hgs: HGSConfig = field(default_factory=HGSConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # Common policy fields
    must_go: List[str] = field(default_factory=list)
    post_processing: List[Any] = field(default_factory=list)
