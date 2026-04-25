"""
AHVPL (Augmented Hybrid Volleyball Premier League) configuration.

Attributes:
    AHVPLConfig: Attributes for AHVPL configuration.

Example:
    >>> from configs.policies.ahvpl import AHVPLConfig
    >>> config = AHVPLConfig()
    >>> config.n_teams
    10
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .aco_ks import KSparseACOConfig
from .alns import ALNSConfig
from .hgs import HGSConfig


@dataclass
class AHVPLConfig:
    """
    Configuration for the Augmented Hybrid Volleyball Premier League policy.

    Extends HVPL with HGS integration parameters for diversity-driven
    crossover and bi-criteria fitness evaluation.

    Attributes:
        n_teams: Number of teams in the VPL league.
        max_iterations: Maximum number of iterations for the VPL algorithm.
        sub_rate: Substitution rate for the VPL algorithm.
        time_limit: Time limit for the VPL algorithm in seconds.
        alns_elite_iterations: Number of iterations for the ALNS elite group.
        alns_not_coached_iterations: Number of iterations for the ALNS not coached group.
        seed: Random seed for the VPL algorithm.
        vrpp: Whether the VPL algorithm is used for VRPP problems.
        profit_aware_operators: Whether to use profit-aware operators in the VPL algorithm.
        hgs: Configuration for HGS integration.
        aco: Configuration for ACO integration.
        alns: Configuration for ALNS integration.
        mandatory_selection: List of mandatory nodes to select.
        route_improvement: List of route improvement strategies.
    """

    # VPL League Parameters
    n_teams: int = 10
    max_iterations: int = 50
    sub_rate: float = 0.2
    time_limit: float = 60.0
    alns_elite_iterations: int = 500
    alns_not_coached_iterations: int = 100
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Nested component configs
    hgs: HGSConfig = field(default_factory=HGSConfig)
    aco: KSparseACOConfig = field(default_factory=KSparseACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # Common policy fields
    mandatory_selection: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
