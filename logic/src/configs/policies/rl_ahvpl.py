"""
RL-AHVPL (Reinforcement Learning Augmented Hybrid Volleyball Premier League) configuration for Hydra.

Attributes:
    RLAHVPLConfig: Configuration for the RL-AHVPL policy.

Example:
    >>> from configs.policies.rl_ahvpl import RLAHVPLConfig
    >>> config = RLAHVPLConfig()
    >>> config.n_teams
    10
    >>> config.time_limit
    60.0
    >>> config.max_iterations
    1000
    >>> config.vrpp
    True
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .aco_ks import KSparseACOConfig
from .alns import ALNSConfig
from .hgs import HGSConfig
from .other import RLConfig
from .rts import RTSConfig


@dataclass
class RLAHVPLConfig:
    """
    Configuration for the Reinforcement Learning Augmented Hybrid Volleyball Premier League policy.

    Combines VPL population dynamics, ACO initialization, ALNS local search, and HGS diversity
    management / crossover, all using Reinforcement Learning (RL) for adaptive operator selection.

    Attributes:
        n_teams: Number of teams in the VPL population.
        time_limit: Time limit for the algorithm in seconds.
        max_iterations: Maximum number of iterations.
        elite_coaching_max_iterations: Maximum number of iterations for elite coaching.
        not_coached_max_iterations: Maximum number of iterations for not coached.
        coaching_acceptance_threshold: Threshold for accepting coaching.
        sub_rate: Rate of subproblem solving.
        aco: Configuration for the ACO initialization.
        alns: Configuration for the ALNS local search.
        hgs: Configuration for the HGS diversity management.
        rts: Configuration for the RTS diversity management.
        rl_config: Configuration for the RL policy.
        tabu_no_repeat_threshold: Threshold for tabu no repeat.
        gls_penalty_lambda: Penalty lambda for GLS.
        gls_penalty_alpha: Penalty alpha for GLS.
        gls_penalty_step: Penalty step for GLS.
        gls_probability: Probability of using GLS.
        vrpp: Whether to use VRPP.
        seed: Random seed.
        profit_aware_operators: Whether to use profit aware operators.
        mandatory_selection: List of mandatory nodes.
        route_improvement: List of route improvements.
    """

    # General parameters
    n_teams: int = 10
    time_limit: float = 60.0
    max_iterations: int = 1000
    elite_coaching_max_iterations: int = 500
    not_coached_max_iterations: int = 100
    coaching_acceptance_threshold: float = 1e-6
    sub_rate: float = 0.2

    # Nested component configs
    aco: KSparseACOConfig = field(default_factory=KSparseACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)
    hgs: HGSConfig = field(default_factory=HGSConfig)
    rts: RTSConfig = field(default_factory=RTSConfig)

    # RL Configuration (Centralized)
    rl_config: RLConfig = field(default_factory=RLConfig)

    # Tabu parameters
    tabu_no_repeat_threshold: int = 2

    # GLS parameters
    gls_penalty_lambda: float = 1.0
    gls_penalty_alpha: float = 0.5
    gls_penalty_step: int = 10
    gls_probability: float = 0.5

    # Common policy fields
    vrpp: bool = True
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    mandatory_selection: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
