"""
RL-GD-HH (Reinforcement Learning Great Deluge Hyper-Heuristic) configuration for Hydra.

This module defines the schema and default values for the RL-GD-HH policy,
ensuring type-safety and providing documentation for each configurable
hyper-parameter.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class RLGDHHConfig:
    """
    Configuration for the RL-GD-HH policy.

    The values defined here represent the system defaults aligned with
    the Ozcan et al. (2010) framework.
    """

    # Primary engine identifier used by the policy registry
    engine: str = "rl_gd_hh"

    # Search Control
    max_iterations: int = 5000
    time_limit: float = 60.0

    # RL1 Adaptation Settings (Ozcan p. 16)
    reward_improvement: float = 1.0
    reward_neutral: float = 0.5
    penalty_worsening: float = 1.0

    # Utility Constraints
    utility_upper_bound: float = 40.0
    min_utility: float = 0.0

    # Great Deluge Level Update Settings
    target_fitness_multiplier: float = 1.20

    # Great Deluge specific parameters
    rain_speed: float = 0.001
    flood_margin: float = 0.1

    # RL1 Adaptation
    initial_utility: float = 20.0

    # Deterministic Seed
    seed: Optional[int] = None

    # Problem Flags
    vrpp: bool = True

    # Infrastructure Hooks
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
