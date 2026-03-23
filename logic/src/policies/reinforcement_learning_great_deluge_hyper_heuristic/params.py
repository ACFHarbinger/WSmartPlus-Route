"""
Configuration parameters for the RL-GD-HH solver.

This module defines the hyper-parameters for the Reinforcement Learning –
Great-Deluge Hyper-heuristic, mapping them to the terminology and
experimental settings found in Ozcan et al. (2010).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RLGDHHParams:
    """
    Hyper-parameters for the RL-GD-HH solver.

    Paper Concept Mapping:
    - Reinforcement Learning: Controlled by reward/penalty rates and utility bounds.
    - Great Deluge: Controlled by the time-limited water level growth (target_f).
    - Heuristic Selection: Recommended 'max' utility selection strategy (p. 16).

    Attributes:
        max_iterations: The hard limit on the number of hyper-heuristic search steps.
        time_limit: The total computational budget (T) in seconds. Used for
                    linearizing the Great Deluge water level.
        reward_improvement: The additive reward (e.g., +1) given to a heuristic's
                            utility when it discovers an improving move (Step 15).
        penalty_worsening: The subtractive penalty (e.g., -1) applied to a
                           heuristic's utility for worsening or rejected moves (Step 18).
        utility_upper_bound: The maximum allowed utility (UB). Per paper (p. 16),
                             a typical value is 40.
        min_utility: The lower bound for utility to prevent heuristics from being
                     completely discarded (default 0).
        target_fitness_multiplier: A coefficient to estimate the target fitness
                                    expected by the end of the search.
                                    Level(t) = f0 + (f0*multiplier - f0) * (t/T).
    """

    # Search Control
    max_iterations: int = 5000
    time_limit: float = 60.0
    seed: Optional[int] = None

    # RL1 Adaptation Rates (Section 3.2 Additive Adaptation)
    reward_improvement: float = 1.0
    reward_neutral: float = 0.5
    penalty_worsening: float = 1.0

    # Utility Management (p. 16: Bounds [0, 40])
    utility_upper_bound: float = 40.0
    min_utility: float = 0.0

    # Great Deluge Linearization (Fig 2, Step 19)
    # multiplier = 1.20 implies targeting a 20% profit increase.
    target_fitness_multiplier: float = 1.20

    # Profit-awareness
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "RLGDHHParams":
        """Create parameters from a configuration object."""
        return cls(
            max_iterations=getattr(config, "max_iterations", 5000),
            time_limit=getattr(config, "time_limit", 60.0),
            reward_improvement=getattr(config, "reward_improvement", 1.0),
            penalty_worsening=getattr(config, "penalty_worsening", 1.0),
            utility_upper_bound=getattr(config, "utility_upper_bound", 40.0),
            min_utility=getattr(config, "min_utility", 0.0),
            target_fitness_multiplier=getattr(config, "target_fitness_multiplier", 1.20),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
