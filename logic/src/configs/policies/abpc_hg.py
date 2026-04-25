"""Adaptive Branch-and-Price-and-Cut with Heuristic Guidance (ABPC-HG) configuration.

Attributes:
   ABPCHGConfig: ABPC-HG policy configuration.

Example:
    >>> from logic.src.configs.policies import ABPCHGConfig
    >>> config = ABPCHGConfig()
    >>> print(config)
    ABPCHGConfig(gamma=0.95, seed=None)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ABPCHGConfig:
    """Config for Adaptive Branch-and-Price-and-Cut with Heuristic Guidance (ABPC-HG).

    Attributes:
        gamma (float): Gamma parameter for adaptive branching.
        seed (Optional[int]): Random seed.
    """

    gamma: float = 0.95
    seed: Optional[int] = None
