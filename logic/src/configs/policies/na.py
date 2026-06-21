"""
Neural policy configuration.

Attributes:
    NeuralAgentConfig: Configuration for the Neural Agent policy.

Example:
    >>> from configs.policies.na import NeuralAgentConfig
    >>> config = NeuralAgentConfig()
    >>> config.seed
    None
    >>> config.vrpp
    True
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.configs.envs.objective import ObjectiveConfig
from logic.src.configs.models.model import ModelConfig


@dataclass
class NeuralAgentConfig:
    """Configuration for Neural Agent policy.

    Attributes:
        model: Model configuration (architecture and load path).
        reward: Objective/reward weights configuration.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
        seed: Random seed for reproducibility.
        vrpp: Whether the problem is a VRP with Profits.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    mandatory_selection: Optional[Any] = None
    route_improvement: Optional[Any] = None
    seed: Optional[int] = None
    vrpp: bool = True
