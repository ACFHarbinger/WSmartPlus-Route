"""Adaptive Imitation Learning specific configuration."""

from dataclasses import dataclass
from typing import Optional, Union

from logic.src.configs.rl.policies import (
    ACOConfig,
    ALNSConfig,
    HGSALNSConfig,
    HGSConfig,
    ILSConfig,
    RLSConfig,
)

# Type alias for expert policy configurations
ExpertPolicyConfig = Union[HGSConfig, ALNSConfig, ILSConfig, RLSConfig, ACOConfig, HGSALNSConfig]


@dataclass
class AdaptiveImitationConfig:
    """Adaptive Imitation Learning specific configuration."""

    il_weight: float = 1.0
    il_decay: float = 0.95
    patience: int = 5
    threshold: float = 0.05
    decay_step: int = 1
    epsilon: float = 1e-5
    policy_config: Optional[ExpertPolicyConfig] = None
    loss_fn: str = "nll"

    def __post_init__(self):
        """Set default policy config if not provided."""
        if self.policy_config is None:
            # Default to HGS with reasonable settings
            self.policy_config = HGSConfig(time_limit=30.0)
