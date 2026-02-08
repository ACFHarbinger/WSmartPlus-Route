"""Imitation specific configuration."""

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
class ImitationConfig:
    """Imitation learning configuration.

    Attributes:
        policy_config: Configuration for the expert policy to imitate.
                      Can be any of: HGSConfig, ALNSConfig, ILSConfig, RLSConfig, ACOConfig, HGSALNSConfig.
        loss_fn: Loss function to use ('nll' for negative log-likelihood, 'mse' for mean squared error).
    """

    policy_config: Optional[ExpertPolicyConfig] = None
    loss_fn: str = "nll"

    def __post_init__(self):
        """Set default policy config if not provided."""
        if self.policy_config is None:
            # Default to HGS with reasonable settings
            self.policy_config = HGSConfig(time_limit=30.0)
