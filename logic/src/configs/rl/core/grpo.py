"""GRPO specific configuration.

Attributes:
    GRPOConfig: Configuration for GRPO algorithm.

Example:
    grpo_config = GRPOConfig(
        group_size=8,
        epsilon=0.2,
        epochs=3,
    )
"""

from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """GRPO specific configuration.

    Attributes:
        group_size: Number of trajectories to group together for gradient estimation.
        epsilon: Epsilon value for the algorithm.
        epochs: Number of epochs to train.
    """

    group_size: int = 8
    epsilon: float = 0.2
    epochs: int = 3
