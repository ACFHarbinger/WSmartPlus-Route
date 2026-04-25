"""POMO specific configuration.

Attributes:
    POMOConfig: Configuration for POMO algorithm.

Example:
    pomo_config = POMOConfig(
        num_augment=1,
        num_starts=None,
        augment_fn="dihedral8",
    )
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class POMOConfig:
    """POMO specific configuration.

    Attributes:
        num_augment: Number of augmentations to perform.
        num_starts: Number of starts for the algorithm.
        augment_fn: Function to use for augmentation.
    """

    num_augment: int = 1
    num_starts: Optional[int] = None
    augment_fn: str = "dihedral8"
