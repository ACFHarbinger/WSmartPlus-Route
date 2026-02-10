"""normalization.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import normalization
"""

from dataclasses import dataclass

from logic.src.constants.models import NORM_EPSILON


@dataclass
class NormalizationConfig:
    """Configuration for normalization layers."""

    norm_type: str = "batch"
    epsilon: float = NORM_EPSILON
    learn_affine: bool = True
    track_stats: bool = False
    momentum: float = 0.1
    n_groups: int = 3
    k_lrnorm: float = 1.0
