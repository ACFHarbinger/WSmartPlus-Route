"""normalization.py module.

Attributes:
    NormalizationConfig: Normalization layer configuration.

Example:
    >>> from logic.src.configs.models import NormalizationConfig
    >>> config = NormalizationConfig()
    >>> print(config)
    NormalizationConfig(norm_type='batch', epsilon=1e-05, learn_affine=True, track_stats=False, momentum=0.1, n_groups=1, k_lrnorm=1.0)
"""

from dataclasses import dataclass

from logic.src.constants.models import NORM_EPSILON


@dataclass
class NormalizationConfig:
    """Configuration for normalization layers.

    Attributes:
        norm_type (str): Type of normalization.
        epsilon (float): Epsilon value.
        learn_affine (bool): Whether to learn affine parameters.
        track_stats (bool): Whether to track statistics.
        momentum (float): Momentum value.
        n_groups (int): Number of groups.
        k_lrnorm (float): Learning rate for normalization.
    """

    norm_type: str = "batch"
    epsilon: float = NORM_EPSILON
    learn_affine: bool = True
    track_stats: bool = False
    momentum: float = 0.1
    n_groups: int = 1
    k_lrnorm: float = 1.0
