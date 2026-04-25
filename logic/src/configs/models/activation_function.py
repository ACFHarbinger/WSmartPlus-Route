"""activation_function.py module.

Attributes:
    ActivationConfig: Configuration for activation functions.

Example:
    >>> from logic.src.configs.models.activation_function import ActivationConfig
    >>> config = ActivationConfig()
    >>> print(config)
    ActivationConfig(name='gelu', param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, range=[0.125, 0.3333333333333333])
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ActivationConfig:
    """Configuration for activation functions.

    Attributes:
        name (str): Name of the activation function.
        param (float): Parameter of the activation function.
        threshold (float): Threshold of the activation function.
        replacement_value (float): Replacement value of the activation function.
        n_params (int): Number of parameters of the activation function.
        range (List[float]): Range of the activation function.
    """

    name: str = "gelu"
    param: float = 1.0
    threshold: float = 6.0
    replacement_value: float = 6.0
    n_params: int = 3
    # Use default_factory for mutable defaults
    range: List[float] = field(default_factory=lambda: [0.125, 1 / 3])
