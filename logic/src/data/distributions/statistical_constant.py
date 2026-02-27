"""
Statistical sampling distributions - Constant.
"""

from typing import Tuple, Union

import numpy as np
import torch


class Constant:
    """Constant value sampling."""

    def __init__(self, value: Union[float, torch.Tensor] = 1.0):
        """Initialize Class.

        Args:
            value (Union[float, torch.Tensor]): Constant value to return.
        """
        self.value = value

    def sample_tensor(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Return constant tensor.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))

        Returns:
            torch.Tensor: Constant values
        """
        if isinstance(self.value, torch.Tensor):
            return self.value.expand(size)
        return torch.full(size, float(self.value))

    def sample_array(self, size: Tuple[int, ...]) -> np.ndarray:
        """Return constant array.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))

        Returns:
            np.ndarray: Constant values
        """
        return np.full(size, self.value)
