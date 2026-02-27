"""
Statistical sampling distributions - Uniform.
"""

from typing import Tuple

import torch
import numpy as np


class Uniform:
    """Discrete uniform sampling distribution."""

    def __init__(self, low: int = 0, high: int = 100):
        """Initialize Class.

        Args:
            low (int): Minimum value for randint.
            high (int): Maximum value for randint.
        """
        self.low = low
        self.high = high

    def sample_tensor(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Sample from discrete uniform distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))

        Returns:
            torch.Tensor: Sampled values in range [0.01, 1.0].
        """
        res = torch.randint(self.low, self.high, size)
        return res.float() / 100.0

    def sample_array(self, size: Tuple[int, ...]) -> np.ndarray:
        """Sample from discrete uniform distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))

        Returns:
            np.ndarray: Sampled values in range [1, 100].
        """
        return np.random.randint(self.low, self.high, size=size)
