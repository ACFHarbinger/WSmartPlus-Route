"""
Statistical sampling distributions - Beta.
"""

from typing import Tuple, Union

import numpy as np
import torch


class Beta:
    """Beta distribution sampling."""

    def __init__(self, alpha: Union[float, torch.Tensor] = 0.5, beta: Union[float, torch.Tensor] = 0.5):
        """Initialize Class.

        Args:
            alpha (Union[float, torch.Tensor]): Alpha (shape) parameter.
            beta (Union[float, torch.Tensor]): Beta (scale) parameter.
        """
        self.alpha = alpha
        self.beta = beta

    def sample_tensor(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Sample from Beta distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))

        Returns:
            torch.Tensor: Sampled values
        """
        m = torch.distributions.Beta(self.alpha, self.beta)
        return m.sample(torch.Size(size))

    def sample_array(self, size: Tuple[int, ...]) -> np.ndarray:
        """Sample from Beta distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))

        Returns:
            np.ndarray: Sampled values
        """
        return np.random.beta(self.alpha, self.beta, size=size)
