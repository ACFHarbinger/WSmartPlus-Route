"""
Statistical sampling distributions.
"""

from typing import Tuple, Union

import torch


class Gamma:
    """Gamma distribution sampling."""

    def __init__(self, alpha: Union[float, torch.Tensor] = 2.0, beta: Union[float, torch.Tensor] = 2.0):
        """Initialize Class.

        Args:
            alpha (Union[float, torch.Tensor]): Description of alpha.
            beta (Union[float, torch.Tensor]): Description of beta.
        """
        self.alpha = alpha
        self.beta = beta

    def sample(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Sample from Gamma distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))
        """
        m = torch.distributions.Gamma(self.alpha, self.beta)
        return m.sample(torch.Size(size))
