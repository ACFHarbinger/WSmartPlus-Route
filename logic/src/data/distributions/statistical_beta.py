"""
Statistical sampling distributions - Beta.
"""

from typing import Optional, Tuple, Union, cast

import numpy as np
import torch

from .base import BaseDistribution


class Beta(BaseDistribution):
    """Beta distribution sampling."""

    def __init__(self, alpha: Union[float, torch.Tensor] = 0.5, beta: Union[float, torch.Tensor] = 0.5):
        """Initialize Class.

        Args:
            alpha (Union[float, torch.Tensor]): Alpha (shape) parameter.
            beta (Union[float, torch.Tensor]): Beta (scale) parameter.
        """
        self.alpha = alpha
        self.beta = beta

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from Beta distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))
            generator (Optional[torch.Generator], optional): Description of generator.

        Returns:
            torch.Tensor: Sampled values
        """
        if generator is None:
            generator = torch.Generator().manual_seed(42)

        m = torch.distributions.Beta(self.alpha, self.beta)
        return m.sample(torch.Size(size))

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample from Beta distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))
            rng: Optional numpy RandomState for reproducibility.

        Returns:
            np.ndarray: Sampled values
        """
        if rng is None:
            rng = cast(np.random.RandomState, np.random)
        return rng.beta(self.alpha, self.beta, size=size)
