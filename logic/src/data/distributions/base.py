"""
Statistical sampling distributions.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch

from logic.src.constants import MAX_WASTE


class BaseDistribution(ABC):
    """Constant value sampling."""

    def __init__(self, *args, **kwargs):
        """Initialize Class."""
        self._sampling_method = None

    def set_sampling_method(self, sampling_method: str):
        """Set sampling method.

        Args:
            sampling_method: Sampling method.
        """
        self._sampling_method = sampling_method
        return self

    def sample(
        self, size: Tuple[int, ...], rng: Optional[Union[torch.Generator, np.random.RandomState]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Sample from distribution.

        Args:
            size: Tuple of integers representing the shape of the sample.
            rng: Random number generator.

        Returns:
            Union[np.ndarray, torch.Tensor]: Sampled values.
        """
        sampling_strategies = {
            "sample_array": lambda s: np.clip(self._sample_array(s, rng=rng), 0, MAX_WASTE),
            "sample_tensor": lambda s: torch.clamp(self._sample_tensor(s, generator=rng), min=0.0, max=MAX_WASTE),
        }

        if self._sampling_method not in sampling_strategies:
            raise ValueError(f"Unsupported sample_method: {self._sampling_method}")

        return sampling_strategies[self._sampling_method](size)

    @abstractmethod
    def _sample_array(
        self, size: Tuple[int, ...], rng: Optional[Union[torch.Generator, np.random.RandomState]] = None
    ) -> np.ndarray:
        """Sample from distribution.

        Args:
            size: Tuple of integers representing the shape of the sample.
            rng: Random number generator.

        Returns:
            np.ndarray: Sampled values.
        """
        raise NotImplementedError

    @abstractmethod
    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from distribution.

        Args:
            size: Tuple of integers representing the shape of the sample.
            generator: Random number generator.

        Returns:
            torch.Tensor: Sampled values.
        """
        raise NotImplementedError
