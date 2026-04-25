"""
Statistical sampling distributions.

Attributes:
    BaseDistribution: Base class for sampling distributions.

Example:
    >>> from logic.src.data.distributions import BaseDistribution
    >>> distribution = BaseDistribution()
    >>> print(distribution.sample())
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch

from logic.src.constants import MAX_WASTE


class BaseDistribution(ABC):
    """Constant value sampling.

    Attributes:
        _sampling_method: Sampling method.
    """

    def __init__(self, *args, **kwargs):
        """Initialize Class.

        Args:
            args: Description of args.
            kwargs: Description of kwargs.
        """
        self._sampling_method = None

    def set_sampling_method(self, sampling_method: str):
        """Set sampling method.

        Args:
            sampling_method: Sampling method.

        Returns:
            Description of return value.
        """
        self._sampling_method = sampling_method
        return self

    def sample(
        self, size: Tuple[int, ...], rng: Optional[Union[torch.Generator, np.random.Generator]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Sample from distribution.

        Args:
            size: Tuple of integers representing the shape of the sample.
            rng: Random number generator.

        Returns:
            Union[np.ndarray, torch.Tensor]: Sampled values.
        """
        if self._sampling_method == "sample_array":
            assert rng is None or isinstance(rng, np.random.Generator)
            return np.clip(self._sample_array(size, rng=rng), 0, MAX_WASTE)  # type: ignore[arg-type]
        elif self._sampling_method == "sample_tensor":
            assert rng is None or isinstance(rng, torch.Generator)
            return torch.clamp(self._sample_tensor(size, generator=rng), min=0.0, max=MAX_WASTE)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unsupported sample_method: {self._sampling_method}")

    @abstractmethod
    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.Generator] = None) -> np.ndarray:
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
