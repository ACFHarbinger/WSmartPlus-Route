"""
Base module for vectorized selection strategies.
"""

from abc import ABC, abstractmethod

from torch import Tensor


class VectorizedSelector(ABC):
    """Abstract base class for vectorized bin selection strategies."""

    @abstractmethod
    def select(
        self,
        fill_levels: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Select bins that must be collected.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes).
                         Values in [0, 1] where 1.0 = 100% full.
            **kwargs: Strategy-specific parameters.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes) where True = must collect.
                    Note: Index 0 is the depot and should always be False.
        """
        pass
