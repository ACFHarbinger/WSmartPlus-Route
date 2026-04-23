"""Last-minute selection strategy.

This module provides a threshold-based reactive selection strategy that marks
bins for collection only when their fill levels exceed a specified limit.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import VectorizedSelector


class LastMinuteSelector(VectorizedSelector):
    """Threshold-based reactive selection strategy.

    Selects bins where the current fill level exceeds a predefined threshold.
    This strategy is simple but reactive, only triggering collection when bins
    are nearly full.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        """Initialize the last-minute selector.

        Args:
            threshold: Fill level threshold in [0, 1].
        """
        self.threshold = threshold

    def select(
        self,
        fill_levels: torch.Tensor,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins exceeding the fill threshold.

        Args:
            fill_levels: Current fill levels [B, N] in [0, 1].
            threshold: Optional override for the predefined fill threshold.
            **kwargs: Extra parameters (ignored).

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
        """
        thresh = threshold if threshold is not None else self.threshold
        mandatory = fill_levels > thresh

        # Depot (index 0) is never mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
