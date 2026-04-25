"""Service level (Statistical) selection strategy.

This module provides a statistical selection strategy that predicts overflow
probabilities using mean accumulation rates and standard deviations, marking
bins for collection based on a specified confidence level.

Attributes:
    ServiceLevelSelector: Statistical overflow prediction policy.

Example:
    >>> selector = ServiceLevelSelector(confidence_factor=1.5)
    >>> mask = selector.select(fill_levels, rates, stds)
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import VectorizedSelector


class ServiceLevelSelector(VectorizedSelector):
    """Statistical overflow prediction strategy.

    Uses mean accumulation rates and their standard deviations to predict the
    likelihood of overflow. A bin is selected if its predicted peak fill level
    (mean + confidence * std) exceeds the overflow threshold.

    Attributes:
        confidence_factor: Safety factor (z-score) for overflow risk.
        max_fill: Target maximum fill level before overflow.
    """

    def __init__(self, confidence_factor: float = 1.0, max_fill: float = 1.0) -> None:
        """Initialize the service level selector.

        Args:
            confidence_factor: Number of standard deviations for prediction.
                Higher values lead to more conservative (earlier) collections.
            max_fill: Maximum fill level (overflow threshold).
        """
        self.confidence_factor = confidence_factor
        self.max_fill = max_fill

    def select(
        self,
        fill_levels: torch.Tensor,
        accumulation_rates: Optional[torch.Tensor] = None,
        std_deviations: Optional[torch.Tensor] = None,
        confidence_factor: Optional[float] = None,
        max_fill: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins statistically likely to overflow.

        Prediction Logic:
            predicted_fill = current + rate + (confidence * std) >= max_fill

        Args:
            fill_levels: Current fill levels [B, N].
            accumulation_rates: Mean daily waste generation per node [B, N].
            std_deviations: Standard deviation of daily generation [B, N].
            confidence_factor: Override for risk-aversion factor.
            max_fill: Override for overflow limit.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
        """
        conf = confidence_factor if confidence_factor is not None else self.confidence_factor
        overflow_thresh = max_fill if max_fill is not None else self.max_fill

        if accumulation_rates is None or std_deviations is None:
            # Without statistics, fall back to threshold-based
            mandatory = fill_levels >= overflow_thresh
        else:
            # Statistical prediction: current + mean + confidence * std
            predicted_fill = fill_levels + accumulation_rates + (conf * std_deviations)
            mandatory = predicted_fill >= overflow_thresh

        # Depot is never mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
