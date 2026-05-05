"""Lookahead selection strategy.

This module provides a predictive selection strategy that marks bins for
collection based on when they are expected to overflow, using current fill
levels and accumulation rates to calculate a dynamic lookahead horizon.

Attributes:
    LookaheadSelector: Predictive selection policy looking ahead several days.

Example:
    >>> selector = LookaheadSelector(current_collection_day=0)
    >>> mask = selector.select(fill_levels, rates)
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from logic.src.constants import MAX_WASTE

from .base import VectorizedSelector


class LookaheadSelector(VectorizedSelector):
    """Predictive selection looking several days ahead.

    Selects bins that are predicted to overflow within a dynamic lookahead
    horizon. The horizon is determined by the earliest upcoming overflow among
    the currently full bins.

    Attributes:
        current_collection_day: Reference day for lookahead simulation.
    """

    def __init__(self, current_collection_day: int = 0, **kwargs: Any) -> None:
        """Initialize the lookahead selector.

        Args:
            current_collection_day: Current day in the collection cycle.
            kwargs: Additional keyword arguments.
        """
        self.current_collection_day = current_collection_day

    def select(
        self,
        fill_levels: torch.Tensor,
        accumulation_rates: Optional[torch.Tensor] = None,
        current_collection_day: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins predicted to overflow within dynamic horizon.

        Args:
            fill_levels: Current fill levels [B, N].
            accumulation_rates: Mean daily waste generation per node [B, N].
            current_collection_day: Override for the reference day.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
        """
        overflow_thresh = MAX_WASTE
        day = current_collection_day if current_collection_day is not None else self.current_collection_day

        if accumulation_rates is None:
            # Without rates, no prediction possible - select nothing
            mandatory = torch.zeros_like(fill_levels, dtype=torch.bool)
        else:
            # 1. Identify bins that will overflow today after accumulation
            initial_mandatory = (fill_levels + accumulation_rates) >= overflow_thresh

            # 2. Simulate collection: collected bins become 0.
            # Calculate absolute day of next overflow for these bins
            rates_safe = accumulation_rates.clamp(min=1e-6)
            days_until_overflow = torch.ceil(overflow_thresh / rates_safe)

            # day can be a scalar or [B], ensure it can broadcast with [B, N]
            day_tensor = day.unsqueeze(1) if isinstance(day, torch.Tensor) and day.dim() > 0 else day
            abs_overflow_days = day_tensor + days_until_overflow

            # Mask: only consider days for initially selected bins
            LARGE_NUM = 1e9  # Use a larger number for absolute days
            next_abs_overflow_days = torch.where(
                initial_mandatory,
                abs_overflow_days,
                torch.tensor(LARGE_NUM, device=fill_levels.device, dtype=abs_overflow_days.dtype),
            )

            # 3. Find the minimum next absolute overflow day for each batch
            next_collection_day = next_abs_overflow_days.min(dim=1).values

            # 4. Check if other bins overflow before this next collection day
            # Horizon relative to today: (next_collection_day - today - 1)
            # day is [B], next_collection_day is [B]
            check_days = (next_collection_day - day - 1.0).clamp(min=1.0)

            # Broadcast check_days to (B, 1) for multiplication
            check_days = check_days.unsqueeze(1)

            # Final check includes initial selection + new candidates
            predicted_fill = fill_levels + (check_days * accumulation_rates)
            mandatory = predicted_fill >= overflow_thresh

        # Depot is never a mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False
        return mandatory
