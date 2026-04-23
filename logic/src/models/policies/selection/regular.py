"""Regular selection strategy.

This module provides a periodic collection strategy that marks all bins for
collection on scheduled days based on a fixed frequency interval.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from .base import VectorizedSelector


class RegularSelector(VectorizedSelector):
    """Periodic collection strategy.

    Selects all bins on scheduled collection days based on a fixed frequency.
    If it is a collection day for a specific batch, all active bins in that
    batch are marked as mandatory.
    """

    def __init__(self, frequency: int = 3) -> None:
        """Initialize the regular selector.

        Args:
            frequency: Collection interval in days (e.g., 3 means collect every 3rd day).
        """
        self.frequency = frequency

    def select(
        self,
        fill_levels: torch.Tensor,
        current_day: Optional[Union[torch.Tensor, int]] = None,
        frequency: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select all bins if today is a scheduled collection day.

        Args:
            fill_levels: Current fill levels [B, N].
            current_day: Current simulation day [B] or scalar.
            frequency: Optional override for collection frequency.
            **kwargs: Extra parameters (ignored).

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device
        freq = frequency if frequency is not None else self.frequency

        if freq <= 0:
            # Collect every day
            mandatory = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        elif current_day is None:
            # No day info - assume collection day
            mandatory = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        else:
            # Ensure current_day is a tensor
            if not isinstance(current_day, torch.Tensor):
                current_day = torch.tensor(current_day, device=device)

            # Expand to batch if scalar
            if current_day.dim() == 0:
                current_day = current_day.expand(batch_size)

            # Collection day check: day % (freq + 1) == 1
            is_collection_day = (current_day % (freq + 1)) == 1  # (batch_size,)
            is_collection_day = is_collection_day.unsqueeze(-1)  # (batch_size, 1)

            # If collection day, all bins are mandatory
            mandatory = is_collection_day.expand(-1, num_nodes)

        # Depot is never a mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
