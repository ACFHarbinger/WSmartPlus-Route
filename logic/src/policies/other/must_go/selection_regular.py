"""
Regular Selection Strategy Module.

This module implements a fixed-frequency selection strategy. Bins are selected
if the current day matches their assigned schedule frequency (e.g., every 3 days).

Attributes:
    None

Example:
    >>> from logic.src.policies.must_go.selection_regular import RegularSelection
    >>> strategy = RegularSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List

from logic.src.interfaces.must_go import IMustGoSelectionStrategy
from logic.src.policies.other.must_go.base.selection_context import SelectionContext


class RegularSelection(IMustGoSelectionStrategy):
    """
    Periodic collection strategy.

    Logic: Collect all bins if today is a scheduled collection day.
    Scheduled if (current_day % (frequency + 1)) == 1.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Select all bins if it is a scheduled collection day.

        Args:
            context: SelectionContext with day and threshold (freq).

        Returns:
            List[int]: List of all bin IDs if collection day, else empty.
        """
        # threshold is used as frequency 'lvl'
        # Prevent ZeroDivisionError if threshold + 1 is 0 (i.e., threshold is -1)
        if context.threshold < 0:
            return []

        if context.threshold == 0:
            return (context.bin_ids + 1).tolist()

        if (context.current_day % (int(context.threshold) + 1)) == 1:
            # Return all bins (1-based IDs)
            return (context.bin_ids + 1).tolist()
        return []
