"""
Regular interval selection strategy module.
"""
from typing import List

from ..must_go_selection import MustGoSelectionStrategy, SelectionContext


class RegularSelection(MustGoSelectionStrategy):
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
