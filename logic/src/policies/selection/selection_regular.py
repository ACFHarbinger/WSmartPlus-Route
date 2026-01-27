from typing import List

from ..must_go_selection import MustGoSelectionStrategy, SelectionContext


class RegularSelection(MustGoSelectionStrategy):
    """
    Periodic collection strategy.

    Logic: Collect all bins if today is a scheduled collection day.
    Scheduled if (current_day % (frequency + 1)) == 1.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        # threshold is used as frequency 'lvl'
        if (context.current_day % (int(context.threshold) + 1)) == 1:
            # Return all bins (1-based IDs)
            return (context.bin_ids + 1).tolist()
        return []
