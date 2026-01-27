"""
Lookahead selection strategy module.
"""
from typing import List

import numpy as np

from ..must_go_selection import MustGoSelectionStrategy, SelectionContext


class LookaheadSelection(MustGoSelectionStrategy):
    """
    Predictive selection strategy looking N days ahead.

    Selects bins that will overflow within the lookahead horizon.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Select bins predicted to overflow within N days.

        Args:
            context: SelectionContext with fill levels, accumulation rates, and lookahead parameters.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        if context.lookahead_days is None and context.accumulation_rates is None:
            return []

        if context.accumulation_rates is None:
            return []

        must_go_bins = []
        for i in range(len(context.current_fill)):
            if context.current_fill[i] + context.accumulation_rates[i] >= 100:
                must_go_bins.append(i)

        if not must_go_bins:
            return []

        next_coll_day = context.next_collection_day
        if context.lookahead_days is not None:
            next_coll_day = context.lookahead_days
        elif next_coll_day is None:
            coll_days = []
            for i in must_go_bins:
                if context.accumulation_rates[i] > 0:
                    days = int(np.ceil(100.0 / context.accumulation_rates[i]))
                    coll_days.append(days)
            next_coll_day = int(np.min(coll_days)) if coll_days else 1

        for i in range(len(context.current_fill)):
            if i in must_go_bins:
                continue

            for j in range(context.current_day + 1, context.current_day + next_coll_day):
                if context.current_fill[i] + j * context.accumulation_rates[i] >= 100:
                    must_go_bins.append(i)
                    break

        # Return 1-based bin IDs
        return [idx + 1 for idx in must_go_bins]
