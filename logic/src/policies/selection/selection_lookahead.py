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

        # 1. Static Lookahead Logic (if days provided)
        if context.lookahead_days is not None:
            horizon = context.lookahead_days
            # Vectorized check: which bins will overflow in 'horizon' days?
            # Assuming monotonic increase: checking the furthest day is sufficient.
            # current + horizon * rate >= 100
            predicted_fill = context.current_fill + (horizon * context.accumulation_rates)

            # Find indices where fill >= 100
            must_go_indices = np.where(predicted_fill >= 100)[0]

            # Convert to 1-based IDs
            return [int(idx) + 1 for idx in must_go_indices]

        # 2. Dynamic Lookahead Logic (Legacy/Cluster behavior)
        # First, identify bins that will overflow "soon" (e.g., tomorrow) to set a baseline cadence
        must_go_bins = []
        for i in range(len(context.current_fill)):
            if context.current_fill[i] + context.accumulation_rates[i] >= 100:
                must_go_bins.append(i)

        if not must_go_bins:
            return []

        # Determine dynamic horizon based on critical bins
        next_coll_day = context.next_collection_day
        if next_coll_day is None:
            coll_days = []
            for i in must_go_bins:
                if context.accumulation_rates[i] > 0:
                    days = int(np.ceil((100.0 - context.current_fill[i]) / context.accumulation_rates[i]))
                    coll_days.append(max(1, days))  # Ensure at least 1 day
            next_coll_day = int(np.min(coll_days)) if coll_days else 1

        # Extend selection to others overflowing within this dynamic horizon
        for i in range(len(context.current_fill)):
            if i in must_go_bins:
                continue

            # Iterate from tomorrow up to horizon
            for k in range(1, next_coll_day + 1):
                if context.current_fill[i] + k * context.accumulation_rates[i] >= 100:
                    must_go_bins.append(i)
                    break

        return [idx + 1 for idx in must_go_bins]
