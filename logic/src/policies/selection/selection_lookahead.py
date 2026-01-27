from typing import List

import numpy as np

from ..must_go_selection import MustGoSelectionStrategy, SelectionContext


class LookaheadSelection(MustGoSelectionStrategy):
    """
    Predictive look-ahead selection strategy.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
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

        return must_go_bins
