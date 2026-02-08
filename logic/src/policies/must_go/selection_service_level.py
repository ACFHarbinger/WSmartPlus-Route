"""
Mean and Standard Deviation based selection strategy module.
"""

from typing import List

import numpy as np

from .base.selection_strategy import MustGoSelectionStrategy, SelectionContext


class ServiceLevelSelection(MustGoSelectionStrategy):
    """
    Statistical overflow prediction strategy.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Select bins that are statistically likely to overflow.

        Args:
            context: Selection context containing fill levels, rates, and std devs.

        Returns:
            List[int]: List of bin IDs (1-based) predicted to overflow.
        """
        if context.accumulation_rates is None or context.std_deviations is None:
            return []

        predicted_fill = (
            context.current_fill + context.accumulation_rates + (context.threshold * context.std_deviations)
        )
        must_go = np.nonzero(predicted_fill >= 100)[0] + 1
        return must_go.tolist()
