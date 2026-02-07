"""
Last Minute selection strategy module.
"""
from typing import List

import numpy as np

from ..must_go_selection import MustGoSelectionStrategy, SelectionContext


class LastMinuteSelection(MustGoSelectionStrategy):
    """
    Simple threshold-based reactive strategy.

    Logic: Collect if current_fill > threshold.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Select bins that exceed the fill threshold.

        Args:
            context: SelectionContext with fill levels and threshold.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        must_go = np.nonzero(context.current_fill > context.threshold)[0] + 1
        return must_go.tolist()
