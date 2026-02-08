"""
Last Minute Selection Strategy Module.

This module implements the "Last Minute" or "Reactive" strategy, which
selects bins only when they exceed a certain fill threshold (e.g., 100%).

Attributes:
    None

Example:
    >>> from logic.src.policies.must_go.selection_last_minute import LastMinuteSelection
    >>> strategy = LastMinuteSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List

import numpy as np

from logic.src.interfaces.must_go import MustGoSelectionStrategy
from logic.src.policies.must_go.base.selection_context import SelectionContext


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
