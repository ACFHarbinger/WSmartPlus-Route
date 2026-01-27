"""
Revenue-based selection strategy module.
"""
from typing import List

import numpy as np

from ..must_go_selection import MustGoSelectionStrategy, SelectionContext


class RevenueThresholdSelection(MustGoSelectionStrategy):
    """
    Revenue-based selection strategy.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Select bins where expected revenue exceeds the threshold.

        Args:
            context: SelectionContext with bin properties, revenue parameters, and threshold.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        bin_cap = context.bin_volume * context.bin_density
        expected_revenue = (context.current_fill / 100.0) * bin_cap * context.revenue_kg

        must_go = np.nonzero(expected_revenue > context.threshold)[0] + 1
        return must_go.tolist()
