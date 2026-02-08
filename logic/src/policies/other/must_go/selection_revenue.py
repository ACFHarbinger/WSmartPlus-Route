"""
Revenue Selection Strategy Module.

This module implements a value-based selection strategy. Bins are selected
if the estimated revenue of their current content exceeds a threshold.

Attributes:
    None

Example:
    >>> from logic.src.policies.must_go.selection_revenue import RevenueThresholdSelection
    >>> strategy = RevenueThresholdSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List

import numpy as np

from logic.src.interfaces.must_go import IMustGoSelectionStrategy

from .base.selection_context import SelectionContext


class RevenueThresholdSelection(IMustGoSelectionStrategy):
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
