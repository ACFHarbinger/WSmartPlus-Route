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

from .base.eoq import resolve_trigger_threshold
from .base.selection_context import SelectionContext
from .base.selection_registry import MustGoSelectionRegistry


@MustGoSelectionRegistry.register("revenue")
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
        fill_ratios = context.current_fill / context.max_fill
        expected_revenue = fill_ratios * bin_cap * context.revenue_kg

        if getattr(context, "use_eoq_threshold", False):
            # When EOQ is active, the trigger is the fill-level, but we still
            # check revenue-positivity as a secondary guard.
            must_go_mask = resolve_trigger_threshold(context, fill_ratios)
            must_go_indices = np.nonzero(must_go_mask & (expected_revenue > 0))[0]
        else:
            must_go_indices = np.nonzero(expected_revenue > context.threshold)[0]

        return (must_go_indices + 1).tolist()
