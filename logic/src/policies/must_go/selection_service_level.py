"""
Service Level Selection Strategy Module.

This module implements a robust optimization strategy that selects bins
to maintain a service level guarantee. It uses fill rate uncertainty (std dev)
to ensure bins are collected before they have a high probability of overflow.

Attributes:
    None

Example:
    >>> from logic.src.policies.must_go.selection_service_level import ServiceLevelSelection
    >>> strategy = ServiceLevelSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List

import numpy as np

from logic.src.interfaces.must_go import MustGoSelectionStrategy
from logic.src.policies.must_go.base.selection_context import SelectionContext


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
