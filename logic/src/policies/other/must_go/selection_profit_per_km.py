"""
Marginal Profit-per-km Selection Module.

This module implements a routing-aware economic strategy. Unlike a pure revenue
strategy that naively selects bins based strictly on their valuable contents,
this strategy normalizes the expected revenue by the spatial cost required to
collect it. It naturally deprioritizes highly valuable but highly isolated bins,
acting as a robust proxy for marginal insertion cost.

Example:
    >>> from logic.src.policies.other.must_go.selection_profit_per_km import ProfitPerKmSelection
    >>> strategy = ProfitPerKmSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List

import numpy as np

from logic.src.interfaces.must_go import IMustGoSelectionStrategy
from logic.src.policies.other.must_go.base.selection_context import SelectionContext
from logic.src.policies.other.must_go.base.selection_registry import MustGoSelectionRegistry


@MustGoSelectionRegistry.register("profit_per_km")
class ProfitPerKmSelection(IMustGoSelectionStrategy):
    """
    Economic selection strategy based on spatial ROI (Return on Investment).

    Computes a score for each bin defined as:
    Score_i = (Expected Revenue of Bin i) / (2 * Distance from Depot to Bin i)
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Selects bins whose profit-per-kilometer strictly exceeds a minimum threshold.

        Args:
            context: Selection context containing distance_matrix, fill levels,
                     bin capacity parameters, revenue multipliers, and the
                     profitability threshold.

        Returns:
            List[int]: List of bin IDs (1-based index) exceeding the profit ratio.
        """
        if context.distance_matrix is None:
            raise ValueError("ProfitPerKmSelection requires a distance_matrix.")
        if context.revenue_kg <= 0:
            # Short-circuit if there's no economic value configured
            return []

        # Compute total mass capacity of the bin
        bin_cap = context.bin_volume * context.bin_density

        # Calculate expected revenue based on current fill ratio
        revenue = (context.current_fill / context.max_fill) * bin_cap * context.revenue_kg

        # distance_matrix[0] is the depot. Index 1: end are the bins.
        dist_to_depot = context.distance_matrix[0, 1:]

        # Prevent division by zero for bins that might be co-located with the depot
        safe_dist = np.where(dist_to_depot == 0, 1e-9, dist_to_depot)

        # Proxy for detour cost: Round trip distance to depot (2 * distance)
        profit_per_km = revenue / (2 * safe_dist)

        # Select bins that pass the required economic threshold
        # (e.g., must generate at least $0.50 per km traveled)
        must_go_indices = np.nonzero(profit_per_km > context.threshold)[0]

        return (must_go_indices + 1).tolist()
