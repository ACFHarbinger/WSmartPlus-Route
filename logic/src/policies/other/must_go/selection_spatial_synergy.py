"""
Spatial Synergy Selection Strategy Module.

This strategy leverages spatial graph properties to amortize routing costs.
It identifies "critical" bins that must be collected (e.g., > 90% full) and
extends the obligatory collection set to include nearby bins that are
moderately full (e.g., > 60% full). By visiting these moderately full bins
opportunistically, the vehicle avoids returning to the same neighborhood
the very next day.

Example:
    >>> from logic.src.policies.other.must_go.selection_spatial_synergy import SpatialSynergySelection
    >>> strategy = SpatialSynergySelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List

import numpy as np

from logic.src.interfaces.must_go import IMustGoSelectionStrategy
from logic.src.policies.other.must_go.base.selection_context import SelectionContext
from logic.src.policies.other.must_go.base.selection_registry import MustGoSelectionRegistry


@MustGoSelectionRegistry.register("spatial_synergy")
class SpatialSynergySelection(IMustGoSelectionStrategy):
    """
    Spatial-density based collection strategy.

    Selects bins that are critically full, plus any moderately full bins
    within a strict distance radius of those critical bins. Operates statelessly
    by extracting parameters dynamically from the SelectionContext.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Select critical bins and their valid spatial synergies.

        Args:
            context: SelectionContext containing fill levels, distance_matrix,
                     and optional spatial parameters (critical_threshold,
                     synergy_threshold, radius).

        Returns:
            List[int]: List of bin IDs (1-based index) strictly required for collection.
        """
        if context.distance_matrix is None:
            raise ValueError("SpatialSynergySelection requires a distance_matrix in the context.")

        # Extract parameters dynamically from context with robust defaults
        critical_threshold = getattr(context, "critical_threshold", 0.90)
        synergy_threshold = getattr(context, "synergy_threshold", 0.60)
        radius = getattr(context, "radius", 10.0)

        # Calculate fill ratio array
        fill_ratios = context.current_fill / context.max_fill

        # 1. Identify critically full bins (0-based indices)
        critical_indices = np.nonzero(fill_ratios > critical_threshold)[0]
        if len(critical_indices) == 0:
            return []

        # 2. Identify moderately full bins that are eligible for synergy
        moderate_indices = np.nonzero(fill_ratios > synergy_threshold)[0]
        must_go_set = set(critical_indices.tolist())

        # 3. For each critical bin, find moderately full neighbors
        # Note: distance_matrix includes the depot at index 0.
        # Bin i corresponds to index i+1 in the distance_matrix.
        for crit_idx in critical_indices:
            dist_to_crit = context.distance_matrix[crit_idx + 1, 1:]

            # Find all bins within the specified spatial radius
            neighbors = np.nonzero(dist_to_crit <= radius)[0]

            # Intersect spatial neighbors with the moderately full condition
            synergy_bins = np.intersect1d(neighbors, moderate_indices)
            must_go_set.update(synergy_bins.tolist())

        # Convert back to 1-based IDs for the routing engine
        return [i + 1 for i in must_go_set]
