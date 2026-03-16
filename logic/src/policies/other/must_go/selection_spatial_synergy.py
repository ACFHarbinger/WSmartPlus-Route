"""
Spatial Synergy Selection Strategy Module.

This strategy leverages spatial graph properties. It identifies "critical" bins
that must be collected (e.g., > 90% full) and extends the obligatory collection
set to include nearby bins that are moderately full (e.g., > 60% full). This
amortizes the routing cost to distant clusters.
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
    within a strict distance radius of those critical bins.
    """

    def __init__(
        self, critical_threshold: float = 0.90, synergy_threshold: float = 0.60, radius: float = 10.0, **kwargs
    ):
        """
        Initialize SpatialSynergySelection.

        Args:
            critical_threshold: Fill ratio at which a bin absolutely must be collected.
            synergy_threshold: Lower fill ratio for neighbors of critical bins.
            radius: Maximum distance to consider a bin a "neighbor".
        """
        self.critical_threshold = critical_threshold
        self.synergy_threshold = synergy_threshold
        self.radius = radius

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Select critical bins and their valid spatial synergies.

        Args:
            context: SelectionContext with fill levels and distance matrix.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        if context.distance_matrix is None:
            raise ValueError("SpatialSynergySelection requires a distance_matrix in the context.")

        fill_ratios = context.current_fill / context.max_fill

        # 1. Identify critically full bins (0-based indices)
        critical_indices = np.nonzero(fill_ratios > self.critical_threshold)[0]

        if len(critical_indices) == 0:
            return []

        # 2. Identify moderately full bins that are eligible for synergy
        moderate_indices = np.nonzero(fill_ratios > self.synergy_threshold)[0]

        must_go_set = set(critical_indices.tolist())

        # 3. For each critical bin, find moderately full neighbors
        # Note: distance_matrix includes depot at index 0. Bin i corresponds to index i+1 in distance_matrix.
        for crit_idx in critical_indices:
            dist_to_crit = context.distance_matrix[crit_idx + 1, 1:]  # Distances from crit_idx to all other bins

            # Find neighbors within radius
            neighbors = np.nonzero(dist_to_crit <= self.radius)[0]

            # Intersect spatial neighbors with moderately full bins
            synergy_bins = np.intersect1d(neighbors, moderate_indices)
            must_go_set.update(synergy_bins.tolist())

        # Convert back to 1-based IDs for routing
        return [i + 1 for i in must_go_set]
