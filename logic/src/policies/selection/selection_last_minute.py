from typing import List

import numpy as np

from ..must_go_selection import MustGoSelectionStrategy, SelectionContext


class LastMinuteSelection(MustGoSelectionStrategy):
    """
    Simple threshold-based reactive strategy.

    Logic: Collect if current_fill > threshold.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        must_go = np.nonzero(context.current_fill > context.threshold)[0] + 1
        return must_go.tolist()


class LastMinuteAndPathSelection(MustGoSelectionStrategy):
    """
    Threshold-based strategy with opportunistic path-based expansion.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        # 1. Identify critical bins (exceeding threshold)
        critical_bins = np.nonzero(context.current_fill > context.threshold)[0] + 1

        if len(critical_bins) == 0:
            return []

        if context.paths_between_states is None or context.distance_matrix is None:
            return critical_bins.tolist()

        from ..single_vehicle import find_route

        try:
            dist_int = np.round(context.distance_matrix).astype(int)
            tour = find_route(dist_int, critical_bins)
        except Exception:
            return critical_bins.tolist()

        selected_bins = set(critical_bins.tolist())
        total_waste = np.sum(context.current_fill[critical_bins - 1])

        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            path = context.paths_between_states[u][v]
            for node in path:
                if node == 0:
                    continue
                if node not in selected_bins:
                    waste = context.current_fill[node - 1]
                    if waste + total_waste <= context.vehicle_capacity:
                        total_waste += waste
                        selected_bins.add(node)

        return list(selected_bins)
