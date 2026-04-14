"""
Cross-Exchange Post-Processor.
"""

from typing import Any, List

from logic.src.interfaces.post_processing import IPostProcessor

from .base import PostProcessorRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@PostProcessorRegistry.register("cross_exchange")
class CrossExchangePostProcessor(IPostProcessor):
    """
    Cross-exchange post-processor that swaps segments between different routes.
    Wraps LocalSearchManager.cross_exchange_op.
    """

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Apply cross-exchange improvement to the tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s).
            **kwargs: Context containing 'distance_matrix', 'cross_exchange_max_segment_len', 'iterations', etc.

        Returns:
            List[int]: Refined tour.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour

        # Parameters
        max_seg_len = kwargs.get("cross_exchange_max_segment_len", self.config.get("cross_exchange_max_segment_len", 3))
        iterations = kwargs.get("iterations", kwargs.get("max_iterations", self.config.get("iterations", 500)))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        R = kwargs.get("R", self.config.get("R", 1.0))
        C = kwargs.get("C", self.config.get("C", 1.0))

        # Convert distance matrix to numpy
        dm = to_numpy(distance_matrix)

        if len(tour) < 3:
            return tour

        try:
            from logic.src.policies.other.local_search.local_search_manager import LocalSearchManager

            routes = split_tour(tour)
            if len(routes) < 2:
                # Cross-exchange requires at least two routes
                return tour

            manager = LocalSearchManager(
                dist_matrix=dm,
                wastes=wastes,
                capacity=capacity,
                R=R,
                C=C,
                improvement_threshold=1e-6,
                seed=seed,
            )
            manager.set_routes(routes)

            for _ in range(iterations):
                if not manager.cross_exchange_op(max_seg_len=max_seg_len):
                    break

            return assemble_tour(manager.get_routes())

        except Exception:
            # Fallback to original tour on any unexpected error
            return tour
