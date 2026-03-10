"""
LKH (Lin-Kernighan-Helsgaun) Post-Processor.
"""

from typing import Any, List

import numpy as np
import torch

from logic.src.interfaces import IPostProcessor
from logic.src.policies.other.operators.heuristics.lin_kernighan_helsgaun import solve_lkh

from .registry import PostProcessorRegistry


@PostProcessorRegistry.register("lkh")
class LinKernighanHelsgaunPostProcessor(IPostProcessor):
    """
    Refines tours using the Lin-Kernighan-Helsgaun heuristic.

    Splits the tour into sub-tours by depot, re-optimizes each segment
    with LKH (2-opt / 3-opt + double-bridge kicks), and reconstructs.
    """

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Apply LKH refinement to each sub-tour.

        Args:
            tour: The initial tour to refine (list of node IDs).
            **kwargs: Must contain 'distance_matrix'; optionally 'wastes'
                and 'capacity' for VRP penalty evaluation.

        Returns:
            List[int]: The refined tour.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None:
            return tour

        if isinstance(distance_matrix, torch.Tensor):
            distance_matrix = distance_matrix.cpu().numpy()
        elif not isinstance(distance_matrix, np.ndarray):
            distance_matrix = np.array(distance_matrix)

        # Optional VRP parameters
        wastes_raw = kwargs.get("wastes")
        capacity = kwargs.get("capacity")
        waste_arr = None
        if wastes_raw is not None:
            if isinstance(wastes_raw, dict):
                n = len(distance_matrix)
                waste_arr = np.zeros(n)
                for idx, w in wastes_raw.items():
                    waste_arr[int(idx)] = w
            elif isinstance(wastes_raw, np.ndarray):
                waste_arr = wastes_raw
            else:
                waste_arr = np.array(wastes_raw)

        # Split tour into sub-tours (trips) at depot 0
        trips: List[List[int]] = []
        current_trip: List[int] = []
        for node in tour:
            if node == 0:
                if current_trip:
                    trips.append(current_trip)
                    current_trip = []
            else:
                current_trip.append(node)

        if not trips:
            return tour

        refined_tour = [0]
        for trip in trips:
            if len(trip) > 2:
                # Build closed sub-tour: [0, ...nodes..., 0]
                sub_tour = [0] + trip + [0]
                optimized, _ = solve_lkh(
                    distance_matrix,
                    initial_tour=sub_tour,
                    max_iterations=kwargs.get("n_iterations", self.max_iterations),
                    waste=waste_arr,
                    capacity=capacity,
                    np_rng=np.random.RandomState(kwargs.get("seed", 42)),
                )
                # Strip depot from result
                refined_tour.extend([n for n in optimized if n != 0])
            else:
                refined_tour.extend(trip)
            refined_tour.append(0)

        return refined_tour
