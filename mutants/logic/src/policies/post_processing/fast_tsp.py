"""
Fast TSP Refinement Post-Processor.
"""

from typing import Any, List

import numpy as np
import torch
from logic.src.interfaces import IPostProcessor

from ..single_vehicle import find_route
from .registry import PostProcessorRegistry


@PostProcessorRegistry.register("fast_tsp")
class FastTSPPostProcessor(IPostProcessor):
    """
    Refines all sub-tours using the fast_tsp library.
    Splits long tours by depot (0), re-optimizes each segment, and reconstructs.
    """

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Refine the tour by splitting it into trips and optimizing each with fast_tsp.

        Args:
            tour: The initial tour to refine (list of node IDs).
            **kwargs: Keyword arguments containing 'distance_matrix'.

        Returns:
            List[int]: The optimized tour with reduced total distance.
        """
        distance_matrix = kwargs.get("distance_matrix")
        if distance_matrix is None:
            return tour

        if isinstance(distance_matrix, torch.Tensor):
            distance_matrix = distance_matrix.cpu().numpy()

        # Split tour into sub-tours (trips)
        trips = []
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
            if len(trip) > 1:
                # Re-optimize with fast_tsp
                refined_trip = find_route(distance_matrix, np.array(trip))
                # strip depot 0s from the constructed route to avoid doubles
                refined_tour.extend([n for n in refined_trip if n != 0])
            else:
                refined_tour.extend(trip)
            refined_tour.append(0)

        return refined_tour
