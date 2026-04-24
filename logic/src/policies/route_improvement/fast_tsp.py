"""Fast TSP Solver for route reoptimization.

Provides a lightweight TSP implementation suitable for recurring
reoptimization in inner loops.

Attributes:
    FastTSPRouteImprover: Main class for fast TSP solving.
"""

from typing import Any, List, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import (
    find_route,
)

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@GlobalRegistry.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.HEURISTIC,
)
@RouteImproverRegistry.register("fast_tsp")
class FastTSPRouteImprover(IRouteImprovement):
    """Fast TSP route improver.

    Refines all sub-tours by splitting the tour by depot (0),
    re-optimizing each segment, and reconstructing the final tour.

    Attributes:
        config (Dict[str, Any]): Internal configuration state.
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """Apply fast TSP reoptimization to each route in the tour.

        Args:
            tour (List[int]): Initial tour sequence.
            **kwargs: Context containing:
                distance_matrix: Distance lookup (np.ndarray).
                fast_tsp_iterations: Heuristic iteration count (default 100).
                wastes: Bin mass dictionary.
                capacity: Vehicle capacity.

        Returns:
            Tuple[List[int], ImprovementMetrics]: (refined_tour, metrics).
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        dm = to_numpy(distance_matrix)

        routes = split_tour(tour)
        if not routes:
            return tour, {"algorithm": "FastTSPRouteImprover"}

        refined_routes = []
        for trip in routes:
            if len(trip) > 1:
                # Re-optimize with fast_tsp
                time_limit = kwargs.get("time_limit", self.config.get("time_limit", 2.0))
                seed = kwargs.get("seed", self.config.get("seed", 42))
                refined_trip = find_route(
                    dm,
                    np.array(trip),
                    time_limit=time_limit,
                    seed=seed,
                )
                # strip depot 0s from the constructed route to avoid doubles
                refined_routes.append([n for n in refined_trip if n != 0])
            else:
                refined_routes.append(trip)

        return assemble_tour(refined_routes), {"algorithm": "FastTSPRouteImprover"}
