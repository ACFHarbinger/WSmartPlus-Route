"""
HGS Policy Adapter.

Adapts the Hybrid Genetic Search (HGS) logic to the common policy interface.
Now agnostic to bin selection.
"""
from typing import Any, List, Tuple

import numpy as np

from .adapters import IPolicy, PolicyRegistry
from .hybrid_genetic_search import run_hgs
from .single_vehicle import get_route_cost


@PolicyRegistry.register("hgs")
class HGSPolicy(IPolicy):
    """
    Hybrid Genetic Search policy class.
    Visits pre-selected 'must_go' bins.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the HGS policy.
        """
        must_go = kwargs.get("must_go", [])
        if not must_go:
            return [0, 0], 0.0, None

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})
        hgs_config = config.get("hgs", {})

        # 1. Prepare Data for HGS
        # indices in must_go are 1-based bin IDs (1..N)
        # Bins.c is 0-indexed (0..N-1), so we use idx-1 to access it.
        demands = {i + 1: bins.c[i] for i in range(bins.n)}

        capacity = hgs_config.get("capacity", 100.0)
        revenue = hgs_config.get("revenue", 1.0)
        cost_unit = hgs_config.get("cost_unit", 1.0)

        # Subset mapping (nodes to visit: depot + must_go)
        # must_go contains 1-based IDs
        subset_indices = [0] + must_go

        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        # demands for subset nodes
        sub_demands = {i: demands[orig_idx] for i, orig_idx in enumerate(must_go, 1)}

        # Run HGS
        best_routes, _, _ = run_hgs(sub_dist_matrix, sub_demands, capacity, revenue, cost_unit, hgs_config)

        # Map routes back
        tour = [0]
        if best_routes:
            for route in best_routes:
                for node_idx in route:
                    original_matrix_idx = subset_indices[node_idx]
                    tour.append(original_matrix_idx)
                tour.append(0)

        if len(tour) == 1:
            tour = [0, 0]

        cost = get_route_cost(distance_matrix, tour)
        return tour, cost, None
