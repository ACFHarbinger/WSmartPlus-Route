"""
HGS-ALNS Hybrid Policy Adapter.

Adapts the Hybrid HGS-ALNS solver to the common simulator policy interface.
"""
from typing import Any, List, Tuple

import numpy as np

from .adapters import IPolicy, PolicyRegistry
from .hgs_alns_solver import HGSALNSSolver
from .hgs_aux.types import HGSParams
from .single_vehicle import get_route_cost


@PolicyRegistry.register("hgs_alns")
class HGSALNSPolicy(IPolicy):
    """
    Hybrid HGS-ALNS policy class for the simulator.
    Uses ALNS for the intensive education phase of HGS.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the HGS-ALNS hybrid policy.
        """
        must_go = kwargs.get("must_go", [])
        if not must_go:
            return [0, 0], 0.0, None

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})

        # Merge HGS and Hybrid configs
        hgs_config = config.get("hgs", {})
        hybrid_config = config.get("hgs_alns", {})

        # 1. Prepare Data
        demands = {i + 1: bins.c[i] for i in range(bins.n)}

        # Subset mapping
        subset_indices = [0] + must_go
        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]
        sub_demands = {i: demands[orig_idx] for i, orig_idx in enumerate(must_go, 1)}

        capacity = hybrid_config.get("capacity", hgs_config.get("capacity", 100.0))
        revenue = hybrid_config.get("revenue", hgs_config.get("revenue", 1.0))
        cost_unit = hybrid_config.get("cost_unit", hgs_config.get("cost_unit", 1.0))

        params = HGSParams(
            time_limit=hybrid_config.get("time_limit", hgs_config.get("time_limit", 10)),
            population_size=hybrid_config.get("population_size", hgs_config.get("population_size", 50)),
            elite_size=hybrid_config.get("elite_size", hgs_config.get("elite_size", 10)),
            mutation_rate=hybrid_config.get("mutation_rate", hgs_config.get("mutation_rate", 0.2)),
            max_vehicles=hybrid_config.get("max_vehicles", hgs_config.get("max_vehicles", 0)),
        )

        alns_iter = hybrid_config.get("alns_education_iterations", 50)

        # 2. Run Hybrid Solver
        solver = HGSALNSSolver(
            dist_matrix=sub_dist_matrix,
            demands=sub_demands,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            alns_education_iterations=alns_iter,
        )

        best_routes, _, _ = solver.solve()

        # 3. Map routes back
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
