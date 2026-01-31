"""
BCP Policy Adapter.
"""
from typing import Any, List, Tuple

import numpy as np

from .adapters import IPolicy, PolicyRegistry
from .branch_cut_and_price import run_bcp
from .single_vehicle import get_route_cost


@PolicyRegistry.register("bcp")
class BCPPolicy(IPolicy):
    """
    Branch-Cut-and-Price policy class.
    Visits pre-selected 'must_go' bins.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the BCP policy.
        """
        must_go = kwargs.get("must_go", [])
        if not must_go:
            return [0, 0], 0.0, None

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})
        bcp_config = config.get("bcp", {})
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")

        # Load Area Parameters
        from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params

        Q, R, _, C, _ = load_area_and_waste_type_params(area, waste_type)

        # Override with config if present
        capacity = bcp_config.get("capacity", Q)
        revenue = bcp_config.get("revenue", R)
        cost_unit = bcp_config.get("cost_unit", C)

        values = {"Q": capacity, "R": revenue, "C": cost_unit}
        values.update(bcp_config)

        # Prepare Data for BCP
        # must_go contains 1-based IDs. We visit ALL these bins (matching HGS strategy).
        subset_indices = [0] + list(must_go)
        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        # Build local demands and mark all as must-go for the solver
        sub_demands = {}
        # Solver expects nodes 1..M for the subset
        must_go_subset = set()
        for i, global_idx in enumerate(must_go, 1):
            fill = bins.c[global_idx - 1]
            sub_demands[i] = float(fill)
            must_go_subset.add(i)

        # 2. Run BCP Solver
        # Note: run_bcp handles OR-Tools/Gurobi/VRPy
        # We pass the subset and treat all as must-go (PC-CVRP with high penalty)
        best_routes, solver_cost = run_bcp(
            sub_dist_matrix,
            sub_demands,
            capacity,
            revenue,
            cost_unit,
            values,
            must_go_indices=must_go_subset,
            env=kwargs.get("model_env"),
        )

        # 3. Map back to global IDs
        tour = [0]
        if best_routes:
            for route in best_routes:
                for node_idx in route:
                    tour.append(subset_indices[node_idx])
                tour.append(0)

        if len(tour) == 1:
            tour = [0, 0]

        return tour, get_route_cost(distance_matrix, tour), None
