"""
BCP Policy Adapter.
"""
from typing import Any, List, Tuple

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

        # BCP uses global IDs (1..N)
        # bins.c is 0-indexed, so we use idx-1.
        demands = {idx: bins.c[idx - 1] for idx in must_go}
        global_must_go = set(must_go)

        capacity = bcp_config.get("capacity", 100.0)
        revenue = bcp_config.get("revenue", 1.0)
        cost_unit = bcp_config.get("cost_unit", 1.0)
        values = {"Q": capacity, "R": revenue, "C": cost_unit}
        values.update(bcp_config)

        routes, cost = run_bcp(
            distance_matrix,
            demands,
            capacity,
            revenue,
            cost_unit,
            values,
            must_go_indices=global_must_go,
            env=kwargs.get("model_env"),
        )

        tour = [0]
        for r in routes:
            tour.extend(r)
            tour.append(0)

        return tour, get_route_cost(distance_matrix, tour), None
