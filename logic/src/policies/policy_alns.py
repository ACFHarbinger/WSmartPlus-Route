"""
ALNS Policy Adapter.

Adapts the Adaptive Large Neighborhood Search (ALNS) logic to the agnostic interface.
"""
from typing import Any, List, Tuple

from .adapters import IPolicy, PolicyRegistry
from .adaptive_large_neighborhood_search import run_alns, run_alns_ortools, run_alns_package
from .single_vehicle import get_route_cost


@PolicyRegistry.register("alns")
class ALNSPolicy(IPolicy):
    """
    ALNS policy class.
    Visits pre-selected 'must_go' bins.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the ALNS policy.
        """
        must_go = kwargs.get("must_go", [])
        if not must_go:
            return [0, 0], 0.0, None

        policy_name = kwargs.get("policy", "alns")
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})
        alns_config = config.get("alns", {})

        # Prepare demands (0-based indices in must_go)
        # ALNS uses {global_id: weight} where global_id is typically 1..N
        demands = {idx + 1: bins.c[idx] for idx in must_go}

        capacity = alns_config.get("capacity", 100.0)
        revenue = alns_config.get("revenue", 1.0)
        cost_unit = alns_config.get("cost_unit", 1.0)
        values = {"Q": capacity, "R": revenue, "C": cost_unit}
        values.update(alns_config)

        # Dispatch variant
        if "package" in policy_name:
            routes, _, _ = run_alns_package(distance_matrix, demands, capacity, revenue, cost_unit, values)
        elif "ortools" in policy_name:
            routes, _, _ = run_alns_ortools(distance_matrix, demands, capacity, revenue, cost_unit, values)
        else:
            routes, _, _ = run_alns(distance_matrix, demands, capacity, revenue, cost_unit, values)

        tour = [0]
        for r in routes:
            tour.extend(r)
            tour.append(0)

        return tour, get_route_cost(distance_matrix, tour), None
