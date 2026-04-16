"""
Cluster-First Route-Second (CF-RS) Policy Adapter.

Adapts the classic geometric decomposition algorithm (Fisher & Jaikumar, 1981)
to the WSmart+ Route simulator interface. This adapter facilitates the
integration of the solver into the simulation pipeline, handling parameter
mapping, data extraction, and result formatting.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.cf_rs import CFRSConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import RouteConstructorRegistry

from .params import CFRSParams
from .solver import run_cf_rs


@RouteConstructorRegistry.register("cf_rs")
class ClusterFirstRouteSecondPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Cluster-First Route-Second (CFRS) policy.

    The CFRS approach of Vidal, Freitas, and Junior (VFJ) decomposes the
    Vehicle Routing Problem with Profits (VRPP) into:
    1.  **Clustering Phase**: Partitioning nodes into clusters that are
        likely to be feasible within a single route.
    2.  **Routing Phase**: Solving a Traveling Salesman Problem (TSP)
        or VRP for each cluster to find the optimal sequence.

    This policy uses a variable-size clustering based on node proximity and
    profit-to-distance ratios, followed by local search to refine the clusters.

    Attributes:
        config (Optional[Dict[str, Any]]): The raw Hydra configuration dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return CFRSConfig

    def _get_config_key(self) -> str:
        return "cf_rs"

    def _run_solver(
        self,
        sub_dist_matrix: Any,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Not used - CFRS requires specialized execute()."""
        return [], 0.0, 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the CFRS policy.
        """
        # 1. Initialize type-safe Params
        params = CFRSParams.from_config(self._config or kwargs.get("config", {}).get("cf_rs", {}))

        # 2. Extract state parameters
        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        mandatory_nodes = kwargs.get("mandatory", [])
        revenue = kwargs.get("R", 1.0)
        cost_unit = kwargs.get("C", 1.0)
        n_vehicles = kwargs.get("number_vehicles", 1)
        sub_coords = kwargs.get("coords", [])
        seed = params.seed if params.seed is not None else kwargs.get("seed", 42)

        num_clusters = int(params.num_clusters or 1)

        # 4. Run CFRS
        routes, solver_cost, extra_data = run_cf_rs(
            coords=sub_coords,
            mandatory=mandatory_nodes,
            distance_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            n_vehicles=n_vehicles,
            seed=seed,
            num_clusters=num_clusters,
            time_limit=params.time_limit,
            assignment_method=params.assignment_method,
            route_optimizer=params.route_optimizer,
            strict_fleet=params.strict_fleet,
            seed_criterion=params.seed_criterion,
            mip_objective=params.mip_objective,
        )

        # Flatten routes for consistent output if necessary
        tour = [node for route in routes for node in route]

        return tour, float(solver_cost), extra_data
