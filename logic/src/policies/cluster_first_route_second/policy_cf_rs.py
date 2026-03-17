"""
Cluster-First Route-Second (CF-RS) Policy Adapter.

Adapts the classic geometric decomposition algorithm (Fisher & Jaikumar, 1981)
to the WSmart+ Route simulator interface. This adapter facilitates the
integration of the solver into the simulation pipeline, handling parameter
mapping, data extraction, and result formatting.

Pattern:
    The adapter follows the 'Policy Adapter' pattern, acting as a bridge
    between Hydra-managed configurations and the core routing logic.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from logic.src.configs.policies.cf_rs import CFRSConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.cluster_first_route_second.solver import run_cf_rs


@PolicyRegistry.register("cluster_first_route_second")
class ClusterFirstRouteSecondPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Cluster-First Route-Second (CF-RS) routing algorithm.

    This class encapsulates the execution of an angular partitioning strategy for
    waste collection routing. It manages:
    1. Parsing policy-specific configurations (CFRSConfig).
    2. Extracting geographic coordinates and bin indices from the simulation context.
    3. Delegating the core optimization to the `run_cf_rs` solver.
    4. Mapping the resulting multi-cluster tour back to the simulator's expectations.

    Architecture:
        - Inherits from `BaseRoutingPolicy` for standardized parameter loading.
        - Registered under the key 'cluster_first_route_second' for dynamic instantiation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CF-RS policy adapter.

        Args:
            config: A configuration dictionary typically sourced from Hydra YAML files.
                If None, default parameters for `CFRSConfig` will be used.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Reference the configuration dataclass associated with this policy.

        Returns:
            The `CFRSConfig` class for validation and structured loading.
        """
        return CFRSConfig

    def _get_config_key(self) -> str:
        """
        Identify the configuration key in the root YAML structure.

        Returns:
            The string key 'cluster_first_route_second'.
        """
        return "cluster_first_route_second"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run the Cluster-First Route-Second solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        cfg = self._parse_config(values, CFRSConfig)
        coords = kwargs["coords"]
        subset_indices = kwargs.get("subset_indices", list(range(sub_dist_matrix.shape[0])))
        n_vehicles = kwargs.get("n_vehicles", 1)

        # Subset coordinates matching the sub_dist_matrix nodes
        sub_coords = coords.iloc[subset_indices]

        # Priority-aware seeding (Config seed > Simulation seed)
        seed = cfg.seed if cfg.seed is not None else kwargs.get("seed", 42)

        # Run core algorithm.
        tour, solver_cost, extra_data = run_cf_rs(
            coords=sub_coords,
            must_go=mandatory_nodes,
            distance_matrix=sub_dist_matrix,
            n_vehicles=n_vehicles,
            seed=seed,
            num_clusters=cfg.num_clusters,
        )

        # extra_data contains 'clusters' (list of lists of local indices)
        # However, run_cf_rs already solved the TSPs.
        # To reuse _map_tour_to_global, we need to return segments between 0s as routes.
        routes = []
        current_route: List[int] = []
        for node in tour:
            if node == 0:
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(node)

        # Profit is not directly returned by run_cf_rs (it returns cost),
        # but BaseRoutingPolicy computes profit based on collected waste.
        # We need to return a profit estimate for the solver return.
        total_profit = sum(sub_wastes.get(n, 0.0) for r in routes for n in r) * revenue - solver_cost

        return routes, total_profit, solver_cost
