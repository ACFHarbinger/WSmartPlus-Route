"""
Cluster-First Route-Second (CF-RS) Policy Adapter.

Adapts the classic geometric decomposition algorithm (Fisher & Jaikumar, 1981)
to the WSmart+ Route simulator interface. This adapter facilitates the
integration of the solver into the simulation pipeline, handling parameter
mapping, data extraction, and result formatting.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from logic.src.configs.policies.cf_rs import CFRSConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.cluster_first_route_second.solver import run_cf_rs


@PolicyRegistry.register("cf_rs")
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
        - Registered under the key 'cf_rs' for dynamic instantiation.
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
            The string key 'cf_rs'.
        """
        return "cf_rs"

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
        # 1. Parse policy-specific config
        cfg = self._parse_config(values, CFRSConfig)

        # 2. Extract coordinates and fleet size
        coords = kwargs.get("coords")
        if coords is None:
            # Fallback to 'all_coords' if 'coords' is missing (simulators vary)
            coords = kwargs.get("all_coords")

        subset_indices = kwargs.get("subset_indices", list(range(sub_dist_matrix.shape[0])))
        n_vehicles = kwargs.get("n_vehicles") or 1

        if not mandatory_nodes:
            # Return an empty list of routes, 0 profit, 0 cost
            return [], 0.0, 0.0

        # 3. Subset coordinates matching the search space (Depot + Bins)
        # Handle both DataFrame (iloc) and Array-like
        sub_coords = coords.iloc[subset_indices].values if hasattr(coords, "iloc") else np.array(coords)[subset_indices]

        if sub_coords is None:
            return [], 0.0, 0.0

        # 4. Priority-aware seeding
        seed = cfg.seed if cfg.seed is not None else kwargs.get("seed", 42)

        # 5. Run core algorithm (VFJ Algorithm)
        routes, solver_cost, extra_data = run_cf_rs(
            coords=sub_coords,
            must_go=mandatory_nodes,
            distance_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            n_vehicles=n_vehicles,
            seed=seed,
            num_clusters=cfg.num_clusters,
            time_limit=cfg.time_limit,
            assignment_method=cfg.assignment_method,
            route_optimizer=cfg.route_optimizer,
            strict_fleet=cfg.strict_fleet,
            seed_criterion=cfg.seed_criterion,
            mip_objective=cfg.mip_objective,
        )

        # 7. Final Profit Calculation (Revenue - Distance Cost)
        total_fill = sum(sub_wastes.get(n, 0.0) for r in routes for n in r)
        total_profit = (total_fill * revenue) - solver_cost

        return routes, total_profit, solver_cost
