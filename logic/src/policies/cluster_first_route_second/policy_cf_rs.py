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

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the CF-RS algorithm within a simulation day context.

        This method acts as the primary entry point for the simulator. It extracts
        environmental data (coordinates, selected nodes, distances) and executes
         the angular clustering logic.

        Args:
            **kwargs: Flexible keyword arguments representing the simulation state:
                - coords (pd.DataFrame): Geographical coordinates of all nodes.
                    Must contain 'Lat' and 'Lng' columns.
                - must_go (List[int]): Global indices of bins designated for collection.
                - distance_matrix (np.ndarray): Full distance matrix between nodes.
                - n_vehicles (int): Number of available vehicles for routing.
                - seed (int, optional): Global simulation seed for deterministic runs.

        Returns:
            A tuple of (tour, cost, extra_data):
                - tour (List[int]): A sequence of node indices representing the optimized collection route.
                - cost (float): Total routing distance computed from the tour.
                - extra_data (Any): Secondary outputs (e.g., sector visual data or cluster mappings).
        """
        # Load and validate structured configuration
        cfg = self._parse_config(self.config, CFRSConfig)

        # Extract environment state from simulation context
        coords = kwargs["coords"]
        must_go = kwargs["must_go"]
        distance_matrix = kwargs["distance_matrix"]
        n_vehicles = kwargs.get("n_vehicles", 1)

        # Priority-aware seeding (Config seed > Simulation seed)
        seed = cfg.seed if cfg.seed is not None else kwargs.get("seed", 42)

        # Delegate execution to the core algorithm logic
        tour, cost, extra_data = run_cf_rs(
            coords=coords,
            must_go=must_go,
            distance_matrix=distance_matrix,
            n_vehicles=n_vehicles,
            seed=seed,
            num_clusters=cfg.num_clusters,
        )

        return tour, cost, extra_data
