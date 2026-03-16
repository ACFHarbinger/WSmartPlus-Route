"""
Simulator adapter for the Adaptive Kernel Search (AKS) matheuristic.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.aks import AdaptiveKernelSearchConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .solver import run_adaptive_kernel_search_gurobi


@PolicyRegistry.register("aks")
class AdaptiveKernelSearchPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Adaptive Kernel Search (AKS) matheuristic.

    Adaptive Kernel Search (Guastaroba et al., 2017) is an advanced matheuristic
    framework designed to solve complex Mixed-Integer Linear Programming (MILP)
    problems by decomposing them into a sequence of adaptive, more tractable
    restricted sub-problems.

    Relationship to Basic Kernel Search:
    While basic Kernel Search relies on a static decomposition into a fixed
    Kernel and steady-sized buckets, AKS introduces a dynamic, self-tuning
    loop that reacts to objective improvements by expanding the search
    neighborhood (buckets) and permanently promoting variables to the Kernel.

    Algorithm Summary:
    1.  **Selection Phase**: Computes LP relaxation of the VRPP model to obtain
        profitability scores for all variables (nodes and edges).
    2.  **Kernel Solve**: Solves an initial MIP on the most promising variables
        (the 'Kernel') to find a high-quality baseline solution.
    3.  **Adaptive Improvement Loop**: Iteratively selects 'Buckets' of
        remaining variables and solves sub-MIPs. If an improvement is found:
        - Variables that take positive values in the solution are promoted to
          the Kernel.
        - The bucket size for the next step is increased (scaled by growth factor).
    4.  **Final Solve**: Performs a final optimization over the final-state
        promoted Kernel to ensure global feasibility and local optimality.

    Attributes:
        config (Optional[Dict[str, Any]]): The raw Hydra configuration dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Adaptive Kernel Search policy with a specific configuration.

        Args:
            config (Optional[Dict[str, Any]]): A dictionary-like configuration object,
                typically provided by Hydra. If None, default settings are used.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Return the configuration dataclass associated with this policy.

        This allows the simulator to perform type checking and validation on the
        provided configuration parameters.

        Returns:
            Type: The `AdaptiveKernelSearchConfig` class.
        """
        return AdaptiveKernelSearchConfig

    def _get_config_key(self) -> str:
        """
        Return the unique Hydra configuration key for AKS.

        This key is used to look up the algorithm-specific parameters in the
        Hydra configuration hierarchy.

        Returns:
            str: The configuration key "aks".
        """
        return "aks"

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
        """Not used - AKS requires specialized execute()."""
        return [], 0.0, 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the AKS matheuristic on the provided simulation state.

        This method extracts relevant environment data (distance matrices, waste levels,
        vehicle capacity) and delegates the adaptive optimization loop to the
        specialized Gurobi solver.

        Args:
            **kwargs (Any): A dictionary containing the current simulation state.
                Required keys:
                    - `distance_matrix` (np.ndarray): Cost matrix for node transitions.
                Optional keys:
                    - `wastes` (Dict[int, float]): Current waste levels per node.
                    - `capacity` (float): Maximum vehicle load.
                    - `must_go` (List[int]): IDs of nodes that must be visited.
                    - `R` (float): Revenue multiplier for node collection.
                    - `C` (float): Travel cost multiplier.
                    - `seed` (int): Random seed for reproducibility.

        Returns:
            Tuple[List[int], float, Any]: A 3-tuple containing:
                - `tour` (List[int]): The ordered sequence of visited nodes.
                - `cost` (float): The total travel cost of the resulting tour.
                - `extra_data` (Dict[str, Any]): Metadata including the MILP objective value.
        """
        # 1. Parse and validate the configuration
        cfg = self._parse_config(self.config, AdaptiveKernelSearchConfig)

        # 2. Extract environment parameters from the simulation state
        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        mandatory_nodes = kwargs.get("must_go", [])

        # 3. Handle objective multipliers (Revenue and Cost weights)
        R = kwargs.get("R", 1.0)
        C = kwargs.get("C", 1.0)

        # 4. Enforce deterministic behavior if a seed is provided
        seed = cfg.seed if cfg.seed is not None else kwargs.get("seed", 42)

        # 5. Call the core standalone AKS solver
        tour, obj_val, cost = run_adaptive_kernel_search_gurobi(
            dist_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            initial_kernel_size=cfg.initial_kernel_size,
            bucket_size=cfg.bucket_size,
            max_buckets=cfg.max_buckets,
            bucket_growth_factor=cfg.bucket_growth_factor,
            time_limit=cfg.time_limit,
            mip_limit_nodes=cfg.mip_limit_nodes,
            mip_gap=cfg.mip_gap,
            seed=seed,
        )

        # 6. Return standardized results
        return tour, float(cost), {"obj_val": obj_val}
