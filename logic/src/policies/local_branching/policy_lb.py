"""
Simulator adapter for the Local Branching (LB) matheuristic.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from ...configs.policies.lb import LocalBranchingConfig
from ..base.base_routing_policy import BaseRoutingPolicy
from ..base.factory import PolicyRegistry
from .solver import run_local_branching_gurobi


@PolicyRegistry.register("lb")
class LocalBranchingPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Local Branching (LB) matheuristic.

    Local Branching (Fischetti and Lodi, 2003) is a matheuristic framework that
    systematically improves an incumbent solution by exploring its
    k-neighborhood using a standard MILP solver.

    Relationship to Other Matheuristics:
    Unlike Kernel Search which decomposes variables by their contribution to the
    LP relaxation, Local Branching decomposes the SOLUTION SPACE into small,
    tractable neighborhoods centered around the current best solution.

    Algorithm Summary:
    1.  **Phase 1: Initialization**: Find an initial feasible solution (e.g.,
        by solving the full VRPP model with a short time limit).
    2.  **Phase 2: Iterative Search**:
        - Define a neighborhood by adding a 'Local Branching Cut'
          (Hamming distance constraint).
        - Solve the resulting restricted MIP.
        - If an improvement is found, update the incumbent and jump to the
          new neighborhood.
    3.  **Phase 3: Termination**: Stop if no improvement is found within the
        current neighborhood, or if the global time/iteration budget is reached.

    Attributes:
        config (Optional[Dict[str, Any]]): The raw Hydra configuration dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Local Branching policy with a specific configuration.

        Args:
            config (Optional[Dict[str, Any]]): A dictionary-like configuration object.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Return the configuration dataclass associated with this policy.

        Returns:
            Type: The `LocalBranchingConfig` class.
        """
        return LocalBranchingConfig

    def _get_config_key(self) -> str:
        """
        Return the unique Hydra configuration key for LB.

        Returns:
            str: The configuration key "lb".
        """
        return "lb"

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the Local Branching matheuristic on the current simulation state.

        This method extracts environment parameters and delegates the iterative
        neighborhood search to the specialized Gurobi-based LB solver.

        Args:
            **kwargs (Any): Simulation state including `distance_matrix`,
                `wastes`, `capacity`, and `must_go` nodes.

        Returns:
            Tuple[List[int], float, Any]: (tour, total_cost, metadata)
        """
        # 1. Parse configuration
        cfg = self._parse_config(self.config, LocalBranchingConfig)

        # 2. Extract state parameters
        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        mandatory_nodes = kwargs.get("must_go", [])
        R = kwargs.get("R", 1.0)
        C = kwargs.get("C", 1.0)
        seed = cfg.seed if cfg.seed is not None else kwargs.get("seed", 42)

        # 3. Call the core LB solver
        tour, obj_val, cost = run_local_branching_gurobi(
            dist_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            k=cfg.k,
            max_iterations=cfg.max_iterations,
            time_limit=cfg.time_limit,
            time_limit_per_iteration=cfg.time_limit_per_iteration,
            mip_limit_nodes=cfg.node_limit_per_iteration,
            mip_gap=cfg.mip_gap,
            seed=seed,
        )

        return tour, float(cost), {"obj_val": obj_val}
