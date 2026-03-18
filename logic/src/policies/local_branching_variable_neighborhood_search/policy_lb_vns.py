"""
Simulator adapter for the Local Branching with Variable Neighborhood Search (LB-VNS).

This policy adapts the LB-VNS matheuristic (Hansen et al., 2006) for use within
the WSmart+ Route simulation framework, extracting problem state and
orchestrating the Gurobi optimization process.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from ...configs.policies.lb_vns import LocalBranchingVNSConfig
from ..base.base_routing_policy import BaseRoutingPolicy
from ..base.factory import PolicyRegistry
from .lb_vns import run_lb_vns_gurobi


@PolicyRegistry.register("lb_vns")
class LocalBranchingVNSPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Local Branching with Variable Neighborhood Search (LB-VNS).

    This policy implements the high-level integration required to run the LB-VNS
    matheuristic within the WSmart+ Route simulator. It handles:
    1. Configuration parsing and dataclass conversion.
    2. State extraction (distance matrices, node profits, capacities).
    3. Orchestration of the Local Branching intensification and VNS shaking phases.
    4. Solution reconstruction and coordinate-to-node mapping.

    Technical Context:
        LB-VNS is designed for hard combinatorial problems where standard local search
        fails to reach global minima. By systematically increasing the radius of the
        'restricted' MILP search space, it avoids entrapment in sub-optimal attraction basins.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LB-VNS policy with a provided or default configuration.

        Args:
            config: Optional configuration dictionary containing hyperparameters for
                the neighborhood sequence (k_min, k_max, k_step) and solver limits.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Return the structured configuration dataclass for LB-VNS.

        Returns:
            Type[LocalBranchingVNSConfig]: The dataclass defining LB-VNS hyperparameters.
        """
        return LocalBranchingVNSConfig

    def _get_config_key(self) -> str:
        """
        Return the top-level YAML key used for this policy's configuration.

        Returns:
            str: "lb_vns"
        """
        return "lb_vns"

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
        """Not used - LB-VNS requires specialized execute()."""
        return [], 0.0, 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Extract the current problem state and execute the LB-VNS metaheuristic.

        This method bridges the simulation state (numpy arrays and dictionaries)
        with the gurobipy-based mathematical model.

        Parameters (passed via kwargs):
            distance_matrix (np.ndarray): NxN symmetric matrix representing travel costs.
            wastes (Dict[int, float]): Dictionary mapping node indices to expected
                collection profits (represented as waste weight).
            capacity (float): Maximum weight the vehicle can carry in a single tour.
            must_go (List[int]): List of indices for 'mandatory' nodes that must be visited.
            R (float): Revenue multiplier for the objective function.
            C (float): Cost multiplier (distance) for the objective function.
            seed (int): Optional seed override for reproducibility.

        Returns:
            Tuple[List[int], float, Dict[str, Any]]:
                - List[int]: The optimized tour sequence starting and ending at the depot (0).
                - float: The total travel cost of the reconstructed tour.
                - Dict[str, Any]: Metadata containing the final mathematical objective value.
        """
        cfg = self._parse_config(self.config, LocalBranchingVNSConfig)

        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        mandatory_nodes = kwargs.get("must_go", [])
        R = kwargs.get("R", 1.0)
        C = kwargs.get("C", 1.0)
        seed = cfg.seed if cfg.seed is not None else kwargs.get("seed", 42)

        # Delegate the actual optimization loop to the mathematical solver core.
        # This keeps the policy adapter thin and decoupled from Gurobi specificities.
        tour, obj_val, cost = run_lb_vns_gurobi(
            dist_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            k_min=cfg.k_min,
            k_max=cfg.k_max,
            k_step=cfg.k_step,
            time_limit=cfg.time_limit,
            time_limit_per_lb=cfg.time_limit_per_lb,
            max_lb_iterations=cfg.max_lb_iterations,
            mip_gap=cfg.mip_gap,
            seed=seed,
        )

        return tour, float(cost), {"obj_val": obj_val}
