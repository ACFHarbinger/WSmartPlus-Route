"""
Simulator adapter for the Relaxation Enforced Neighborhood Search (RENS) matheuristic.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.rens import RENSConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.rens.solver import run_rens_gurobi


@PolicyRegistry.register("rens")
class RENSPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Relaxation Enforced Neighborhood Search (RENS) matheuristic.

    RENS (Berthold, 2009) is a start heuristic for Mixed-Integer Programming (MIP).
    In the context of WSmart+ Route, it solves a VRPP/WCVRP instance by:
    1. Solving the initial LP relaxation.
    2. Fixing variables that are integer in the LP solution.
    3. Solving a sub-MIP on the remaining fractional binary variables.

    This policy belongs to the 'matheuristic' category and requires a mathematical
    modeling engine (Gurobi).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RENS policy with a configuration dictionary.

        Args:
            config: Hydra configuration for RENS.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration dataclass associated with this policy."""
        return RENSConfig

    def _get_config_key(self) -> str:
        """Return the unique Hydra configuration key for RENS."""
        return "rens"

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the RENS matheuristic on the current state.

        Extracts environmental variables (distance matrix, waste levels, capacity)
        and invokes the Gurobi-based RENS solver.

        Args:
            **kwargs: Cumulative simulation state including:
                - distance_matrix (np.ndarray): Cost matrix between nodes.
                - wastes (Dict[int, float]): Current waste levels for each bin.
                - capacity (float): Vehicle volume limit.
                - must_go (List[int]): Optional nodes requiring collection.
                - R (float): Revenue multiplier.
                - C (float): Cost multiplier.

        Returns:
            A tuple of (tour, total_travel_cost, extra_data_dict).
        """
        cfg = self._parse_config(self.config, RENSConfig)

        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        mandatory_nodes = kwargs.get("must_go", [])

        # Multipliers
        R = kwargs.get("R", 1.0)
        C = kwargs.get("C", 1.0)

        seed = cfg.seed if cfg.seed is not None else kwargs.get("seed", 42)

        tour, obj_val, cost = run_rens_gurobi(
            dist_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            time_limit=cfg.time_limit,
            lp_time_limit=cfg.lp_time_limit,
            mip_gap=cfg.mip_gap,
            seed=seed,
        )

        return tour, float(cost), {"obj_val": obj_val}
