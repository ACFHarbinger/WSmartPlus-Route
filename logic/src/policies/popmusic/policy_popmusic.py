"""
Simulator adapter for the POPMUSIC matheuristic.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.popmusic import POPMUSICConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.popmusic.solver import run_popmusic


@PolicyRegistry.register("popmusic")
class POPMUSICPolicy(BaseRoutingPolicy):
    """
    Adapter for the POPMUSIC (Partial Optimization Metaheuristic Under Special
    Intensification Conditions) matheuristic.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return POPMUSICConfig

    def _get_config_key(self) -> str:
        return "popmusic"

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute POPMUSIC on the current collection problem.
        """
        cfg = self._parse_config(self.config, POPMUSICConfig)

        coords = kwargs["coords"]
        must_go = kwargs["must_go"]
        distance_matrix = kwargs["distance_matrix"]
        n_vehicles = kwargs.get("n_vehicles", 1)
        seed = cfg.seed if cfg.seed is not None else kwargs.get("seed", 42)

        # Environmental parameters for NN initialization
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        # Use standard multipliers for VRPP
        R = kwargs.get("R", 1.0)
        C = kwargs.get("C", 1.0)

        tour, cost, extra_data = run_popmusic(
            coords=coords,
            must_go=must_go,
            distance_matrix=distance_matrix,
            n_vehicles=n_vehicles,
            subproblem_size=cfg.subproblem_size,
            max_iterations=cfg.max_iterations,
            base_solver=cfg.base_solver,
            seed=seed,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
        )

        return tour, cost, extra_data
