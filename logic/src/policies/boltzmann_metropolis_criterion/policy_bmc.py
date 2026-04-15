"""
BMC (Boltzmann-Metropolis Criterion) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.bmc import BMCConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.boltzmann_metropolis_criterion.params import BMCParams
from logic.src.policies.boltzmann_metropolis_criterion.solver import BMCSolver


@PolicyRegistry.register("bmc")
class BMCPolicy(BaseRoutingPolicy):
    """Boltzmann-Metropolis Criterion policy class."""

    def __init__(self, config: Optional[Union[BMCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return BMCConfig

    def _get_config_key(self) -> str:
        return "bmc"

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
        params = BMCParams(
            initial_temp=float(values.get("initial_temp", 100.0)),
            alpha=float(values.get("alpha", 0.995)),
            min_temp=float(values.get("min_temp", 0.01)),
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = BMCSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
