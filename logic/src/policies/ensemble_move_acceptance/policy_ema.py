"""
Ensemble Move Acceptance (EMA) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ema import EMAConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import EMAParams
from .solver import EMASolver


@PolicyRegistry.register("ema")
class EnsembleMoveAcceptancePolicy(BaseRoutingPolicy):
    """
    Adapter for the Ensemble Move Acceptance (EMA) solver.
    """

    def __init__(self, config: Optional[Union[EMAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return EMAConfig

    def _get_config_key(self) -> str:
        return "ema"

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
        params = EMAParams(
            max_iterations=values.get("max_iterations", 1000),
            rule=values.get("rule", "G-VOT"),
            criteria=values.get("criteria", ["sa", "gd", "ie"]),
            sub_params=values.get("sub_params", {}),
            time_limit=values.get("time_limit", 60.0),
            n_removal=values.get("n_removal", 2),
            n_llh=values.get("n_llh", 5),
        )
        solver = EMASolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
            seed=values.get("seed"),
        )
        return solver.solve()
