"""
SWC-TCF (Smart Waste Collection - Two-Commodity Flow) Policy Wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import SWCTCFConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .dispatcher import run_swc_tcf_optimizer


@PolicyRegistry.register("swc_tcf")
class SWCTCFPolicy(BaseRoutingPolicy):
    """
    Agnostic SWC-TCF Policy adapter.
    Delegates to run_swc_tcf_optimizer.
    """

    def __init__(self, config: Optional[Union[SWCTCFConfig, Dict[str, Any]]] = None):
        """Initialize SWC-TCF policy with optional config.

        Args:
            config: SWCTCFConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SWCTCFConfig

    def _get_config_key(self) -> str:
        return "swc_tcf"

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
        """Not used - SWC-TCF requires specialized execute()."""
        return [], 0.0, 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the SWC-TCF policy.
        """
        # 1. Extract context
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        bins = kwargs.get("bins")
        distance_matrix = kwargs.get("distance_matrix")
        must_go = kwargs.get("must_go", [])
        config = kwargs.get("config", {})

        # 2. Load parameters and merge with config
        _, _, _, values = self._load_area_params(area, waste_type, config)
        self._log_solver_params(values, kwargs)

        # 3. Handle bins input (WSmart+ simulator passes bins object or amounts)
        amounts = bins
        if bins is not None and hasattr(bins, "c"):
            amounts = bins.c

        # Ensure binsids are present (local 1-based for solver consistency)
        n_bins = len(amounts) if amounts is not None else 0
        binsids = list(range(1, n_bins + 1))

        # 4. Get solver parameters from typed config or values dict
        cfg = self._config
        seed = kwargs.get("seed") if kwargs.get("seed") is not None else 42
        time_limit = int(cfg.time_limit) if cfg is not None else int(values.get("time_limit", 60))
        optimizer = cfg.engine if cfg is not None else values.get("engine", "gurobi")

        # 5. Run optimizer
        route, profit, cost = run_swc_tcf_optimizer(
            bins=amounts,  # type: ignore[arg-type]
            distance_matrix=distance_matrix,  # type: ignore[arg-type]
            values=values,
            binsids=binsids,
            must_go=must_go,
            number_vehicles=kwargs.get("number_vehicles", 1),
            time_limit=time_limit,
            optimizer=optimizer,
            seed=seed,
        )

        return route, cost, {"profit": profit}
