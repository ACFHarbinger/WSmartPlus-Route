"""
TS (Tabu Search) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ts import TSConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.tabu_search.params import TSParams
from logic.src.policies.tabu_search.solver import TSSolver


@PolicyRegistry.register("ts")
class TSPolicy(BaseRoutingPolicy):
    """Tabu Search policy class."""

    def __init__(self, config: Optional[Union[TSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return TSConfig

    def _get_config_key(self) -> str:
        return "ts"

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
        params = TSParams(
            # Short-term memory
            tabu_tenure=int(values.get("tabu_tenure", 7)),
            dynamic_tenure=bool(values.get("dynamic_tenure", True)),
            min_tenure=int(values.get("min_tenure", 5)),
            max_tenure=int(values.get("max_tenure", 15)),
            # Aspiration criteria
            aspiration_enabled=bool(values.get("aspiration_enabled", True)),
            # Long-term memory
            intensification_enabled=bool(values.get("intensification_enabled", True)),
            diversification_enabled=bool(values.get("diversification_enabled", True)),
            intensification_interval=int(values.get("intensification_interval", 100)),
            diversification_interval=int(values.get("diversification_interval", 200)),
            elite_size=int(values.get("elite_size", 5)),
            frequency_penalty_weight=float(values.get("frequency_penalty_weight", 0.1)),
            # Candidate list
            candidate_list_enabled=bool(values.get("candidate_list_enabled", True)),
            candidate_list_size=int(values.get("candidate_list_size", 20)),
            # Strategic oscillation
            oscillation_enabled=bool(values.get("oscillation_enabled", False)),
            feasibility_tolerance=float(values.get("feasibility_tolerance", 0.1)),
            # General search
            max_iterations=int(values.get("max_iterations", 5000)),
            max_iterations_no_improve=int(values.get("max_iterations_no_improve", 500)),
            n_removal=int(values.get("n_removal", 3)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
            # Neighborhood structure
            use_swap=bool(values.get("use_swap", True)),
            use_relocate=bool(values.get("use_relocate", True)),
            use_2opt=bool(values.get("use_2opt", True)),
            use_insertion=bool(values.get("use_insertion", True)),
        )

        solver = TSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
            seed=values.get("seed"),
        )

        return solver.solve()
