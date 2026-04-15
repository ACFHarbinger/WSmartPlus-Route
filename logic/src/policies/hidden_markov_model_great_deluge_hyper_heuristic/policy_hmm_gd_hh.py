"""
HMM-GD Policy Adapter.

Adapts the HMM + Great Deluge (HMM-GD) hyper-heuristic solver to the
agnostic BaseRoutingPolicy interface.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hmm_gd_hh import HMMGDHHConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.hidden_markov_model_great_deluge_hyper_heuristic.params import HMMGDHHParams
from logic.src.policies.hidden_markov_model_great_deluge_hyper_heuristic.solver import HMMGDHHSolver
from logic.src.policies.other.operators.heuristics.greedy_initialization import build_greedy_routes


@PolicyRegistry.register("hmm_gd_hh")
class HMMGDHHPolicy(BaseRoutingPolicy):
    """
    HMM-GD-HH policy class.

    Visits bins using the online-learning HMM + Great Deluge hyper-heuristic.
    The HMM learns which Low-Level Heuristic to apply based on observed search
    states (improving / stagnating / escaping).  The Great Deluge criterion
    provides acceptance control without temperature parameters.
    """

    def __init__(self, config: Optional[Union[HMMGDHHConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HMMGDHHConfig

    def _get_config_key(self) -> str:
        return "hmm_gd_hh"

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
        params = HMMGDHHParams(
            max_iterations=int(values.get("max_iterations", 500)),
            flood_margin=float(values.get("flood_margin", 0.05)),
            rain_speed=float(values.get("rain_speed", 0.001)),
            learning_rate=float(values.get("learning_rate", 0.1)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = HMMGDHHSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        rng = random.Random(params.seed)
        heuristic_routes = build_greedy_routes(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            mandatory_nodes=mandatory_nodes,
            rng=rng,
        )

        routes, profit, cost = solver.solve(initial_routes=heuristic_routes)
        return routes, profit, cost
