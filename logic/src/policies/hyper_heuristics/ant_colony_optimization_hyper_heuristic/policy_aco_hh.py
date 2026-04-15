"""
Hyper-ACO Policy Adapter.

Adapts the Hyper-Heuristic ACO solver to the common policy interface.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HyperHeuristicACOConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import HyperHeuristicACO
from logic.src.policies.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators import OPERATOR_NAMES
from logic.src.policies.hyper_heuristics.ant_colony_optimization_hyper_heuristic.params import HyperACOParams
from logic.src.policies.other.operators.heuristics.greedy_initialization import build_greedy_routes


@PolicyRegistry.register("aco_hh")
class HyperACOPolicy(BaseRoutingPolicy):
    """
    Hyper-Heuristic ACO policy class.

    Uses ACO to construct sequences of local search operators.
    """

    def __init__(self, config: Optional[Union[HyperHeuristicACOConfig, Dict[str, Any]]] = None):
        """Initialize Hyper-ACO policy with optional config.

        Args:
            config: HyperHeuristicACOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HyperHeuristicACOConfig

    def _get_config_key(self) -> str:
        """Return config key for Hyper-ACO."""
        return "hyper_aco"

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
        """
        Run Hyper-ACO solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        # Parse parameters
        params = HyperACOParams(
            n_ants=values.get("n_ants", 10),
            alpha=values.get("alpha", 1.0),
            beta=values.get("beta", 2.0),
            rho=values.get("rho", 0.1),
            tau_0=values.get("tau_0", 1.0),
            tau_min=values.get("tau_min", 0.01),
            tau_max=values.get("tau_max", 10.0),
            max_iterations=values.get("max_iterations", 50),
            time_limit=values.get("time_limit", 30.0),
            q0=values.get("q0", 0.9),
            lambda_factor=values.get("lambda_factor", 1.0001),
            operators=values.get("operators", OPERATOR_NAMES.copy()),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        # Determine initial routes
        initial_routes = build_greedy_routes(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            mandatory_nodes=mandatory_nodes,
            rng=random.Random(params.seed) if params.seed is not None else random.Random(),
        )

        solver = HyperHeuristicACO(
            sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, params, initial_routes, mandatory_nodes
        )
        return solver.solve()
