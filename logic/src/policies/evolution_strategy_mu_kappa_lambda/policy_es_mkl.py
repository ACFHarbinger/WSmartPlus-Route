"""
(μ,κ,λ) Evolution Strategy Policy Adapter with Age-Based Selection.

Adapts the (μ,κ,λ)-ES with age control for vehicle routing problems.
Individuals exceeding age κ are excluded from selection.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MuKappaLambdaESConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.evolution_strategy_mu_kappa_lambda import MuKappaLambdaESParams


@PolicyRegistry.register("mu_kappa_lambda_es")
class MuKappaLambdaESPolicy(BaseRoutingPolicy):
    """
    (μ,κ,λ) Evolution Strategy policy with age-based selection.

    Age-limited parent survival prevents stagnation in long runs.
    """

    def __init__(self, config: Optional[Union[MuKappaLambdaESConfig, Dict[str, Any]]] = None):
        """Initialize (μ,κ,λ)-ES policy with optional config.

        Args:
            config: MuKappaLambdaESConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MuKappaLambdaESConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "mu_kappa_lambda_es"

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
        Run (μ,κ,λ)-ES solver for routing problems.

        Uses a routing-adapted version of the classical ES with age control.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        # Import the routing solver
        from logic.src.policies.evolution_strategy_mu_kappa_lambda.routing_solver import (
            MuKappaLambdaESRoutingSolver,
        )

        params = MuKappaLambdaESParams(
            mu=values.get("mu", 15),
            kappa=values.get("kappa", 7),
            lambda_=values.get("lambda_", 100),
            rho=values.get("rho", 2),
            recombination_type=values.get("recombination_type", "intermediate"),
            n_removal=values.get("n_removal", 3),
            stagnation_limit=values.get("stagnation_limit", 10),
            max_iterations=values.get("max_iterations", 500),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
        )

        solver = MuKappaLambdaESRoutingSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
            seed=values.get("seed"),
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
