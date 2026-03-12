"""
Stochastic Tournament GA Policy Adapter.

Adapts the rigorous Stochastic Tournament GA (replaces LCA).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import StochasticTournamentGAConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import StochasticTournamentGAParams
from .solver import StochasticTournamentGASolver


@PolicyRegistry.register("stochastic_tournament_ga")
class StochasticTournamentGAPolicy(BaseRoutingPolicy):
    """
    Stochastic Tournament Genetic Algorithm policy class.

    GA with sigmoid-based pairwise tournament selection. Replaces LCA.
    """

    def __init__(self, config: Optional[Union[StochasticTournamentGAConfig, Dict[str, Any]]] = None):
        """Initialize Stochastic Tournament GA policy with optional config.

        Args:
            config: StochasticTournamentGAConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return StochasticTournamentGAConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "stochastic_tournament_ga"

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
        Run Stochastic Tournament GA solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = StochasticTournamentGAParams(
            population_size=values.get("population_size", 50),
            tournament_competitors=values.get("tournament_competitors", 5),
            selection_pressure=values.get("selection_pressure", 0.1),
            crossover_rate=values.get("crossover_rate", 0.8),
            mutation_rate=values.get("mutation_rate", 0.2),
            elitism_count=values.get("elitism_count", 2),
            max_generations=values.get("max_generations", 100),
            time_limit=values.get("time_limit", 60.0),
        )

        solver = StochasticTournamentGASolver(
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
