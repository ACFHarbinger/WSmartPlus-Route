"""
Progressive Hedging (PH) policy adapter.

Adapts the Progressive Hedging iterative algorithm to the BaseRoutingPolicy
interface, allowing it to be used within the standard simulation and evaluation
pipelines.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from logic.src.configs.policies import PHConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.registry import PolicyRegistry
from logic.src.policies.integer_l_shaped_benders_decomposition.scenario import ScenarioGenerator

from .ph_engine import ProgressiveHedgingEngine


@PolicyRegistry.register("progressive_hedging")
class ProgressiveHedgingPolicy(BaseRoutingPolicy):
    """Policy adapter for Progressive Hedging decomposition.

    This class handles the boilerplate of scenario generation and invokes the
    ProgressiveHedgingEngine to solve the stochastic VRPP iteratively.
    """

    def __init__(self, config: PHConfig) -> None:
        """Initialize the PH policy.

        Args:
            config: PH configuration dataclass.
        """
        super().__init__(config)
        self.engine = ProgressiveHedgingEngine(self.config)
        self.scenario_generator = ScenarioGenerator()

    @classmethod
    def _config_class(cls) -> Type[PHConfig]:
        """Return the PH configuration dataclass."""
        return PHConfig

    def _get_config_key(self) -> str:
        """Return the configuration key."""
        return "progressive_hedging"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Execute the Progressive Hedging solver.

        Args:
            sub_dist_matrix: Localized distance matrix.
            sub_wastes: Observed node wastes/fill levels.
            capacity: Vehicle capacity.
            revenue: Revenue per unit collected.
            cost_unit: Travel cost per distance unit.
            values: Additional parameters from the configuration.
            mandatory_nodes: Nodes that must be visited.
            **kwargs: Additional parameters including 'scenarios' override.

        Returns:
            Tuple of (routes, expected_profit, total_distance).
        """
        # 1. Prepare Scenarios (SAA)
        scenarios = kwargs.get("scenarios")
        if scenarios is None:
            # If no scenarios provided (e.g. from simulation), generate them based on sub_wastes
            n_scenarios = values.get("num_scenarios", self.config.num_scenarios)
            scenarios = self.scenario_generator.generate(
                sub_wastes=sub_wastes, n_scenarios=n_scenarios, seed=self.config.seed
            )

        # 2. Invoke PH Engine
        routes, expected_profit, stats = self.engine.solve(
            sub_dist_matrix=sub_dist_matrix,
            scenario_wastes=scenarios,
            capacity=capacity,
            revenue=revenue,
            cost_unit=cost_unit,
            mandatory_nodes=mandatory_nodes or [],
            **kwargs,
        )

        # 3. Compute final distance based on the chosen routes
        total_distance = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total_distance += sub_dist_matrix[route[i], route[i + 1]]

        return routes, expected_profit, total_distance
