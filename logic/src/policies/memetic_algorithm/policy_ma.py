"""
MA (Memetic Algorithm) Policy Adapter.

This module provides the integration layer between the MASolver and the
broader simulator infrastructure. It adapts the evolutionary engine to the
agnostic BaseRoutingPolicy interface.

Reference:
    Moscato, P., Cotta, C., & Mendes, A. (2004). "Memetic Algorithms".
    Reference: bibliography/Memetic_Algorithms.pdf
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ma import MAConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import MAParams
from .solver import MASolver


@PolicyRegistry.register("ma")
class MAPolicy(BaseRoutingPolicy):
    """
    Adapter for the Memetic Algorithm (MA) solver.

    This policy handles the conversion from abstract problem variables
    (distance matrices, waste levels) into the specific parameters and state
    required by the MASolver's evolutionary generational step.

    Design Pattern: Adapter
    ----------------------
    MAPolicy acts as an adapter, allowing the MASolver implementation (which
    focused on numerical optimization) to be used within the simulator's
    BaseRoutingPolicy framework.
    """

    def __init__(self, config: Optional[Union[MAConfig, Dict[str, Any]]] = None):
        """
        Initialize the MAPolicy adapter.

        Args:
            config: An optional configuration object or dictionary containing
                    the engine hyper-parameters.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Defines the Hydra-compliant configuration schema for this policy.

        Returns:
            MAConfig class.
        """
        return MAConfig

    def _get_config_key(self) -> str:
        """
        Returns the unique identifier for this policy in the registry.

        Returns:
            "ma"
        """
        return "ma"

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
        Internal execution bridge to the MASolver.

        Prepares high-level simulator data into the mathematical representation
        expected by the generic Memetic Algorithm engine.

        Algorithm Process:
        1. Extract and cast hyper-parameters from the combined 'values' dictionary.
        2. Instantiate the MASolver with problem-specific data (Distance, Waste, Capacity).
        3. Invoke the solve() method to run the generational evolutionary processes.
        4. Post-process results into the standard (Routes, Profit, Cost) format.

        Returns:
            Tuple[List[List[int]], float, float]:
                - best_routes: Optimal collection sequences found.
                - best_reward: Net profit (Revenue - Distance Cost).
                - total_cost: Distance component of the solution.
        """
        # 1. Parameter Extraction & Mapping (See params.py for conceptual mapping)
        params = MAParams(
            pop_size=int(values.get("pop_size", 30)),
            max_generations=int(values.get("max_generations", 100)),
            crossover_rate=float(values.get("crossover_rate", 0.8)),
            mutation_rate=float(values.get("mutation_rate", 0.1)),
            local_search_rate=float(values.get("local_search_rate", 1.0)),
            tournament_size=int(values.get("tournament_size", 3)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
        )

        # 2. Solver Initialization
        solver = MASolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
            seed=values.get("seed"),
        )

        # 3. Evolutionary Optimization (Moscato FIG 3.1)
        # Returns the best individual and its fitness (profit).
        best_routes, best_reward = solver.solve()

        # 4. Result Formatting & Supplemental Metrics
        # The simulator Expects (Routes, NetProfit, AbsoluteDistanceCost).
        total_cost = solver._cost(best_routes) * cost_unit

        return best_routes, best_reward, total_cost
