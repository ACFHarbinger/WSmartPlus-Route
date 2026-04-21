"""
ACO Policy Adapter.

Adapts the K-Sparse Ant Colony Optimization solver to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import KSparseACOConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import KSACOParams
from .solver import KSparseACOSolver


@RouteConstructorRegistry.register("aco_ks")
class ACOPolicy(BaseRoutingPolicy):
    """
    K-Sparse Ant Colony Optimization policy class.

    Uses ACS with sparse pheromone matrix for efficient VRP solving.
    """

    def __init__(self, config: Optional[Union[KSparseACOConfig, Dict[str, Any]]] = None):
        """Initialize ACO policy with optional config.

        Args:
            config: KSparseACOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return KSparseACOConfig

    def _get_config_key(self) -> str:
        """Return config key for ACO."""
        return "aco"

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
        Execute the K-Sparse Ant Colony Optimization (ACO) solver logic.

        This implementation uses the Ant Colony System (ACS) framework with a
        sparse pheromone matrix (K-Sparse). Ants construct solutions by
        probabilistically choosing next nodes based on pheromone levels and
        greedy desirability (profit-to-distance ratio). Pheromones are
        updated globally by the best-found solution (and optionally elitist
        ants) and locally by every ant during construction to encourage
        exploration.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                ACO parameters (n_ants, alpha, beta, q0, k_sparse).
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        # Use standardized from_config to ensure all fields (including acceptance) are propagated
        params = KSACOParams.from_config(self.config)

        solver = KSparseACOSolver(sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, params, mandatory_nodes)
        return solver.solve()
