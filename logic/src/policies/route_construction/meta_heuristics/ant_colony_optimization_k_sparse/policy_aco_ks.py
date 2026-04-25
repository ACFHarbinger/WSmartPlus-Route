r"""ACO Policy Adapter.

Adapts the K-Sparse Ant Colony Optimization solver to the common policy interface.

Attributes:
    ACOPolicy: K-Sparse Ant Colony Optimization policy class.

Example:
    >>> from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse import ACOPolicy
    >>> policy = ACOPolicy()
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
    """K-Sparse Ant Colony Optimization policy class.

    Uses ACS with sparse pheromone matrix for efficient VRP solving.

    Attributes:
        config: Configuration for the policy.

    Example:
        >>> policy = ACOPolicy()
    """

    def __init__(self, config: Optional[Union[KSparseACOConfig, Dict[str, Any]]] = None):
        """Initialize ACO policy with optional config.

        Args:
            config: KSparseACOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for ACO.

        Returns:
            The KSparseACOConfig class or None.
        """
        return KSparseACOConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the ACO policy.

        Returns:
            str: The registry key 'aco'.
        """
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
            sub_dist_matrix: Square distance matrix.
            sub_wastes: Node fill levels.
            capacity: Vehicle capacity.
            revenue: Revenue per kg.
            cost_unit: Cost per km.
            values: Merged config dictionary.
            mandatory_nodes: Nodes that must be visited.
            kwargs: Additional context.

        Returns:
            Tuple containing (routes, profit, cost).
        """
        # Use standardized from_config to ensure all fields (including acceptance) are propagated
        params = KSACOParams.from_config(self.config)

        solver = KSparseACOSolver(sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, params, mandatory_nodes)
        return solver.solve()
