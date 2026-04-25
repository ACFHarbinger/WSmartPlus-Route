"""
SANS Policy Adapter (Simulated Annealing Neighborhood Search).

Uses Simulated Annealing for route optimization.
Supports two engines:
  - 'new': Improved SA with initial solution and iterative refinement
  - 'og': Original look-ahead algorithm for collection (LAC)

Attributes:
    SANSPolicy: Policy class for the SANS approach.

Example:
    >>> from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.policy_sans import SANSPolicy
    >>> sans_policy = SANSPolicy()
    >>> sans_policy.execute()
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import SANSConfig
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.dispatcher import (
    execute_new,
    execute_og,
)

from .params import SANSParams


@RouteConstructorRegistry.register("sans")
@RouteConstructorRegistry.register("lac")  # Backward compatibility alias
class SANSPolicy(BaseRoutingPolicy):
    """
    Simulated Annealing Neighborhood Search policy class.

    Uses SA optimization with custom initialization and mandatory enforcement.
    Supports two engines via the 'engine' parameter:
      - 'new': Improved simulated annealing with initial solution computation
      - 'og': Original look-ahead collection (LAC) algorithm

    Attributes:
        None
    """

    def __init__(self, config: Optional[Union[SANSConfig, Dict[str, Any]]] = None):
        """Initialize SANS policy with optional config.

        Args:
            config: SANSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for the SANS policy.

        Returns:
            Optional[Type]: The configuration class for the SANS policy.
        """
        return SANSConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the SANS policy.

        Returns:
            str: The registry key 'sans'.
        """
        return "sans"

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
        """Not used - SANS requires specialized execute().

        Args:
            sub_dist_matrix: Distance matrix for the current route.
            sub_wastes: Dictionary of wastes for the current route.
            capacity: Vehicle capacity.
            revenue: Revenue for the current route.
            cost_unit: Cost per unit of distance.
            values: Dictionary of values for the current route.
            mandatory_nodes: List of mandatory nodes.
            kwargs: Keyword arguments.

        Returns:
            Tuple[List[List[int]], float, float]:
                Tuple of (solution, profit, cost).
        """
        return [[]], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the SANS policy.

        Uses specialized data preparation for simulated annealing.

        Args:
            kwargs: Keyword arguments.

        Returns:
            Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
                Tuple of (solution, profit, cost, search_context, multi_day_context).
        """
        # Determine engine and parameters from typed config, raw config, or kwargs
        params = SANSParams.from_config(self._config or kwargs.get("config", {}).get("sans", {}))

        if params.engine == "og":
            return execute_og(self, params=params, **kwargs)
        else:
            return execute_new(self, params=params, **kwargs)
