"""
KGLS Policy Adapter.

Adapts the Knowledge-Guided Local Search (KGLS) to the global routing interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import KGLSConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .kgls import KGLSSolver
from .params import KGLSParams


@RouteConstructorRegistry.register("kgls")
class KGLSPolicy(BaseRoutingPolicy):
    """
    KGLS policy class.

    Explores CVRP solutions by penalizing structurally sub-optimal connections.
    """

    def __init__(self, config: Optional[Union[KGLSConfig, Dict[str, Any]]] = None):
        """Initialize KGLS policy with optional config.

        Args:
            config: KGLSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return KGLSConfig

    def _get_config_key(self) -> str:
        """Return config key for KGLS."""
        return "kgls"

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
        Run KGLS solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = KGLSParams(
            time_limit=float(values.get("time_limit", 60.0)),
            num_perturbations=int(values.get("num_perturbations", 3)),
            neighborhood_size=int(values.get("neighborhood_size", 20)),
            moves=values.get("moves", ["relocate", "swap", "two_opt", "cross_exchange"]),
            penalization_cycle=values.get("penalization_cycle", ["width", "length", "width_length"]),
            local_search_iterations=int(values.get("local_search_iterations", 100)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        # We need the locations for the Cost Evaluator to compute edge widths
        loc_data = kwargs.get("data_nodes")
        n_nodes = sub_dist_matrix.shape[0]
        locations = np.zeros((n_nodes, 2))
        if loc_data is not None:
            # Re-map 1...N onto 1...len(sub_dist_matrix) maintaining the continuous graph
            locs = loc_data.get("locs", [])
            for i, mapped_n in enumerate(mandatory_nodes + list(set(sub_wastes.keys()) - set(mandatory_nodes))):
                if mapped_n < len(locs):
                    # + 1 offset because 0 is depot
                    locations[i + 1] = locs[mapped_n]
            # Map depot
            locations[0] = loc_data.get("depot", [0.0, 0.0])

        solver = KGLSSolver(
            dist_matrix=sub_dist_matrix,
            locations=locations,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, best_profit, best_cost = solver.solve()

        return routes, best_profit, best_cost
