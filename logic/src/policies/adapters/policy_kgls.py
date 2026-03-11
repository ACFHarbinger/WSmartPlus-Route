"""
KGLS Policy Adapter.

Adapts the Knowledge-Guided Local Search (KGLS) to the global routing interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import KGLSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.knowledge_guided_local_search.kgls import KGLSSolver
from logic.src.policies.knowledge_guided_local_search.params import KGLSParams

from .factory import PolicyRegistry


@PolicyRegistry.register("kgls")
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
        kgls_config = KGLSConfig(**values)
        params = KGLSParams.from_config(kgls_config)

        # We need the locations for the Cost Evaluator to compute edge widths
        # If the environment passes 'data_nodes' directly through the pipeline, use it.
        # Otherwise default to zero'd geometry effectively killing width but preserving length evaluations
        loc_data = kwargs.get("data_nodes")
        n_nodes = sub_dist_matrix.shape[0]
        locations = np.zeros((n_nodes, 2))

        if loc_data is not None:
            # Re-map 1...N onto 1...len(sub_dist_matrix) maintaining the continuous graph
            # This handles both native and sub-graph executions cleanly
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
