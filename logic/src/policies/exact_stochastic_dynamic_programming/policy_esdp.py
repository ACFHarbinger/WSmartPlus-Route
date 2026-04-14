"""
Policy Wrapper for Exact Stochastic Dynamic Programming.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from logic.src.configs.policies.esdp import ExactSDPConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.exact_stochastic_dynamic_programming.esdp_engine import ExactSDPEngine
from logic.src.policies.exact_stochastic_dynamic_programming.params import SDPParams

# Global cache for the DP table to avoid recomputations across simulation days.
_SDP_CACHE = {}


@PolicyRegistry.register("esdp")
class ExactSDPPolicy(BaseRoutingPolicy):
    """
    Adapter bridging WSmart-Route simulation loop to the Exact SDP Engine.
    """

    def _get_config_class(self):
        return ExactSDPConfig

    def _get_config_key(self) -> str:
        return "exact_sdp"

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Retrieves the exact SDP sequence.
        """
        sub_dist_matrix = kwargs["sub_dist_matrix"]
        sub_wastes = kwargs["sub_wastes"]
        capacity = kwargs["capacity"]
        mandatory_nodes = kwargs["mandatory_nodes"]

        cfg_dict = asdict(self._config or kwargs.get("config", {}).get("esdp", {}))
        params = SDPParams.from_config(cfg_dict)

        N = sub_dist_matrix.shape[0]
        if params.max_nodes + 1 < N:  # +1 for depot
            # Fallback for overly large graphs
            return ([0] + list(range(1, N)) + [0], 0.0, {"profit": 0.0})

        day = kwargs.get("day", 1)
        wastes = sub_wastes

        state_lst = []
        for i in range(1, N):
            w = wastes.get(i, kwargs.get("wastes", {}).get(i, 0.0))
            lvl = int(round((w / max(1.0, capacity)) * (params.discrete_levels - 1)))
            lvl = max(0, min(params.discrete_levels - 1, lvl))
            state_lst.append(lvl)

        state_tuple = tuple(state_lst)

        cache_key = (N, params.num_days, params.discrete_levels, params.max_fill_rate)
        if cache_key not in _SDP_CACHE:
            print(f"INITIALIZING EXACT SDP ENGINE for N={N} (Offline Precomputation)...")
            engine = ExactSDPEngine(params, sub_dist_matrix, capacity)
            engine.solve()
            _SDP_CACHE[cache_key] = engine

        engine = _SDP_CACHE[cache_key]

        optimal_subset = engine.get_optimal_action(day, state_tuple)

        # Enforce must_go elements into subset just in case
        route_set = set(optimal_subset)
        for mg in mandatory_nodes:
            route_set.add(mg)

        route = list(route_set)
        if not route:
            return ([0, 0], 0.0, {"profit": 0.0})

        import itertools

        best_cost = float("inf")
        best_tour = []
        for perm in itertools.permutations(route):
            tour = [0] + list(perm) + [0]
            cost = 0
            for i in range(len(tour) - 1):
                cost += sub_dist_matrix[tour[i], tour[i + 1]]
            if cost < best_cost:
                best_cost = cost
                best_tour = tour

        return best_tour, float(best_cost), {"profit": 0.0}
