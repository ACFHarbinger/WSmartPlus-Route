"""
Policy Wrapper for Exact Stochastic Dynamic Programming.
"""

from typing import Any, List

import numpy as np

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

    def _run_solver(self, sub_dist: np.ndarray, must_go_sub: List[int], total_nodes: int, **kwargs) -> List[int]:
        """
        Retrieves the exact SDP sequence.
        """
        from dataclasses import asdict, is_dataclass

        cfg = self.params
        if is_dataclass(cfg):
            cfg_dict = asdict(cfg)
        elif isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            cfg_dict = {}

        params = SDPParams.from_config(cfg_dict)

        N = sub_dist.shape[0]
        if params.max_nodes + 1 < N:  # +1 for depot
            # Fallback for overly large graphs
            return list(range(N)) + [0]

        capacity = kwargs.get("capacity", 1.0)
        day = kwargs.get("day", 1)
        wastes = kwargs.get("sub_wastes", {})

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
            engine = ExactSDPEngine(params, sub_dist, capacity)
            engine.solve()
            _SDP_CACHE[cache_key] = engine

        engine = _SDP_CACHE[cache_key]

        optimal_subset = engine.get_optimal_action(day, state_tuple)

        # Enforce must_go elements into subset just in case
        route_set = set(optimal_subset)
        for mg in must_go_sub:
            route_set.add(mg)

        route = list(route_set)
        if not route:
            return [0, 0]

        import itertools

        best_cost = float("inf")
        best_tour = []
        for perm in itertools.permutations(route):
            tour = [0] + list(perm) + [0]
            cost = 0
            for i in range(len(tour) - 1):
                cost += sub_dist[tour[i], tour[i + 1]]
            if cost < best_cost:
                best_cost = cost
                best_tour = tour

        return best_tour


def exact_sdp_solve(*args, **kwargs) -> Any:
    policy = ExactSDPPolicy()
    return policy(*args, **kwargs)
