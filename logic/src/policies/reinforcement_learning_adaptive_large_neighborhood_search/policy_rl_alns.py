from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry


@PolicyRegistry.register("rl_alns")
class RLALNSPolicy(BaseRoutingPolicy):
    """
    Adapter for the RL-ALNS solver.

    Bridges the centralized RLConfig structure with the RLALNSSolver.
    """

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
        Instantiate and run the RLALNSSolver.

        Args:
            sub_dist_matrix: Subset distance matrix.
            sub_wastes: Local wastes dict.
            capacity: Vehicle capacity.
            revenue: Revenue per distance.
            cost_unit: Cost per distance unit.
            values: Merged configuration dictionary.
            mandatory_nodes: Local indices of nodes that MUST be visited.
            **kwargs: Extra arguments.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        from logic.src.configs.policies.other import (
            BanditConfig,
            FeatureExtractorConfig,
            RewardShapingConfig,
            RLConfig,
            TDLearningConfig,
        )

        from ..reinforcement_learning_adaptive_large_neighborhood_search.params import RLALNSParams
        from ..reinforcement_learning_adaptive_large_neighborhood_search.solver import RLALNSSolver

        # Handle nested rl_config if present (from new Hydra structure)
        rl_vals = values.get("rl_config")
        if isinstance(rl_vals, dict):
            # Safe way to convert nested dict to RLConfig
            rl_config = RLConfig(
                agent_type=rl_vals.get("agent_type", "bandit"),
                bandit=BanditConfig(**{k: v for k, v in rl_vals.get("bandit", {}).items() if k != "_target_"}),
                td_learning=TDLearningConfig(
                    **{k: v for k, v in rl_vals.get("td_learning", {}).items() if k != "_target_"}
                ),
                sarsa=TDLearningConfig(**{k: v for k, v in rl_vals.get("sarsa", {}).items() if k != "_target_"})
                if rl_vals.get("sarsa")
                else None,
                reward=RewardShapingConfig(**{k: v for k, v in rl_vals.get("reward", {}).items() if k != "_target_"}),
                features=FeatureExtractorConfig(
                    **{k: v for k, v in rl_vals.get("features", {}).items() if k != "_target_"}
                ),
                params=rl_vals.get("params", {}),
            )
        else:
            # Fallback for flat config structure (legacy)
            rl_config = RLConfig(
                agent_type=values.get("agent_type", "bandit"),
                bandit=BanditConfig(
                    algorithm=values.get("rl_algorithm", "ucb1"),
                    c=values.get("ucb_c", 2.0),
                    seed=kwargs.get("seed"),
                ),
                td_learning=TDLearningConfig(
                    alpha=values.get("alpha", 0.1),
                    gamma=values.get("gamma", 0.95),
                    epsilon=values.get("epsilon", 0.1),
                ),
                reward=RewardShapingConfig(
                    best_reward=values.get("reward_new_global_best", 10.0),
                    local_reward=values.get("reward_improved_current", 5.0),
                    accepted_reward=values.get("reward_accepted_worse", 1.0),
                    rejected_reward=values.get("reward_rejected", -1.0),
                ),
            )

        params = RLALNSParams(
            time_limit=values.get("time_limit", 60.0),
            max_iterations=values.get("max_iterations", 5000),
            start_temp=values.get("start_temp", 100.0),
            cooling_rate=values.get("cooling_rate", 0.995),
            min_removal=values.get("min_removal", 1),
            max_removal_pct=values.get("max_removal_pct", 0.3),
            rl_config=rl_config,
        )

        solver = RLALNSSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
            seed=kwargs.get("seed"),
        )

        return solver.solve()
