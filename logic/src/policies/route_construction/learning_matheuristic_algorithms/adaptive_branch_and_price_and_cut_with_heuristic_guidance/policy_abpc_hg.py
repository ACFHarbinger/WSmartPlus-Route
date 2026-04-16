from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.abpc_hg import ABPCHGConfig
from logic.src.pipeline.simulations.bins.prediction import ScenarioTree
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .scenario_prize_engine import ScenarioPrizeEngine
from .temporal_benders import TemporalBendersCoordinator


@RouteConstructorRegistry.register("abpc_hg")
class ABPCHGPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Adaptive Branch-and-Price-and-Cut with Heuristic Guidance (ABPC-HG).

    This policy wraps the exact BPC components but injects advanced matheuristic
    and scenario-based logic to solve Stochastic Multi-Period VRPs.
    """

    def __init__(self, config: Optional[ABPCHGConfig] = None):
        super().__init__(config)
        self.gamma = getattr(self.config, "gamma", 0.95) if self.config else 0.95

    @classmethod
    def _config_class(cls) -> Type[ABPCHGConfig]:
        return ABPCHGConfig

    def _get_config_key(self) -> str:
        return "abpc_hg"

    def _run_multi_period_solver(
        self,
        tree: ScenarioTree,
        capacity: float,
        revenue: float,
        cost_unit: float,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """
        Execute the ABPC-HG pipeline for the multi-day horizon.
        """
        # Step 1: Initialize the ScenarioTree (already passed in via context)

        # Step 2: Compute scenario-augmented node prizes (managed by the engine)
        prize_engine = ScenarioPrizeEngine(
            scenario_tree=tree,
            gamma=self.gamma,
            tau=capacity,
        )

        # Step 3 & 4: Execute Temporal Benders Master Problem
        coordinator = TemporalBendersCoordinator(
            tree=tree,
            prize_engine=prize_engine,
            capacity=capacity,
            revenue=revenue,
            cost_unit=cost_unit,
        )

        # The coordinator manages the Benders iterations, internally using:
        # - ProgressiveHedgingCGLoop at the root node.
        # - ALNSMultiPeriodPricer for heuristic column generation.
        # - MLBranchingStrategy and ScenarioConsistentBranching.
        # - DiveAndPricePrimalHeuristic for upper bounds.
        # - FixAndOptimizeRefiner for polishing.

        raw_plan, total_expected_profit = coordinator.solve(**kwargs)

        metadata = {
            "policy": "abpc_hg",
            "expected_profit": total_expected_profit,
        }

        return raw_plan, total_expected_profit, metadata
