"""
ADP Rollout Policy Adapter.

Adapts the ``ADPRolloutEngine`` to the ``BaseMultiPeriodRoutingPolicy``
interface, enabling simulation-level evaluation via ``test_sim`` with the
registry key ``"adp_rollout"``.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ADPRolloutConfig
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .adp_engine import ADPRolloutEngine
from .params import ADPRolloutParams


@RouteConstructorRegistry.register("adp_rollout")
class ADPRolloutPolicy(BaseMultiPeriodRoutingPolicy):
    r"""Approximate Dynamic Programming (ADP) Rollout policy for stochastic IRP.

    Implements Powell's ADP framework with a configureable candidate-set
    strategy and truncated greedy rollout for estimating :math:`V(S_{t+1})`.

    Three candidate strategies are available (set via ``candidate_strategy``):

    * ``"threshold"`` — include all nodes above ``fill_threshold``.
    * ``"top_k"``     — include the top-K nodes by fill level.
    * ``"beam"``      — explore up to ``max_candidate_sets`` random subsets.

    Registry key: ``"adp_rollout"``

    References:
        Powell, W. B. (2011). *Approximate Dynamic Programming:
        Solving the Curses of Dimensionality.* Wiley-Interscience.
    """

    def __init__(
        self,
        config: Optional[Union[ADPRolloutConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the ADP Rollout policy adapter."""
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Type[ADPRolloutConfig]:
        return ADPRolloutConfig

    def _get_config_key(self) -> str:
        return "adp_rollout"

    def _run_multi_period_solver(
        self,
        tree: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """Execute the ADP rollout over the T-day horizon.

        Args:
            tree: ScenarioTree for demand simulation.
            capacity: Vehicle capacity.
            revenue: Revenue per unit waste (R).
            cost_unit: Cost per unit distance (C).
            **kwargs: Additional context (distance_matrix, sub_wastes, mandatory, etc.).

        Returns:
            Tuple of (full_plan[day][route][node], total_expected_profit, metadata).
        """
        cfg: ADPRolloutConfig = self.config  # type: ignore[assignment]
        sub_dist_matrix: np.ndarray = kwargs["distance_matrix"]
        sub_wastes: Dict[int, float] = kwargs.get("sub_wastes", {})
        mandatory_nodes: List[int] = kwargs.get("mandatory", [])

        params = ADPRolloutParams.from_config(asdict(cfg))

        engine = ADPRolloutEngine(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        full_plan, total_profit, metadata = engine.solve(
            scenario_tree=tree,
            horizon=cfg.horizon,
        )

        return full_plan, total_profit, metadata
