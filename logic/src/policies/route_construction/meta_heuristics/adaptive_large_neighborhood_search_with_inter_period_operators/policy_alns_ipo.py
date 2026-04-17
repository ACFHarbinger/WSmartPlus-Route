"""
ALNS with Inter-Period Operators (ALNS-IPO) Policy Adapter.

Adapts the ALNS-IPO solver (``ALNSSolverIPO``) to the ``BaseMultiPeriodRoutingPolicy``
interface, enabling it to be invoked via ``test_sim`` with the key ``"alns_ipo"``.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ALNSIPOConfig
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .alns_ipo import ALNSSolverIPO
from .params import ALNSIPOParams


@RouteConstructorRegistry.register("alns_ipo")
class ALNSInterPeriodOperatorsPolicy(BaseMultiPeriodRoutingPolicy):
    r"""ALNS policy with Inter-Period Operators (IPO).

    Maintains a full T-day horizon chromosome and uses ShiftVisitRemoval,
    PatternRemoval, and ForwardLookingInsertion operators to reshape the
    visit schedule across the entire planning horizon.

    Registry key: ``"alns_ipo"``

    References:
        Coelho, L. C., Cordeau, J.-F., & Laporte, G. (2012).
        "The inventory-routing problem with transshipment."
        Computers & Operations Research, 39(11), 2537–2548.
    """

    def __init__(
        self,
        config: Optional[Union[ALNSIPOConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the ALNS-IPO policy adapter."""
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Type[ALNSIPOConfig]:
        return ALNSIPOConfig

    def _get_config_key(self) -> str:
        return "alns_ipo"

    def _run_multi_period_solver(
        self,
        tree: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """Execute the multi-period ALNS over the T-day horizon.

        Args:
            tree: ScenarioTree for forward-looking inventory evaluation.
            capacity: Vehicle capacity.
            revenue: Revenue per unit waste (R).
            cost_unit: Cost per unit distance (C).
            **kwargs: Additional context (distance_matrix, sub_wastes, mandatory, etc.).

        Returns:
            Tuple of (full_plan[day][route][node], total_expected_profit, metadata).
        """
        cfg: ALNSIPOConfig = self.config  # type: ignore[assignment]
        sub_dist_matrix: np.ndarray = kwargs["distance_matrix"]
        sub_wastes: Dict[int, float] = kwargs.get("sub_wastes", {})
        mandatory_nodes: List[int] = kwargs.get("mandatory", [])

        # Initialize type-safe Params
        params = ALNSIPOParams.from_config(asdict(self.config))

        # Inject Boltzmann acceptance criterion (if not already handled by from_config)
        # Note: base ALNSParams handles this, but we preserve local override if specific method is needed
        if params.acceptance_criterion is None:
            from logic.src.policies.route_construction.acceptance_criteria.base.factory import (
                AcceptanceCriterionFactory,
            )

            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="bmc",
                initial_temp=params.start_temp,
                alpha=params.cooling_rate,
                seed=params.seed,
            )

        solver = ALNSSolverIPO(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        best_horizon, best_profit, best_cost = solver.solve_horizon(scenario_tree=tree)

        metadata: Dict[str, Any] = {
            "horizon_cost": best_cost,
            "horizon_profit": best_profit,
            "n_days": cfg.horizon,
        }

        return best_horizon, best_profit, metadata
