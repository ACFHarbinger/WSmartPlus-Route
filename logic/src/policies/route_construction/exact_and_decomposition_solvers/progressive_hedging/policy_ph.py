r"""
Progressive Hedging (PH) Policy Adapter for Stochastic VRPP.

This policy adapter bridges the Multi-Period Stochastic framework with the
Progressive Hedging iterative decomposition algorithm. PH is a horizontal
decomposition method that solves the Stochastic Integer Routing Problem (SIRP)
by decomposing the scenario tree by scenario.

Mathematical Formulation — Augmented Lagrangian (Rockafellar & Wets, 1991):
    The algorithm seeks to solve:
        min  Σ_{s \in S} p_s f_s(x_s)
        s.t. x_s \in X_s,  s \in S
             x_s - z = 0,  s \in S (Non-anticipativity / Consensus)

    PH relaxes the consensus constraints and solves subproblems of the form:
        min  f_s(x_s) + w_s^T(x_s - z) + (ρ/2) ||x_s - z||^2

    where:
        - f_s(x_s) is the deterministic VRPP objective for scenario s over horizon T.
        - w_s are the dual multipliers (Walrasian prices) penalizing disagreement.
        - ρ is the penalty parameter for the proximal term.
        - z is the scenario-averaged consensus solution.

Implementation Details:
    1.  **Lookahead ($T$)**: Unlike myopic solvers, PH optimizes over the full
        $T$-day horizon provided in the `multi_day_context`.
    2.  **Scenario Decomposition**: Scenarios are extracted from the `ScenarioTree`.
    3.  **Iteration**: The `ProgressiveHedgingEngine` performs the outer loop of
        sub-gradient updates to weights $w_s$ and proximal parameter ρ.
    4.  **Wait-and-See (W&S) Seeding**: Initial subproblems are solved without
        penalties to provide a high-quality warm start for the consensus.

Registry key: ``"ph"``

References:
    Rockafellar, R. T., & Wets, R. J. B. (1991). "Scenarios and policy aggregation
    in optimization under uncertainty". Mathematics of Operations Research, 16(1), 119-147.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from logic.src.configs.policies import PHConfig
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .ph_engine import ProgressiveHedgingEngine


@RouteConstructorRegistry.register("ph")
class ProgressiveHedgingPolicy(BaseMultiPeriodRoutingPolicy):
    """Policy adapter for Progressive Hedging decomposition.

    This class handles the boilerplate of scenario generation and invokes the
    ProgressiveHedgingEngine to solve the stochastic VRPP iteratively.
    """

    def __init__(self, config: PHConfig) -> None:
        """Initialize the PH policy.

        Args:
            config: PH configuration dataclass.
        """
        super().__init__(config)
        self.engine = ProgressiveHedgingEngine(self.config)

    @classmethod
    def _config_class(cls) -> Type[PHConfig]:
        """Return the PH configuration dataclass."""
        return PHConfig

    def _get_config_key(self) -> str:
        """Return the configuration key."""
        return "ph"

    def _run_multi_period_solver(
        self,
        tree: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """
        Execute the Progressive Hedging (PH) decomposition solver logic.

        This method optimizes decisions across the scenario tree by decomposing
        it into individual scenarios, which are solved independently in each
        iteration. A consensus mechanism (based on scenario averaging and
        Augmented Lagrangian penalties) is used to force scenario-specific
        decisions toward a single implementable policy for Day 0.

        Args:
            tree (ScenarioTree): Tree of future fill rate realizations.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            **kwargs: Additional context, including:
                - distance_matrix (np.ndarray): Symmetric distance matrix.
                - mandatory (List[int]): Bins that MUST be collected today.

        Returns:
            Tuple[List[List[List[int]]], float, Dict[str, Any]]:
                A 3-tuple containing:
                - full_plan: Collection plan for all days and scenarios (aggregated).
                - expected_profit: Expected objective value under the consensus policy.
                - stats: Execution statistics, including convergence metrics (dual residuals).
        """
        return self.engine.solve(
            sub_dist_matrix=kwargs["distance_matrix"],
            scenario_tree=tree,
            capacity=capacity,
            revenue=revenue,
            cost_unit=cost_unit,
            mandatory_nodes=kwargs.get("mandatory", []),
            **kwargs,
        )

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Legacy single-day solver (optional fallback)."""
        # For simplicity in this SIRP refactor, we redirect to multi-period with a 1-day tree if needed
        # but here we just return empty to discourage myopic usage of PH.
        return [], 0.0, 0.0
