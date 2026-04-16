r"""
Multi-Period Integer L-Shaped (MP-ILS-BD) Policy Adapter.

This module provides the ``MPIntegerLShapedPolicy``, which adapts the
Two-Stage Stochastic MPVRP Benders engine to the ``BaseMultiPeriodRoutingPolicy``
interface. It enables the simulator to execute long-term inventory-routing
optimization over a multi-day rolling horizon.

Algorithm
---------
The policy decomposes the Inventory Routing Problem into:
1.  **Master Problem**: Decides the binary visit variables $y_{it}$ for each
    node $i$ and day $t$ in the horizon $T$, subject to inventory balance
    and capacity constraints.
2.  **Recourse Sub-problems**: Evaluates the expected cost of routing and
    bin overflows for the selected visit pattern across multiple demand
    scenarios.
3.  **Optimality Cuts**: Generates L-shaped optimality cuts (Laporte & Louveaux, 1993)
    to refine the Master Problem's estimate of the expected recourse cost.

Registry key: ``"mp_ils_bd"``

References
----------
Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
stochastic integer programs with complete recourse."
Operations Research Letters, 13(3), 133-142.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MPILSBDConfig
from logic.src.policies.helpers.branching_solvers.vrpp_model import VRPPModel
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .mp_ils_bd_engine import MPIntegerLShapedEngine
from .params import ILSBDParams


@RouteConstructorRegistry.register("mp_ils_bd")
class MPIntegerLShapedPolicy(BaseMultiPeriodRoutingPolicy):
    r"""Two-Stage Stochastic MPVRP via the Integer L-Shaped Method.

    This policy formulates the Stochastic Inventory Routing Problem (SIRP) as a
    two-stage stochastic integer program with explicit inventory balance linking
    constraints.

    Master Problem Formulation:
    .. math::

        \min \quad \sum_{t \in T} \text{RoutingCost}_t(\mathbf{y}) + \theta
        \text{s.t.} \quad I_{it} = I_{i,t-1} + d_{it}(\bar\xi) - q_{it}
        \quad \forall i \in V,\, t \in T

        q_{it} \leq M \cdot y_{it}
        \quad \forall i \in V,\, t \in T

    Where $\theta$ represents the lower bound on the expected recourse cost,
    successively tightened by Integer L-Shaped optimality cuts as the
    search progresses.

    Recourse Evaluation:
    The expected cost of bin stockouts and future routing is evaluated across
    the scenario tree. If the Master Problem solution $(\mathbf{y})$ yields an
    expected cost higher than $\theta$, a new cut is added to the Master.

    Registry key: ``"mp_ils_bd"``

    .. note::
        Requires a Gurobi license (``gurobipy``). Without it, the policy
        will raise an ``ImportError`` during the execution phase.
    """

    def __init__(
        self,
        config: Optional[Union[MPILSBDConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the MP-ILS-BD policy adapter."""
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Type[MPILSBDConfig]:
        return MPILSBDConfig

    def _get_config_key(self) -> str:
        return "mp_ils_bd"

    def _run_multi_period_solver(
        self,
        tree: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        """Execute the Two-Stage Stochastic MPVRP Benders decomposition.

        Args:
            tree: ScenarioTree for recourse evaluation and demand estimation.
            capacity: Vehicle capacity.
            revenue: Revenue per unit waste (R).
            cost_unit: Cost per unit distance (C).
            **kwargs: Additional context keys:
                - distance_matrix: np.ndarray
                - sub_wastes: Dict[int, float]
                - mandatory: List[int]

        Returns:
            Tuple of (full_plan[day][route][node], expected_profit, metadata).
        """
        cfg: MPILSBDConfig = self.config  # type: ignore[assignment]
        sub_dist_matrix: np.ndarray = kwargs["distance_matrix"]
        sub_wastes: Dict[int, float] = kwargs.get("sub_wastes", {})
        mandatory_nodes: List[int] = kwargs.get("mandatory", [])

        n_nodes = len(sub_dist_matrix)

        vrpp_model = VRPPModel(
            n_nodes=n_nodes,
            cost_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            revenue_per_kg=revenue,
            cost_per_km=cost_unit,
            mandatory_nodes=set(mandatory_nodes),
        )

        params = ILSBDParams(
            time_limit=cfg.time_limit,
            master_time_limit=cfg.master_time_limit,
            mip_gap=cfg.mip_gap,
            max_iterations=cfg.max_iterations,
            theta_lower_bound=cfg.theta_lower_bound,
            max_cuts_per_round=cfg.max_cuts_per_round,
            enable_heuristic_rcc_separation=cfg.enable_heuristic_rcc_separation,
            enable_comb_cuts=cfg.enable_comb_cuts,
            verbose=cfg.verbose,
        )

        # Build initial inventory from current fill levels
        customers_list = sorted(vrpp_model.customers)
        N = len(customers_list)
        initial_inventory = np.array([sub_wastes.get(i, cfg.initial_inventory) for i in customers_list], dtype=float)
        bin_capacities = np.full(N, 100.0, dtype=float)

        engine = MPIntegerLShapedEngine(
            model=vrpp_model,
            params=params,
            horizon=cfg.horizon,
            demand_matrix=None,  # derived from tree inside the engine
            bin_capacities=bin_capacities,
            initial_inventory=initial_inventory,
            stockout_penalty=cfg.stockout_penalty,
            big_m=cfg.big_m,
        )

        routes, _y_hat, profit, stats = engine.solve_stochastic(tree=tree)

        # Build full plan snapshot
        full_plan: List[List[List[int]]] = [[] for _ in range(cfg.horizon + 1)]
        full_plan[0] = routes

        return full_plan, profit, stats
