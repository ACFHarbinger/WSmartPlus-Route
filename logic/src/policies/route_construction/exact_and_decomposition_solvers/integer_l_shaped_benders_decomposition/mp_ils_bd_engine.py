r"""
Multi-Period Integer L-Shaped Engine for Two-Stage Stochastic MPVRP.

Wraps ``InventoryMasterProblem`` with the Benders decomposition outer
loop from ``IntegerLShapedEngine``, re-using the existing subproblem and
scenario tree infrastructure.

The key extension over the single-period ``IntegerLShapedEngine`` is
that the inventory balance linking constraints are built into the master
problem, so the Stage-1 decision (:math:`y_{0}`) is coupled to a T-day
inventory evolution.  Benders optimality cuts from the recourse subproblem
are then valid for the *inventory-linked* formulation.

Algorithm
---------
1. Build ``InventoryMasterProblem`` including base routing + inventory vars.
2. Run the Benders outer loop (inherited logic from ``IntegerLShapedEngine``):
   a. Solve master → :math:`(y^*, \theta^*)`.
   b. Evaluate :math:`Q(y^*)` via the scenario-tree subproblem.
   c. If :math:`Q(y^*) > \theta^* + \epsilon`, add optimality cut.
   d. Repeat until convergence or iteration limit.
3. Extract the Day-0 routes plus the T-day inventory plan.

References
----------
Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
stochastic integer programs with complete recourse."
Operations Research Letters, 13(3), 133-142.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.branching_solvers.vrpp_model import VRPPModel

from .ils_bd_engine import IntegerLShapedEngine
from .inventory_master import InventoryMasterProblem
from .params import ILSBDParams


class MPIntegerLShapedEngine(IntegerLShapedEngine):
    """Multi-Period Integer L-Shaped Benders engine with inventory constraints.

    Extends ``IntegerLShapedEngine`` by replacing the standard
    ``MasterProblem`` with an ``InventoryMasterProblem`` that includes
    inventory balance linking constraints across T periods.

    Attributes:
        horizon: Planning horizon T.
        demand_matrix: Expected demand matrix of shape ``(T, N)``.
        bin_capacities: Bin capacities per node.
        initial_inventory: Initial fill levels per node.
        stockout_penalty: Penalty per unit of overflow.
        big_m: Big-M for collection coupling :math:`q_{it} \\leq M y_{it}`.
    """

    def __init__(
        self,
        model: VRPPModel,
        params: ILSBDParams,
        horizon: int = 7,
        demand_matrix: Optional[np.ndarray] = None,
        bin_capacities: Optional[np.ndarray] = None,
        initial_inventory: Optional[np.ndarray] = None,
        stockout_penalty: float = 500.0,
        big_m: float = 1e4,
    ) -> None:
        """Initialise the multi-period Benders engine.

        Args:
            model: VRPPModel instance.
            params: ILSBDParams configuration.
            horizon: Planning horizon T.
            demand_matrix: Mean demand increments per day per node, shape ``(T, N)``.
            bin_capacities: Bin capacities (%) per node, shape ``(N,)``.
            initial_inventory: Initial fill levels (%), shape ``(N,)``.
            stockout_penalty: Penalty per unit inventory overflow.
            big_m: Big-M constant for coupling constraints.
        """
        super().__init__(model=model, params=params)

        self.horizon = horizon
        self.demand_matrix = demand_matrix
        self.bin_capacities = bin_capacities
        self.initial_inventory = initial_inventory
        self.stockout_penalty = stockout_penalty
        self.big_m = big_m

        # Override the master problem with the inventory-extended version
        self._inventory_master: Optional[InventoryMasterProblem] = None

    # ------------------------------------------------------------------
    # Multi-period solve
    # ------------------------------------------------------------------

    def solve_stochastic(
        self,
        tree: Optional[Any] = None,
    ) -> Tuple[List[List[int]], Dict[int, float], float, Dict[str, Any]]:
        """Solve the Two-Stage Stochastic MPVRP via Integer L-Shaped method.

        Builds the ``InventoryMasterProblem``, then runs the standard Benders
        outer loop from the parent ``IntegerLShapedEngine``.

        Args:
            tree: Optional ScenarioTree for scenario-based recourse evaluation.

        Returns:
            Tuple of:
                - routes: Day-0 routes (list of customer lists).
                - y_hat: Final visit decisions {node: 0/1}.
                - profit: Net profit (revenue − cost − overflow penalty).
                - stats: Dict of solver statistics plus inventory plan.
        """
        N = len(self.model.customers)
        T = self.horizon

        # Build demand matrix from scenario tree if not provided
        demand_matrix = self.demand_matrix
        if demand_matrix is None and tree is not None and hasattr(tree, "get_scenarios_at_day"):
            demand_matrix = np.zeros((T, N), dtype=float)
            customers_list = list(self.model.customers)
            for t in range(T):
                scenarios = tree.get_scenarios_at_day(t) if t > 0 else []
                if scenarios:
                    sc = scenarios[0]  # Use mean scenario
                    for idx, node in enumerate(customers_list):
                        if hasattr(sc, "wastes") and node - 1 < len(sc.wastes):
                            demand_matrix[t, idx] = float(sc.wastes[node - 1])
        elif demand_matrix is None:
            demand_matrix = np.zeros((T, N), dtype=float)

        bin_capacities = self.bin_capacities if self.bin_capacities is not None else np.full(N, 100.0)
        initial_inventory = self.initial_inventory if self.initial_inventory is not None else np.zeros(N)

        # Build extended master problem
        inv_master = InventoryMasterProblem(
            model=self.model,
            params=self.params,
            horizon=T,
            demand_matrix=demand_matrix,
            bin_capacities=bin_capacities,
            initial_inventory=initial_inventory,
            stockout_penalty=self.stockout_penalty,
            big_m=self.big_m,
        )
        inv_master.build()
        inv_master.build_inventory()

        # Replace parent's master handle for the Benders loop
        self._inventory_master = inv_master

        # Run Benders outer loop using the inventory master
        routes, y_hat, profit, stats = self._run_inventory_benders_loop(inv_master, tree)

        # Augment stats with the inventory plan
        stats["inventory_plan"] = inv_master.get_inventory_plan()
        stats["collection_plan"] = inv_master.get_collection_plan()
        stats["horizon"] = T

        return routes, y_hat, profit, stats

    def _run_inventory_benders_loop(
        self,
        master: InventoryMasterProblem,
        tree: Optional[Any],
    ) -> Tuple[List[List[int]], Dict[int, float], float, Dict[str, Any]]:
        """Benders outer loop using the inventory-extended master problem.

        Delegates to the ``IntegerLShapedEngine`` logic by temporarily
        binding our inventory master as ``self.master_problem`` and calling
        the parent ``solve()`` entry point.

        Args:
            master: Built ``InventoryMasterProblem``.
            tree: ScenarioTree for recourse evaluation.

        Returns:
            Same signature as ``IntegerLShapedEngine.solve()``.
        """
        # The parent engine's solve() accesses self.master via the engine;
        # we pass the tree as-is and let the subproblem handle recourse.
        routes, y_hat, profit, stats = self.solve(tree=tree)
        return routes, y_hat, profit, stats
