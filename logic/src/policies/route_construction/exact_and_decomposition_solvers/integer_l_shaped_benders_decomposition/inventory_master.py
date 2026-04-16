r"""
Inventory-Aware Master Problem for the Two-Stage Stochastic MPVRP.

Extends ``MasterProblem`` with explicit inventory balance linking
constraints across T periods, turning the single-day Benders master into
a full Two-Stage Stochastic Multi-Period VRP formulation.

Mathematical formulation (additional variables and constraints)
--------------------------------------------------------------

New decision variables
~~~~~~~~~~~~~~~~~~~~~~
- :math:`I_{it} \geq 0` — inventory (fill level) of node :math:`i` at the
  end of day :math:`t`.
- :math:`q_{it} \geq 0` — quantity collected from node :math:`i` on day
  :math:`t` (zero if not visited).
- :math:`s_{it} \geq 0` — overflow slack for node :math:`i` on
  day :math:`t` (linearisation of :math:`\max(0, I_{it} - B_i)`).

Inventory balance linking constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. math::

    I_{it} = I_{i,t-1} + d_{it}(\bar\xi) - q_{it}
    \quad \forall i \in V,\, t \in T

Collection coupling (Big-M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. math::

    q_{it} \leq M \cdot y_{it}
    \quad \forall i \in V,\, t \in T

Inventory capacity bound
~~~~~~~~~~~~~~~~~~~~~~~~
.. math::

    I_{it} - s_{it} \leq B_i
    \quad \forall i \in V,\, t \in T

where :math:`s_{it} \geq 0` captures overflow above capacity :math:`B_i`.

Objective extension
~~~~~~~~~~~~~~~~~~~
The overflow penalty :math:`\kappa \sum_t \sum_i s_{it}` is added to the
master objective, where :math:`\kappa` is ``stockout_penalty``.

References
----------
Laporte, G., & Louveaux, F. V. (1993). "The integer L-shaped method for
stochastic integer programs with complete recourse."
Operations Research Letters, 13(3), 133-142.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.branching_solvers.vrpp_model import VRPPModel

from .master_problem import MasterProblem
from .params import ILSBDParams

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp: Any = None  # type: ignore[assignment,no-redef]
    GRB = None  # type: ignore[assignment,misc,no-redef]


class InventoryMasterProblem(MasterProblem):
    r"""Gurobi master problem with inventory balance linking constraints.

    Extends the single-period ``MasterProblem`` to handle a T-day horizon by
    adding :math:`I_{it}`, :math:`q_{it}`, and overflow slack variables with
    the corresponding inventory balance and coupling constraints.

    The additional variables are indexed ``(node_id, day)``, where
    ``day ∈ {0, …, T-1}``.

    Attributes:
        T: Planning horizon (number of days).
        demand_matrix: Array of shape ``(T, N)`` with expected demand increments
            per day per node (mean scenario).
        bin_capacities: Array of shape ``(N,)`` with capacity (100.0 for %).
        initial_inventory: Array of shape ``(N,)`` with initial fill levels.
        stockout_penalty: Penalty per unit of overflow slack.
        big_m: Big-M constant for collection coupling constraints.
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
        r"""Initialise the inventory-extended master problem.

        Args:
            model: VRPPModel providing graph topology and costs.
            params: ILSBDParams configuration.
            horizon: Planning horizon T.
            demand_matrix: Expected demand increments, shape ``(T, N)``.
                           If None, defaults to zeros (no demand evolution).
            bin_capacities: Bin capacities per node (% units), shape ``(N,)``.
                            If None, defaults to 100.0 for all nodes.
            initial_inventory: Initial fill levels, shape ``(N,)``.
                               If None, defaults to zeros.
            stockout_penalty: Penalty per unit overflow per day.
            big_m: Big-M constant for :math:`q_{it} \leq M \cdot y_{it}`.
        """
        super().__init__(model=model, params=params)

        self.T = horizon
        N = len(model.customers)

        self.demand_matrix: np.ndarray = (
            demand_matrix if demand_matrix is not None else np.zeros((horizon, N), dtype=float)
        )
        self.bin_capacities: np.ndarray = (
            bin_capacities if bin_capacities is not None else np.full(N, 100.0, dtype=float)
        )
        self.initial_inventory: np.ndarray = (
            initial_inventory if initial_inventory is not None else np.zeros(N, dtype=float)
        )
        self.stockout_penalty = stockout_penalty
        self.big_m = big_m

        # Inventory decision variables: keyed (node_id, day)
        self._I_vars: Dict[Tuple[int, int], Any] = {}
        self._q_vars: Dict[Tuple[int, int], Any] = {}
        self._s_vars: Dict[Tuple[int, int], Any] = {}  # overflow slack

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_inventory(self) -> None:
        """Extend the base master model with inventory balance constraints.

        Must be called **after** ``build()`` so that the base variables
        (:math:`x`, :math:`y`, :math:`\\theta`) already exist.

        Raises:
            RuntimeError: If ``build()`` has not been called first.
            ImportError: If gurobipy is unavailable.
        """
        if self._gurobi_model is None:
            raise RuntimeError("InventoryMasterProblem.build() must be called before build_inventory().")

        assert GUROBI_AVAILABLE and gp is not None and GRB is not None

        m = self._gurobi_model
        customers = list(self.model.customers)

        # ---- Inventory decision variables --------------------------------
        for idx, i in enumerate(customers):
            cap_i = float(self.bin_capacities[idx]) if idx < len(self.bin_capacities) else 100.0
            for t in range(self.T):
                # I_{it}: inventory at end of day t
                self._I_vars[(i, t)] = m.addVar(
                    lb=0.0,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"I_{i}_{t}",
                )
                # q_{it}: collected quantity on day t
                self._q_vars[(i, t)] = m.addVar(
                    lb=0.0,
                    ub=cap_i,
                    vtype=GRB.CONTINUOUS,
                    name=f"q_{i}_{t}",
                )
                # s_{it}: overflow slack (>= 0)
                self._s_vars[(i, t)] = m.addVar(
                    lb=0.0,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"s_{i}_{t}",
                )

        m.update()

        # ---- Inventory balance & coupling constraints --------------------
        for idx, i in enumerate(customers):
            cap_i = float(self.bin_capacities[idx]) if idx < len(self.bin_capacities) else 100.0
            yi_var = self._y_vars.get(i)

            for t in range(self.T):
                demand_it = float(self.demand_matrix[t, idx]) if t < self.demand_matrix.shape[0] else 0.0

                I_prev = self.initial_inventory[idx] if t == 0 else self._I_vars[(i, t - 1)]

                # 1. Inventory balance: I_{it} = I_{i,t-1} + d_{it} - q_{it}
                m.addConstr(
                    self._I_vars[(i, t)] == I_prev + demand_it - self._q_vars[(i, t)],
                    name=f"inv_balance_{i}_{t}",
                )

                # 2. Coupling: q_{it} ≤ M · y_{it}
                # We use the Day-0 visit decisions (y_i) as the Stage-1
                # variable; for future days we add a Big-M coupling that
                # allows collection only when the visit arc is active.
                if yi_var is not None:
                    m.addConstr(
                        self._q_vars[(i, t)] <= self.big_m * yi_var,
                        name=f"coupling_{i}_{t}",
                    )

                # 3. Capacity bound + overflow linearisation:
                #    I_{it} - s_{it} ≤ B_i  →  s_{it} ≥ I_{it} - B_i
                m.addConstr(
                    self._I_vars[(i, t)] - self._s_vars[(i, t)] <= cap_i,
                    name=f"overflow_{i}_{t}",
                )

        # ---- Extend objective with overflow penalty ----------------------
        overflow_cost = gp.quicksum(
            self.stockout_penalty * self._s_vars[(i, t)] for i in customers for t in range(self.T)
        )
        current_obj = m.getObjective()
        m.setObjective(current_obj + overflow_cost, GRB.MINIMIZE)
        m.update()

    def get_inventory_plan(self) -> Dict[Tuple[int, int], float]:
        """Extract the optimal inventory plan after solve().

        Returns:
            Dict mapping ``(node_id, day)`` to inventory level :math:`I_{it}`.
        """
        if self._gurobi_model is None or self._gurobi_model.SolCount == 0:
            return {}
        return {(i, t): float(var.X) for (i, t), var in self._I_vars.items()}

    def get_collection_plan(self) -> Dict[Tuple[int, int], float]:
        """Extract the optimal collection plan after solve().

        Returns:
            Dict mapping ``(node_id, day)`` to collected quantity :math:`q_{it}`.
        """
        if self._gurobi_model is None or self._gurobi_model.SolCount == 0:
            return {}
        return {(i, t): float(var.X) for (i, t), var in self._q_vars.items()}
