from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp: Any = None  # type: ignore[assignment,no-redef]
    GRB: Any = None  # type: ignore[assignment,misc,no-redef]

from logic.src.configs.policies.lbbd import LBBDConfig


class LBBDMasterProblem:
    """
    Master Problem for Logic-Based Benders Decomposition (LBBD).
    Modelled as an allocation problem across multiple days.
    """

    def __init__(self, config: LBBDConfig, num_customers: int):
        self.config = config
        self.num_customers = num_customers
        self.num_nodes = num_customers + 1
        self.customers = list(range(1, self.num_nodes))

        self.model: gp.Model
        self._z_vars: Dict[Tuple[int, int], Any] = {}  # (node, day)
        self._q_vars: Dict[Tuple[int, int], Any] = {}  # (node, day)
        self._w_vars: Dict[Tuple[int, int], Any] = {}  # (node, day)
        self._of_vars: Dict[Tuple[int, int], Any] = {}  # (node, day)
        self._theta_vars: Dict[int, Any] = {}  # (day)

        self._benders_cuts_count = 0

    def build(self, initial_wastes: Dict[int, float], distance_matrix: np.ndarray):
        """Builds the initial allocation model."""
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is required for LBBD Master Problem.")

        self.model = gp.Model("LBBD_Master")
        self.model.Params.OutputFlag = 0
        self.model.Params.MIPGap = self.config.mip_gap

        # 1. Variables
        for d in range(1, self.config.num_days + 1):
            self._theta_vars[d] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"theta_d{d}")
            for i in self.customers:
                self._z_vars[i, d] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{d}")
                self._q_vars[i, d] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_{i}_{d}")
                self._w_vars[i, d] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"w_{i}_{d}")
                self._of_vars[i, d] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"of_{i}_{d}")

        # 2. Constraints
        for d in range(1, self.config.num_days + 1):
            # Capacity
            self.model.addConstr(gp.quicksum(self._q_vars[i, d] for i in self.customers) <= 1.0, name=f"cap_d{d}")

            for i in self.customers:
                # Inventory Balance
                if d == 1:
                    w_prev = initial_wastes.get(i, 0.0)
                    q_prev = 0.0
                else:
                    w_prev = self._w_vars[i, d - 1]
                    q_prev = self._q_vars[i, d - 1]

                # Simplified multi-day increment (Expected value used in Master)
                increment = self.config.mean_increment
                self.model.addConstr(self._w_vars[i, d] == w_prev - q_prev + increment, name=f"inv_bal_{i}_{d}")

                # Collection logic
                self.model.addConstr(self._q_vars[i, d] <= self._w_vars[i, d], name=f"q_max_{i}_{d}")
                self.model.addConstr(self._q_vars[i, d] <= 2.0 * self._z_vars[i, d], name=f"q_z_{i}_{d}")

                # Overflow
                self.model.addConstr(self._of_vars[i, d] >= self._w_vars[i, d] - 1.0, name=f"of_def_{i}_{d}")

        # 3. Objective
        profit = gp.quicksum(
            self.config.waste_weight * self.config.num_scenarios * self._q_vars[i, d]  # Corrected scale if needed
            - self.config.overflow_penalty * self._of_vars[i, d]
            for i in self.customers
            for d in range(1, self.config.num_days + 1)
        )
        travel_cost = gp.quicksum(
            self.config.cost_weight * self._theta_vars[d] for d in range(1, self.config.num_days + 1)
        )

        # Scaling adjustment: the revenue is per day, travel cost is per day.
        # ST-EF used an extensive form with probabilities.
        # Here we maximize expected profit.
        self.model.setObjective(profit - travel_cost, GRB.MAXIMIZE)
        self.model.update()

    def add_nogood_cut(self, day: int, assigned_nodes: List[int]):
        """Adds a cut preventing this set of nodes from being assigned together on this day."""
        subset = [i for i in assigned_nodes if i != 0]
        if not subset:
            return

        self._benders_cuts_count += 1
        self.model.addConstr(
            gp.quicksum(self._z_vars[i, day] for i in subset) <= len(subset) - 1,
            name=f"nogood_{day}_{self._benders_cuts_count}",
        )
        self.model.update()

    def add_optimality_cut(self, day: int, assigned_nodes: List[int], distance: float):
        """
        Adds a logic-based optimality cut: theta >= Dist * (1 - sum(1-z_i) - sum(z_j)).
        This is a standard LBBD optimality cut for routing.
        """
        subset = [i for i in assigned_nodes if i != 0]
        others = [i for i in self.customers if i not in subset]

        self._benders_cuts_count += 1
        # theta >= Dist * (1 - sum_{i in S}(1 - z_i) - sum_{j not in S}(z_j))
        # Simplified: theta >= Dist - Dist * (sum_{i in S}(1 - z_i) + sum_{j not in S}(z_j))
        cut_expr = distance * (
            1
            - gp.quicksum(1 - self._z_vars[i, day] for i in subset)
            - gp.quicksum(self._z_vars[j, day] for j in others)
        )
        self.model.addConstr(self._theta_vars[day] >= cut_expr, name=f"opt_cut_{day}_{self._benders_cuts_count}")
        self.model.update()

    def solve(self) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
        """Solves the master problem and returns assignments and theta values."""
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL and self.model.SolCount == 0:
            return {}, {}

        assignments = {}
        thetas = {}
        for d in range(1, self.config.num_days + 1):
            assignments[d] = [i for i in self.customers if self._z_vars[i, d].X > 0.5]
            thetas[d] = self._theta_vars[d].X

        return assignments, thetas
