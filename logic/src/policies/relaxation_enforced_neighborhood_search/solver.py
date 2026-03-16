"""
Relaxation Enforced Neighborhood Search (RENS) matheuristic solver.

RENS is a primal heuristic for Mixed Integer Programming (MIP) that exploits
the information provided by the continuous relaxation (LP) of the problem
to identify a promising neighborhood.

Algorithm Overview:
    1. Continuous Relaxation: Solve the LP relaxation (binary variables -> [0, 1]).
    2. Fix and Restrict:
        a. For binary variables x_j that are already integer in the LP solution:
           Fix x_j = x_j_LP (either 0 or 1).
        b. For binary variables x_j that are fractional in the LP solution:
           Transform to binary x_j in {0, 1}.
    3. Solve Sub-MIP: Solve the resulting restricted MIP, which is typically
       much smaller and faster to solve than the original problem.
    4. Reconstruction: Decode the best integer variables back into a feasible tour.

Advantages:
    - Extremely effective when the LP relaxation provides a good "hint"
      about the integer structure.
    - Can find high-quality solutions much faster than solving the full MIP.
    - Standard component in modern MILP solvers like SCIP and Gurobi.

Reference:
    Berthold, T. (2009). "Rens - The relaxation enforced neighborhood search".
    ZIB-Report 09-11.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.policies.base.base_matheuristic import BaseMatheuristicSolver
from logic.src.tracking.viz_mixin import PolicyStateRecorder


class RENSSolver(BaseMatheuristicSolver):
    """
    Core implementation of the Relaxation Enforced Neighborhood Search (RENS).

    This class handles the transformation of a routing problem into an LP-based
    sub-MIP, fixes variables based on the LP relaxation, and reconstructs
    the final solution.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        values: Dict[str, Any],
        seed: int = 42,
        mandatory_nodes: Optional[List[int]] = None,
        env: Optional[gp.Env] = None,
        recorder: Optional[PolicyStateRecorder] = None,
    ):
        """
        Initialize the RENS solver.

        Args:
            dist_matrix: Symmetric distance matrix.
            wastes: Dictionary mapping node indices to waste quantities.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            values: Configuration dictionary containing search parameters.
            seed: Random seed for Gurobi.
            mandatory_nodes: Nodes that MUST be visited.
            env: Gurobi environment shared across solver calls.
            recorder: Optional telemetry recorder for visualization.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.values = values
        self.seed = seed
        self.mandatory_nodes = mandatory_nodes or []
        self.env = env
        self.recorder = recorder

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[int], float, float]:
        """
        Execute the RENS matheuristic optimization loop.

        Orchestrates the LP relaxation phase, variable fixing, and sub-MIP solve.

        Algorithmic Steps:
            1. Setup base VRP model with continuous [0,1] variables (LP relaxation).
            2. Optimize the relaxation to find the fractional boundary.
            3. Apply RENS restrictions (fix integer variables, convert fractional
               to binary).
            4. Solve the restricted sub-MIP with defined gap and time limits.
            5. Reconstruct the tour sequence from active binary edges.

        Args:
            initial_solution: Ignored for RENS as it starts from LP relaxation.

        Returns:
            Tuple of (full_tour, objective_value, travel_cost).
        """
        lp_time_limit = self.values.get("lp_time_limit", 10.0)
        time_limit = self.values.get("time_limit", 60.0)
        mip_gap = self.values.get("mip_gap", 0.01)

        # 1. Setup and Relaxation Solve
        model, x, y = self._setup_rens_model()

        model.setParam("TimeLimit", lp_time_limit)
        model.optimize()

        # Check if a relaxation solution was found
        if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or model.SolCount == 0:
            return [0, 0], 0.0, 0.0

        # 2. Variable Fixing and Sub-MIP Solve
        self._apply_restrictions(model, x, y)

        rem_time = max(1.0, time_limit - model.Runtime)
        model.setParam("TimeLimit", rem_time)
        model.setParam("MIPGap", mip_gap)
        # Focus heuristic search as RENS is primarily a primal start heuristic
        model.setParam("MIPFocus", 1)
        model.optimize()

        # 3. Extraction/Reconstruction
        if model.SolCount == 0:
            return [0, 0], 0.0, 0.0

        tour, cost = self._reconstruct_tour(x)

        if self.recorder:
            self.recorder.record(engine="rens", solved=1, obj_val=model.ObjVal, cost=cost)

        return tour, float(model.ObjVal), float(cost)

    def _setup_rens_model(self) -> Tuple[gp.Model, Dict[Tuple[int, int], gp.Var], Dict[int, gp.Var]]:
        """
        Build the base VRP model for RENS with continuous variables.

        Mathematical Formulation (LP Relaxation):
            Max sum(p_i * y_i) - sum(c_ij * x_ij)
            s.t. sum(x_ij, j) == y_i (Flow Out)
                 sum(x_ji, j) == y_i (Flow In)
                 u_j >= u_i + d_j - Q(1 - x_ij) (Subtour elimination/Capacity)
                 y_i = 1 for i in Mandatory

        Returns:
            Tuple of (Gurobi Model, x_edge_vars, y_node_vars).
        """
        num_nodes = len(self.dist_matrix)
        nodes = list(range(num_nodes))
        customers = list(range(1, num_nodes))
        m_set = set(self.mandatory_nodes)

        model = gp.Model("RENS_VRPP", env=self.env) if self.env else gp.Model("RENS_VRPP")
        model.setParam("OutputFlag", 0)
        model.setParam("Seed", self.seed)

        # 1. VARIABLES
        x = {}
        for i in nodes:
            for j in nodes:
                if i != j:
                    x[i, j] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

        y = {}
        for i in customers:
            y[i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{i}")

        u = {}
        for i in customers:
            u[i] = model.addVar(lb=self.wastes.get(i, 0), ub=self.capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}")

        # 2. OBJECTIVE
        travel_cost = quicksum(self.dist_matrix[i][j] * self.C * x[i, j] for i in nodes for j in nodes if i != j)
        revenue = quicksum(self.wastes.get(i, 0) * self.R * y[i] for i in customers)
        model.setObjective(revenue - travel_cost, GRB.MAXIMIZE)

        # 3. CONSTRAINTS
        for i in customers:
            model.addConstr(quicksum(x[i, j] for j in nodes if i != j) == y[i], name=f"out_{i}")
            model.addConstr(quicksum(x[j, i] for j in nodes if i != j) == y[i], name=f"in_{i}")

        model.addConstr(quicksum(x[0, j] for j in customers) <= 1, name="depot_out")
        model.addConstr(
            quicksum(x[j, 0] for j in customers) == quicksum(x[0, j] for j in customers), name="depot_balance"
        )

        for i in customers:
            for j in customers:
                if i != j:
                    dj = self.wastes.get(j, 0)
                    model.addConstr(u[j] >= u[i] + dj - self.capacity * (1 - x[i, j]), name=f"mtz_{i}_{j}")

        for i in m_set:
            model.addConstr(y[i] == 1, name=f"mandatory_{i}")

        return model, x, y

    def _apply_restrictions(self, model: gp.Model, x: Dict[Tuple[int, int], gp.Var], y: Dict[int, gp.Var]) -> None:
        """
        Apply RENS rounding restrictions to the model variables.

        This is the core of the RENS heuristic.
        - Variables with Val < 1e-6 in LP are fixed to 0.
        - Variables with Val > 1 - 1e-6 in LP are fixed to 1.
        - All variables are then converted to BINARY for the subsequent sub-MIP solve.
        """
        for var in x.values():
            val = var.X
            if val < 1e-6:
                var.lb, var.ub = 0, 0
            elif val > 1.0 - 1e-6:
                var.lb, var.ub = 1, 1
            var.vtype = GRB.BINARY

        for var in y.values():
            val = var.X
            if val < 1e-6:
                var.lb, var.ub = 0, 0
            elif val > 1.0 - 1e-6:
                var.lb, var.ub = 1, 1
            var.vtype = GRB.BINARY

    def _reconstruct_tour(self, x: Dict[Tuple[int, int], gp.Var]) -> Tuple[List[int], float]:
        """
        Reconstruct a routing tour from selected binary variables.

        Traverses the active edges (X > 0.5) to build a sequential node list.

        Args:
            x: Dictionary of edge variables from the optimized model.

        Returns:
            Tuple of (node_sequence, travel_distance).
        """
        active_edges = [edge for edge, var in x.items() if var.X > 0.5]
        adj: Dict[int, List[int]] = {i: [] for i in range(len(self.dist_matrix))}
        for i, j in active_edges:
            adj[i].append(j)

        full_tour, current = [0], 0
        visited: Set[int] = {0}

        while adj.get(current):
            nx_node = adj[current].pop(0)
            if nx_node in visited and nx_node != 0:
                # Prevent sub-cycle corruption in reconstruction
                break
            full_tour.append(nx_node)
            visited.add(nx_node)
            current = nx_node
            if current == 0:
                break

        if full_tour[-1] != 0:
            full_tour.append(0)

        cost = sum(self.dist_matrix[i][j] for i, j in active_edges)
        return full_tour, float(cost)


def run_rens_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    **kwargs: Any,
) -> Tuple[List[int], float, float]:
    """
    Compatibility wrapper for functional calls to the RENS engine.

    Args:
        **kwargs: Combined configuration for solver and search limits.

    Returns:
        Standard tuple (tour, profit, cost).
    """
    solver = RENSSolver(
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        values=kwargs,
        seed=kwargs.get("seed", 42),
        mandatory_nodes=mandatory_nodes,
        env=kwargs.get("env"),
        recorder=kwargs.get("recorder"),
    )
    return solver.solve()
